"""infrastructure: reconciler wake + failure backoff (marker: infrastructure).

Two independent stacks: one with RECONCILE_INTERVAL_S=3600 proving intent
submission converges via the event-driven wake (not the tick), and the
standard stack for the CREATE-failure backoff test.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from conftest import (
    MANAGEMENT_API_KEY,
    _build_stack,
    _ensure_model_registered,
)
from fixtures.helpers import wait_for
from fixtures.intents import create_intent, get_intent, wait_intent_ready

pytestmark = pytest.mark.infrastructure


@pytest.fixture(scope="module")
def wake_stack(
    db_env,
    stub_harbor,
    stub_model_artifact,
    alembic_data_repo,
    alembic_solar_control,
    tmp_path_factory,
):
    """A full stack whose reconciler only acts on wake() events (interval 3600s)."""
    tmp_root = tmp_path_factory.mktemp("wake-stack")
    harbor_ref, _files = stub_model_artifact
    stack = asyncio.run(
        _build_stack(
            db_env, stub_harbor, harbor_ref, tmp_root=tmp_root,
            reconcile_interval_s=3600.0,
        )
    )
    try:
        asyncio.run(_ensure_model_registered(stack))
        yield stack
    finally:
        for svc in (stack.host_b, stack.host_a, stack.control, stack.data_repo):
            if svc is not None:
                svc.terminate()


async def test_intent_create_wakes_reconciler(wake_stack, clean_state):
    """With a 3600s tick interval, intent POST still converges promptly."""
    import httpx

    async with httpx.AsyncClient(
        base_url=wake_stack.control_url,
        headers={"X-API-Key": MANAGEMENT_API_KEY},
        timeout=15.0,
    ) as http_control:
        intent = await create_intent(
            http_control, alias=f"wake-{uuid.uuid4().hex[:8]}"
        )
        # Would take 3600s on the tick alone — wake() must drive it.
        ready = await wait_intent_ready(http_control, intent["id"], timeout=60.0)
        assert ready["status"]["ready_replicas"] == 1
        assert ready["status"]["reconcile"] == "succeeded"


async def test_failed_create_backoff(stack, clean_state):
    """CREATE fails (data-repo down) -> last_error recorded; recovery on next tick."""
    import httpx

    async with httpx.AsyncClient(
        base_url=stack.control_url,
        headers={"X-API-Key": MANAGEMENT_API_KEY},
        timeout=15.0,
    ) as http_control:
        # Kill data-repo: the reconciler's resolve step fails deterministically.
        stack.data_repo.terminate()

        intent = await create_intent(
            http_control, alias=f"backoff-{uuid.uuid4().hex[:8]}"
        )

        # The create attempt fails -> last_error / reconcile failed.
        await wait_for(
            lambda: _failed_or_erroring(http_control, intent["id"]),
            timeout=120.0,
            interval=0.5,
            description="reconcile failure recorded (backoff)",
        )

        # Restore data-repo; the next tick's CREATE succeeds -> ready.
        await stack.respawn_data_repo()
        ready = await wait_intent_ready(http_control, intent["id"], timeout=120.0)
        assert ready["status"]["ready_replicas"] == 1
        assert ready["status"]["phase"] == "ready"


async def _failed_or_erroring(http_control, intent_id: str) -> bool:
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    status = intent["status"]
    return bool(status.get("last_error")) or status.get("reconcile") == "failed"

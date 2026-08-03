"""infrastructure: reconciler wake (marker: infrastructure).

A 3600s-interval stack proving intent submission converges via the
event-driven wake (not the tick). The CREATE-failure backoff test lives in
test_failed_create_backoff.py (its own module) — a second live control stack
in this module would cross-reconcile through the shared hosts table / Redis
and stall that test's backoff window.
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
from fixtures.intents import create_intent, wait_intent_ready

pytestmark = pytest.mark.infrastructure


@pytest.fixture(scope="module")
def wake_stack(
    stack,
    db_env,
    stub_harbor,
    stub_model_artifact,
    alembic_data_repo,
    alembic_solar_control,
    tmp_path_factory,
):
    """A full stack whose reconciler only acts on wake() events (interval 3600s).

    Reordered to run last (conftest.pytest_collection_modifyitems). This
    module stops the session control and truncates the shared hosts table
    when building, so nothing may run after it; stopping the control first
    also prevents its reconciler from cross-acting on this module's intents
    through the shared Postgres/Redis.
    """
    # Stop the session stack's reconciler: while it is alive it resolves
    # this module's hosts (same names) and would race the wake control on
    # every intent (the two-controls-one-db cross-talk bug).
    if stack.control is not None:
        stack.control.terminate()
    tmp_root = tmp_path_factory.mktemp("wake-stack")
    harbor_ref, _files = stub_model_artifact
    stack = asyncio.run(
        _build_stack(
            db_env,
            stub_harbor,
            harbor_ref,
            tmp_root=tmp_root,
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
        intent = await create_intent(http_control, alias=f"wake-{uuid.uuid4().hex[:8]}")
        # Would take 3600s on the tick alone — wake() must drive it.
        ready = await wait_intent_ready(http_control, intent["id"], timeout=60.0)
        assert ready["status"]["ready_replicas"] == 1
        assert ready["status"]["reconcile"] == "succeeded"

"""infrastructure: CREATE-failure backoff (marker: infrastructure).

Lives in its own module (not test_reconciler_wake.py) on purpose: the wake
test's stack (3600s interval) must be torn down before this test's stack
builds. Two live control stacks sharing the session Postgres/Redis
cross-reconcile — the module build truncates the shared hosts table, so the
wake control's "host-a" resolves to this stack's host and its wake-driven
passes (host events) create a spurious instance for this intent; the module
control then saw that instance as failed and took RECREATE, whose ~140s
start-call stall ate the whole 150s backoff window (the last_error landed
1.4s after the deadline).
"""

from __future__ import annotations

import uuid

import pytest

from conftest import MANAGEMENT_API_KEY
from fixtures.helpers import wait_for
from fixtures.intents import create_intent, get_intent, wait_intent_ready

pytestmark = pytest.mark.infrastructure


async def test_failed_create_backoff(stack, clean_state):
    """CREATE fails (data-repo down) -> last_error recorded; recovery on next tick."""
    import httpx

    async with httpx.AsyncClient(
        base_url=stack.control_url,
        headers={"X-API-Key": MANAGEMENT_API_KEY},
        timeout=15.0,
    ) as http_control:
        # Kill data-repo: the reconciler's resolve step fails deterministically.
        # SIGKILL (not terminate) — a graceful shutdown keeps the port bound
        # and the resolve's TCP connect succeeds while the dying server never
        # answers, hanging the reconciler's action ~86-150s past its timeouts.
        stack.data_repo.kill()

        intent = await create_intent(
            http_control, alias=f"backoff-{uuid.uuid4().hex[:8]}"
        )

        # The create attempt fails -> last_error / reconcile failed.
        await wait_for(
            lambda: _failed_or_erroring(http_control, intent["id"]),
            timeout=60.0,
            interval=0.5,
            description="reconcile failure recorded (backoff)",
        )

        # Restore data-repo; the next tick's CREATE succeeds -> ready.
        await stack.respawn_data_repo()
        ready = await wait_intent_ready(http_control, intent["id"], timeout=15.0)
        assert ready["status"]["ready_replicas"] == 1
        assert ready["status"]["phase"] == "ready"


async def _failed_or_erroring(http_control, intent_id: str) -> bool:
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    status = intent["status"]
    return bool(status.get("last_error")) or status.get("reconcile") == "failed"

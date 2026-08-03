"""intent_path: scaling + drift handling (marker: intent_path).

RECREATE restarts drifted managed instances in place (no /stop spam — the
D-017 regression guard); surplus replicas are stopped+deleted; shortfall
degrades the intent until capacity returns.
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.helpers import wait_for
from fixtures.intents import (
    create_intent,
    get_intent,
    replica_hosts,
    replica_states,
    wait_intent_ready,
)
from fixtures.seed import count_host_requests, update_intent_in_db

pytestmark = pytest.mark.intent_path


def _alias(prefix: str = "scale") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _first_replica(intent: dict) -> tuple[str, str]:
    """Return (instance_id, host_id) of the first replica."""
    replica = intent["status"]["replica_set"][0]
    return replica["instance_id"], replica["host_id"]


async def _host_letter(http_control, host_id: str) -> str:
    hosts = (await http_control.get("/api/hosts")).json()
    row = next(h for h in hosts if h["id"] == host_id)
    return row["name"].split("-")[-1]


async def test_drift_stopped_instance_recreated(http_control, stack, clean_state):
    """Managed instance stopped out-of-band -> RECREATE restarts it, no stop spam."""
    intent = await create_intent(http_control, alias=_alias())
    ready = await wait_intent_ready(http_control, intent["id"])
    instance_id, host_id = await _first_replica(ready)
    host_letter = await _host_letter(http_control, host_id)

    stops_before = count_host_requests(stack, host_letter, "POST", "/instances/")
    starts_before = count_host_requests(stack, host_letter, "POST", "/instances/")

    # Stop the managed instance directly on the host (drift).
    resp = await http_control.post(f"/api/hosts/{host_id}/instances/{instance_id}/stop")
    assert resp.status_code == 200, resp.text

    # Reconciler RECREATE: restarts in place. Poll for running again.
    await wait_for(
        lambda: _replica_running(http_control, intent["id"], instance_id),
        timeout=15.0,
        interval=0.5,
        description=f"replica {instance_id} recreated/running",
    )

    # D-017 guard: exactly ONE start followed the drift, and the only stop
    # is our manual one (no reconciler /stop spam).
    stops_after = count_host_requests(stack, host_letter, "POST", "/instances/")
    starts_after = count_host_requests(stack, host_letter, "POST", "/instances/")
    assert stops_after - stops_before == 1, (
        f"reconciler issued extra stops ({stops_after - stops_before}); "
        "RECREATE must restart, not stop"
    )
    assert (
        starts_after - starts_before == 1
    ), f"expected exactly 1 start after drift, got {starts_after - starts_before}"

    intent = await get_intent(http_control, intent["id"])
    assert intent is not None
    assert intent["status"]["phase"] == "ready"


async def _replica_running(http_control, intent_id: str, instance_id: str) -> bool:
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    for replica in intent["status"].get("replica_set", []):
        if replica.get("instance_id") == instance_id:
            return replica.get("state") == "running"
    return False


async def test_scale_down_surplus_stop(http_control, stack, clean_state):
    """replicas 2 -> 1 via DB update: surplus instance stopped+deleted."""
    intent = await create_intent(http_control, alias=_alias(), replicas=2)
    ready = await wait_intent_ready(http_control, intent["id"], ready_replicas=2)
    states = replica_states(ready)
    assert len(states) == 2

    update_intent_in_db(stack.db_env["control_db"], intent["id"], replicas=1)

    # Converges to a single managed replica, still ready.
    await wait_for(
        lambda: _observed_is(http_control, intent["id"], 1),
        timeout=15.0,
        interval=0.5,
        description="intent scaled down to 1 replica",
    )
    final = await get_intent(http_control, intent["id"])
    assert final is not None
    assert final["status"]["phase"] == "ready"
    assert final["status"]["ready_replicas"] == 1
    states = replica_states(final)
    assert len(states) == 1
    assert list(states.values()) == ["running"]

    # The surviving replica is one of the originals (surplus stopped first).
    assert list(states.keys())[0] in replica_states(ready)


async def _observed_is(http_control, intent_id: str, n: int) -> bool:
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    return intent["status"]["observed_replicas"] == n


async def _placed(http_control, intent_id: str, n: int) -> bool:
    """True once the intent's replica_set spans ``n`` distinct hosts."""
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    hosts = {r["host_id"] for r in intent["status"]["replica_set"]}
    return len(hosts) == n


async def test_shortfall_degraded(http_control, stack, clean_state):
    """replicas: 3 with 2 hosts -> degraded + shortfall; 3rd host fills it."""
    intent = await create_intent(http_control, alias=_alias(), replicas=3)

    async def degraded_with_shortfall() -> bool:
        current = await get_intent(http_control, intent["id"])
        if current is None:
            return False
        status = current["status"]
        return (
            status["phase"] == "degraded"
            and status["shortfall"] >= 1
            and status["ready_replicas"] >= 1
        )

    await wait_for(
        degraded_with_shortfall,
        timeout=15.0,
        interval=0.5,
        description="intent degraded with shortfall",
    )

    # The reconciler executes one action per tick (§8.1), so the second
    # replica lands a few ticks after the first — wait for full placement
    # (2 replicas on 2 hosts) rather than asserting immediately.
    await wait_for(
        lambda: _placed(http_control, intent["id"], 2),
        timeout=15.0,
        interval=0.5,
        description="two replicas placed on two hosts",
    )
    current = await get_intent(http_control, intent["id"])
    assert current is not None
    assert len(replica_hosts(current)) == 2  # one replica per host
    assert current["status"]["observed_replicas"] == 2

    # A 3rd host comes online -> reconciler fills the shortfall to ready.
    await stack.spawn_extra_host("c")
    await wait_intent_ready(http_control, intent["id"], ready_replicas=3)
    final = await get_intent(http_control, intent["id"])
    assert final is not None
    assert len(replica_hosts(final)) == 3
    assert final["status"]["shortfall"] == 0

    # Restore the 2-host topology: the migration tests that follow assume
    # exactly two hosts (their "no target" displacement scenario must have
    # no third host for the MIGRATE target search to find).
    stack.remove_extra_host("c")

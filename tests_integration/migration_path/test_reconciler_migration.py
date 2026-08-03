"""migration_path: reconciler interaction after migration (marker: migration_path).

The D-017 regression: a migration-created managed target is auto-started by
the reconciler (RECREATE) and the disowned source is never touched again;
a failing RECREATE deletes the replica and the next tick's CREATE re-places
it (with backoff recorded).
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.helpers import wait_for
from fixtures.intents import create_intent, get_intent, wait_intent_ready
from fixtures.seed import count_host_requests

pytestmark = pytest.mark.migration_path


async def _hosts(http_control) -> dict[str, dict]:
    hosts = (await http_control.get("/api/hosts")).json()
    return {h["name"]: h for h in hosts}


async def _host_instances(http_control, host_id: str) -> list[dict]:
    resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _find_instance(http_control, host_id: str, instance_id: str) -> dict | None:
    for inst in await _host_instances(http_control, host_id):
        if inst.get("id") == instance_id:
            return inst
    return None


async def _registry_visible(http_control, alias: str) -> bool:
    resp = await http_control.get("/v1/models")
    if resp.status_code != 200:
        return False
    body = resp.json()
    names = {m.get("name") for m in body.get("models", [])} | {
        m.get("id") for m in body.get("data", [])
    }
    return alias in names


async def test_migrated_target_auto_started(http_control, stack, clean_state):
    """The full D-017 end state after migrating a managed instance.

    (1) source stopped; (2) source disowned — never surplus-STOPped or
    re-started; (3) target created stopped WITH markers; (4) within a few
    reconcile ticks the target is auto-started (RECREATE) and registered in
    the gateway registry; (5) intent ready_replicas intact.
    """
    hosts = await _hosts(http_control)
    alias = f"d017-{uuid.uuid4().hex[:8]}"

    intent = await create_intent(http_control, alias=alias)
    ready = await wait_intent_ready(http_control, intent["id"])
    replica = ready["status"]["replica_set"][0]
    source_instance_id = replica["instance_id"]
    source_host_id = replica["host_id"]
    source_host_name = next(n for n, h in hosts.items() if h["id"] == source_host_id)
    src_letter = source_host_name[-1]
    dst_id = next(h["id"] for h in hosts.values() if h["id"] != source_host_id)

    stops_before = count_host_requests(stack, src_letter, "POST", "/instances/")

    resp = await http_control.post(
        "/api/instances/migrate",
        json={
            "instance_id": source_instance_id,
            "source_host_id": source_host_id,
            "target_host_id": dst_id,
            # Intent instances default to priority=production.
            "allow_production": True,
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["status"] == "completed", result
    target_instance_id = result["target_instance_id"]

    # (4) The reconciler auto-starts the managed target (RECREATE) and it
    # lands in the gateway registry.
    await wait_for(
        lambda: _instance_running(http_control, dst_id, target_instance_id),
        timeout=15.0,
        interval=0.5,
        description="migration target auto-started by reconciler",
    )
    await wait_for(
        lambda: _registry_visible(http_control, alias),
        timeout=60.0,
        interval=0.5,
        description="alias registered after migration",
    )

    # (1)+(2) Source: stopped, disowned, never re-started or surplus-stopped.
    source = await _find_instance(http_control, source_host_id, source_instance_id)
    assert source is not None, "source instance vanished (surplus-STOP deleted it)"
    assert source["status"] == "stopped"
    assert source.get("managed_by") in (None, "")
    assert source.get("intent_id") in (None, "")

    # The only stop on the source host is migration's own stop_source (no
    # /stop spam, no extra starts of the source).
    stops_after = count_host_requests(stack, src_letter, "POST", "/instances/")
    assert stops_after - stops_before == 1, (
        f"unexpected extra stop calls on source host " f"({stops_after - stops_before})"
    )

    # (3) The target carries the markers top-level.
    target = await _find_instance(http_control, dst_id, target_instance_id)
    assert target is not None
    assert target.get("managed_by") == "intent"
    assert target.get("intent_id") == intent["id"]

    # (5) Intent fully ready again with exactly one managed replica.
    final = await wait_intent_ready(http_control, intent["id"])
    replicas = final["status"]["replica_set"]
    assert len(replicas) == 1
    assert replicas[0]["instance_id"] == target_instance_id
    assert replicas[0]["state"] == "running"
    assert final["status"]["ready_replicas"] == 1


async def _instance_running(http_control, host_id: str, instance_id: str) -> bool:
    inst = await _find_instance(http_control, host_id, instance_id)
    return inst is not None and inst.get("status") == "running"


async def test_recreate_failure_records_backoff_and_recovers(
    http_control, stack, clean_state
):
    """RECREATE start failure -> backoff recorded; recovery restarts in place.

    Failure injection: the host's API key is rotated in the DB (the WS
    channel stays up; only control's HTTP calls to the host start failing
    with 403). The reconciler's RECREATE start fails fast -> replica kept,
    backoff recorded. Restoring the key lets the next RECREATE restart the
    instance in place (§8.2).
    """
    from fixtures.constants import HOST_A_API_KEY, HOST_B_API_KEY
    from fixtures.seed import update_host_api_key

    hosts = await _hosts(http_control)
    alias = f"recreate-{uuid.uuid4().hex[:8]}"

    intent = await create_intent(http_control, alias=alias)
    ready = await wait_intent_ready(http_control, intent["id"])
    replica = ready["status"]["replica_set"][0]
    instance_id = replica["instance_id"]
    host_id = replica["host_id"]
    host_name = next(n for n, h in hosts.items() if h["id"] == host_id)
    real_key = HOST_A_API_KEY if host_name == "host-a" else HOST_B_API_KEY

    # Drift: stop the instance.
    resp = await http_control.post(f"/api/hosts/{host_id}/instances/{instance_id}/stop")
    assert resp.status_code == 200, resp.text

    # Break the start: rotate the host's API key (HTTP calls now 403).
    update_host_api_key(stack.db_env["control_db"], host_id, "definitely-wrong-key")

    # The reconciler's RECREATE start fails -> last_error / backoff recorded.
    await wait_for(
        lambda: _last_error_or_failed(http_control, intent["id"]),
        timeout=15.0,
        interval=0.5,
        description="reconcile failure recorded (backoff)",
    )

    # Restore the key; the next RECREATE restarts the instance in place.
    update_host_api_key(stack.db_env["control_db"], host_id, real_key)
    final = await wait_intent_ready(http_control, intent["id"], timeout=15.0)
    replicas = final["status"]["replica_set"]
    assert len(replicas) == 1
    assert replicas[0]["instance_id"] == instance_id, "restart-in-place expected"
    assert replicas[0]["state"] == "running"


async def _last_error_or_failed(http_control, intent_id: str) -> bool:
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    status = intent["status"]
    return bool(status.get("last_error")) or status.get("reconcile") == "failed"

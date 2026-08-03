"""migration_path: migration guards (marker: migration_path).

Same-host 422, one-replica-per-host 409, production safeguard, active
training block, and the no-target ephemeral/staging fallback (§8.5).
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import BACKEND_CLASSIFICATION, MODEL_SOURCE_URI
from fixtures.helpers import wait_for
from fixtures.intents import create_intent, get_intent, wait_intent_ready

pytestmark = pytest.mark.migration_path


def _instance_payload(alias: str, priority: str = "staging") -> dict:
    return {
        "config": {
            "backend_type": BACKEND_CLASSIFICATION["backend_type"],
            "alias": alias,
            "model_source": MODEL_SOURCE_URI,
            "device": "cpu",
            "dtype": "float32",
            "max_length": 128,
            "labels": ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4"],
        },
        "priority": priority,
    }


async def _hosts(http_control) -> dict[str, dict]:
    hosts = (await http_control.get("/api/hosts")).json()
    return {h["name"]: h for h in hosts}


async def _host_instances(http_control, host_id: str) -> list[dict]:
    resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _create_manual(
    http_control, host_id: str, alias: str, priority: str = "staging"
) -> str:
    resp = await http_control.post(
        f"/api/hosts/{host_id}/instances", json=_instance_payload(alias, priority)
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["instance"]["id"]


async def _migrate(
    http_control,
    instance_id: str,
    source_host_id: str,
    target_host_id: str,
    allow_production: bool = False,
):
    return await http_control.post(
        "/api/instances/migrate",
        json={
            "instance_id": instance_id,
            "source_host_id": source_host_id,
            "target_host_id": target_host_id,
            "allow_production": allow_production,
        },
    )


async def test_same_host_422(http_control, clean_state):
    """source == target -> 422."""
    hosts = await _hosts(http_control)
    host = hosts["host-a"]
    instance_id = await _create_manual(
        http_control, host["id"], f"same-{uuid.uuid4().hex[:6]}"
    )

    resp = await _migrate(http_control, instance_id, host["id"], host["id"])
    assert resp.status_code == 422, resp.text


async def test_one_replica_per_host(http_control, clean_state):
    """Target already serving the alias -> rejected (409)."""
    hosts = await _hosts(http_control)
    src, dst = hosts["host-a"], hosts["host-b"]
    alias = f"onerep-{uuid.uuid4().hex[:6]}"
    src_instance = await _create_manual(http_control, src["id"], alias)
    await _create_manual(http_control, dst["id"], alias)  # same alias on target

    resp = await _migrate(http_control, src_instance, src["id"], dst["id"])
    assert resp.status_code == 409, resp.text
    assert "already exists" in resp.json()["detail"]


async def test_production_not_migratable(http_control, clean_state):
    """allow_production=False blocks production; True succeeds."""
    hosts = await _hosts(http_control)
    src, dst = hosts["host-a"], hosts["host-b"]
    instance_id = await _create_manual(
        http_control, src["id"], f"prod-{uuid.uuid4().hex[:6]}", priority="production"
    )

    resp = await _migrate(http_control, instance_id, src["id"], dst["id"])
    assert resp.status_code == 422, resp.text
    assert "production" in resp.json()["detail"].lower()

    resp = await _migrate(
        http_control, instance_id, src["id"], dst["id"], allow_production=True
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


async def test_active_training_blocks_migration(http_control, http_host, clean_state):
    """Source host with a running job -> migration refused (409)."""
    hosts = await _hosts(http_control)
    src, dst = hosts["host-a"], hosts["host-b"]
    instance_id = await _create_manual(
        http_control, src["id"], f"train-{uuid.uuid4().hex[:6]}"
    )

    # Submit a job on the source host — it is created in status "running".
    resp = await http_host.post(
        "/jobs",
        json={
            "job_id": f"job-{uuid.uuid4().hex[:8]}",
            "name": "training-block-test",
            "steps": [
                {
                    "name": "train",
                    "image": "imgrepo.damit.hu/supernova/no-such-image:v1",
                }
            ],
        },
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    try:
        resp = await _migrate(http_control, instance_id, src["id"], dst["id"])
        assert resp.status_code == 409, resp.text
        assert "training" in resp.json()["detail"].lower()
    finally:
        await http_host.delete(f"/jobs/{job_id}")


async def test_no_target_ephemeral_stop(http_control, stack, clean_state):
    """Displacement with no target: ephemeral stopped+deleted; staging left in place."""
    hosts = await _hosts(http_control)
    src, dst = hosts["host-a"], hosts["host-b"]
    # Pin to the original two hosts: the session stack may have extra hosts
    # (host-c from the shortfall test), and this scenario needs exactly two
    # eligible hosts for the 3-replica intent to fall short.
    pinned = {"host_allow": [src["id"], dst["id"]]}

    # Low-priority intent fills both hosts (one replica per host).
    low = await create_intent(
        http_control,
        alias=f"low-{uuid.uuid4().hex[:6]}",
        replicas=2,
        priority="ephemeral",
        placement=pinned,
    )
    await wait_intent_ready(http_control, low["id"], ready_replicas=2)
    low_ready = await get_intent(http_control, low["id"])
    assert low_ready is not None
    low_replicas = {
        r["host_id"]: r["instance_id"] for r in low_ready["status"]["replica_set"]
    }
    assert set(low_replicas) == {src["id"], dst["id"]}
    displaced_instance_id = low_replicas[src["id"]]

    # Higher-priority intent with more replicas than eligible hosts ->
    # displacement MIGRATE of the ephemeral replica; no target exists
    # (both hosts serve its alias) -> ephemeral fallback: stop + delete.
    high = await create_intent(
        http_control,
        alias=f"high-{uuid.uuid4().hex[:6]}",
        replicas=3,
        priority="production",
        placement=pinned,
    )

    await wait_for(
        lambda: _degraded_with_shortfall(http_control, high["id"]),
        timeout=15.0,
        interval=0.5,
        description="high-priority intent degraded with shortfall",
    )

    # The ephemeral replica on the source host is stopped and deleted.
    await wait_for(
        lambda: _instance_gone(http_control, src["id"], displaced_instance_id),
        timeout=15.0,
        interval=0.5,
        description="displaced ephemeral replica removed",
    )


async def test_no_target_staging_left_in_place(http_control, stack, clean_state):
    """Same displacement but staging: no target -> instance left in place."""
    hosts = await _hosts(http_control)
    src = hosts["host-a"]
    # Pin to the original two hosts (see test_no_target_ephemeral_stop).
    pinned = {"host_allow": [src["id"], hosts["host-b"]["id"]]}

    low = await create_intent(
        http_control,
        alias=f"stag-{uuid.uuid4().hex[:6]}",
        replicas=2,
        priority="staging",
        placement=pinned,
    )
    await wait_intent_ready(http_control, low["id"], ready_replicas=2)
    low_ready = await get_intent(http_control, low["id"])
    assert low_ready is not None
    low_replicas = {
        r["host_id"]: r["instance_id"] for r in low_ready["status"]["replica_set"]
    }
    displaced_instance_id = low_replicas[src["id"]]

    high = await create_intent(
        http_control,
        alias=f"hi2-{uuid.uuid4().hex[:6]}",
        replicas=3,
        priority="production",
        placement=pinned,
    )
    await wait_for(
        lambda: _degraded_with_shortfall(http_control, high["id"]),
        timeout=15.0,
        interval=0.5,
        description="high-priority intent degraded with shortfall",
    )

    # Staging replica survives on the source host (migrate, don't drop).
    await wait_for(
        lambda: _instance_exists(http_control, src["id"], displaced_instance_id),
        timeout=15.0,
        interval=0.5,
        description="staging replica still present",
    )


async def _degraded_with_shortfall(http_control, intent_id: str) -> bool:
    intent = await get_intent(http_control, intent_id)
    if intent is None:
        return False
    status = intent["status"]
    return status["phase"] == "degraded" and status["shortfall"] >= 1


async def _instance_gone(http_control, host_id: str, instance_id: str) -> bool:
    instances = await _host_instances(http_control, host_id)
    return all(i.get("id") != instance_id for i in instances)


async def _instance_exists(http_control, host_id: str, instance_id: str) -> bool:
    instances = await _host_instances(http_control, host_id)
    return any(i.get("id") == instance_id for i in instances)

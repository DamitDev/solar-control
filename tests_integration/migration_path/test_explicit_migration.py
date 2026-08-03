"""migration_path: explicit migration via POST /api/instances/migrate (S-037)."""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import (
    BACKEND_CLASSIFICATION,
    MODEL_SOURCE_URI,
)
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


async def _wait_status(
    http_control, host_id: str, instance_id: str, status: str
) -> dict:
    """Poll control's host-instance view until the instance reaches status."""

    async def reached() -> bool:
        resp = await http_control.get(f"/api/hosts/{host_id}/instances")
        if resp.status_code != 200:
            return False
        for inst in resp.json():
            if inst.get("id") == instance_id and inst.get("status") == status:
                return True
        return False

    await wait_for(
        reached,
        timeout=15.0,
        interval=0.5,
        description=f"instance {instance_id} {status}",
    )
    resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    assert resp.status_code == 200
    return next(i for i in resp.json() if i.get("id") == instance_id)


async def _host_instances(http_control, host_id: str) -> list[dict]:
    resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_migrate_manual_instance(http_control, stack, clean_state):
    """Manual instance migrates: source stopped, target created with same config."""
    hosts = await _hosts(http_control)
    src, dst = hosts["host-a"], hosts["host-b"]
    alias = f"mig-{uuid.uuid4().hex[:8]}"

    resp = await http_control.post(
        f"/api/hosts/{src['id']}/instances", json=_instance_payload(alias)
    )
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance"]["id"]
    await http_control.post(f"/api/hosts/{src['id']}/instances/{instance_id}/start")
    await _wait_status(http_control, src["id"], instance_id, "running")

    resp = await http_control.post(
        "/api/instances/migrate",
        json={
            "instance_id": instance_id,
            "source_host_id": src["id"],
            "target_host_id": dst["id"],
            "allow_production": False,
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["status"] == "completed", result
    assert result["source_instance_id"] == instance_id
    assert result["target_instance_id"], result
    target_id = result["target_instance_id"]
    step_names = [s["step"] for s in result["steps"]]
    assert "ensure_model" in step_names
    assert all(s["status"] == "ok" for s in result["steps"]), result

    # Model was pulled on the target host first.
    manifest = (stack.models_dir_b / "manifest.json").read_text()
    assert MODEL_SOURCE_URI in manifest

    # Source stopped; target created with the same config.
    src_inst = await _wait_status(http_control, src["id"], instance_id, "stopped")
    assert _cfg(src_inst).get("alias") == alias
    target_inst = next(
        i
        for i in await _host_instances(http_control, dst["id"])
        if i["id"] == target_id
    )
    assert _cfg(target_inst).get("alias") == alias
    assert _cfg(target_inst).get("model_source") == MODEL_SOURCE_URI
    assert _cfg(target_inst).get("backend_type") == "huggingface_classification"


def _cfg(inst: dict) -> dict:
    """Instances in control's view may be flat (WS) or nested (HTTP poll)."""
    return inst.get("config", inst) if isinstance(inst.get("config"), dict) else inst


async def test_migrate_managed_instance_keeps_markers(http_control, stack, clean_state):
    """Migrating an intent-owned instance carries markers to the target."""
    hosts = await _hosts(http_control)
    intent = await create_intent(http_control, alias=f"migr-{uuid.uuid4().hex[:8]}")
    ready = await wait_intent_ready(http_control, intent["id"])
    replica = ready["status"]["replica_set"][0]
    instance_id = replica["instance_id"]
    # Migrate from the replica's actual host to the other one.
    src_id = replica["host_id"]
    dst_id = next(h["id"] for h in hosts.values() if h["id"] != src_id)

    resp = await http_control.post(
        "/api/instances/migrate",
        json={
            "instance_id": instance_id,
            "source_host_id": src_id,
            "target_host_id": dst_id,
            # Intent instances default to priority=production; migrating them
            # requires the explicit opt-in.
            "allow_production": True,
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["status"] == "completed", result
    target_id = result["target_instance_id"]

    target = next(
        i for i in await _host_instances(http_control, dst_id) if i["id"] == target_id
    )
    # Markers preserved top-level on the target (managed by the intent).
    assert target.get("managed_by") == "intent"
    assert target.get("intent_id") == intent["id"]
    assert target.get("priority") == "production"

    # Intent's replica_set now points at the target host (status lags the
    # migration by one reconcile pass).
    async def target_in_replica_set() -> bool:
        final = await get_intent(http_control, intent["id"])
        if final is None:
            return False
        replicas = final["status"]["replica_set"]
        return any(
            r["instance_id"] == target_id and r["host_id"] == dst_id for r in replicas
        )

    await wait_for(
        target_in_replica_set,
        timeout=60.0,
        interval=0.5,
        description="migrated target in intent replica_set",
    )

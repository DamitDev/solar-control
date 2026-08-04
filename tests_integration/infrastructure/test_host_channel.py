"""infrastructure: the WS host channel (marker: infrastructure).

Hosts push ``host_health`` + ``instances_update`` over /ws/host-channel;
control writes these into Redis host_store — exactly what the reconciler
observes. Assert through control's API.
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import (
    BACKEND_CLASSIFICATION,
    MODEL_SOURCE_URI,
)
from fixtures.helpers import wait_for

pytestmark = pytest.mark.infrastructure


def _instance_payload(alias: str) -> dict:
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
        "priority": "staging",
    }


async def test_hosts_connect_and_report_health(http_control, clean_state):
    """Both host subprocesses are connected; health flows into control."""
    resp = await http_control.get("/api/hosts")
    assert resp.status_code == 200, resp.text
    rows = {h["name"]: h for h in resp.json()}
    assert "host-a" in rows and "host-b" in rows
    for name in ("host-a", "host-b"):
        assert rows[name]["status"] == "online", f"{name} not online: {rows[name]}"
        assert rows[name]["last_seen"] is not None, f"{name} has no last_seen"
        assert "inference" in (rows[name].get("roles") or []), f"{name} roles missing"

    # The connected set (WS seam) is non-empty from control's perspective.
    resp = await http_control.get("/api/resources")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("reachable_hosts", 0) >= 2, body


async def test_instances_update_populates_redis(
    http_control, http_host, stack, clean_state
):
    """Instance create/stop on host -> instances_update -> control's view."""
    alias = f"ws-{uuid.uuid4().hex[:8]}"
    # The host refuses repo:// sources until the model is pulled; distribute
    # through control first (real pull from the stub Harbor), then create the
    # instance directly on the host.
    hosts = {h["name"]: h for h in (await http_control.get("/api/hosts")).json()}
    host_a = hosts["host-a"]
    resp = await http_control.post(
        "/api/models/distribute",
        json={"target_host_id": host_a["id"], "source_uri": MODEL_SOURCE_URI},
    )
    assert resp.status_code == 200, resp.text

    # The host only accepts already-resolved model sources on instance
    # creation (repo:// is rejected until pulled). Read the pulled slug from
    # the host's model manifest and pass it as a local:// source.
    resp = await http_host.get("/models")
    assert resp.status_code == 200, resp.text
    entry = next(m for m in resp.json() if m.get("source_uri") == MODEL_SOURCE_URI)
    slug = entry["path"].rstrip("/").split("/")[-1]

    payload = _instance_payload(alias)
    payload["config"]["model_source"] = f"local://{slug}"
    resp = await http_host.post("/instances", json=payload)
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance"]["id"]

    # Control's host-instance view (the WS seam) reflects the new instance.
    async def visible() -> bool:
        r = await http_control.get(f"/api/hosts/{host_a['id']}/instances")
        if r.status_code != 200:
            return False
        return any(i.get("id") == instance_id for i in r.json())

    await wait_for(
        visible,
        timeout=60.0,
        interval=0.5,
        description="instance visible in control's host view",
    )
    resp = await http_control.get(f"/api/hosts/{host_a['id']}/instances")
    inst = next(i for i in resp.json() if i["id"] == instance_id)
    # Control's host view proxies the host's live Instance model: alias is
    # nested under config (the reconciler's flat cache is the WS shape, not
    # the API shape).
    assert inst.get("config", {}).get("alias") == alias
    assert inst.get("status") == "stopped"

    # Stop-away: the stop event updates the cache too.
    await http_host.post(f"/instances/{instance_id}/start")
    await wait_for(
        lambda: _running(http_control, host_a["id"], instance_id),
        timeout=60.0,
        interval=0.5,
        description="instance running in control's view",
    )

    # The finer-detail snapshot lists the instance with its alias (U-004).
    resp = await http_control.get("/api/resources")
    assert resp.status_code == 200, resp.text
    snap = next(h for h in resp.json()["hosts"] if h["host_id"] == host_a["id"])
    assert any(
        inst.get("id") == instance_id and inst.get("alias") == alias
        for inst in snap.get("instances", [])
    )


async def test_resources_snapshot_exposes_reservation_details(
    http_control, http_host, clean_state
):
    """Per-reservation details (owner job_id, status, requested vs actual)
    pass through solar-host -> control -> /api/resources (U-004)."""
    hosts = {h["name"]: h for h in (await http_control.get("/api/hosts")).json()}
    host_a = hosts["host-a"]

    # Baseline: every reachable host entry exposes the finer-detail fields.
    resp = await http_control.get("/api/resources")
    assert resp.status_code == 200, resp.text
    snap = next(h for h in resp.json()["hosts"] if h["host_id"] == host_a["id"])
    assert snap["reachable"] is True
    assert isinstance(snap["instances"], list)
    assert isinstance(snap["reservations"], list)
    for key in (
        "vram_training_used_gb",
        "ram_training_used_gb",
        "disk_training_used_gb",
    ):
        assert key in snap

    # Create a reservation directly on the host (bypasses the S-038
    # coordinator; a pending reservation has no actuals yet).
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    resp = await http_host.post(
        "/resources/reservations",
        json={
            "job_id": job_id,
            "requester": "integration-test",
            "workload_type": "training",
            "vram_gb": 0.0,
            "ram_gb": 1.0,
        },
    )
    assert resp.status_code == 201, resp.text
    reservation = resp.json()

    async def visible() -> bool:
        r = await http_control.get("/api/resources")
        if r.status_code != 200:
            return False
        snap = next(
            (h for h in r.json()["hosts"] if h["host_id"] == host_a["id"]),
            None,
        )
        if not snap:
            return False
        return any(
            x.get("id") == reservation["id"] for x in snap.get("reservations", [])
        )

    await wait_for(
        visible,
        timeout=60.0,
        interval=0.5,
        description="reservation visible in /api/resources",
    )

    resp = await http_control.get("/api/resources")
    snap = next(h for h in resp.json()["hosts"] if h["host_id"] == host_a["id"])
    entry = next(x for x in snap["reservations"] if x["id"] == reservation["id"])
    assert entry["job_id"] == job_id
    assert entry["status"] == "pending"
    assert entry["workload_type"] == "training"
    assert entry["ram_gb"] == 1.0
    assert entry["actual_ram_gb"] is None
    assert snap["reservation_count"] >= 1


async def _running(http_control, host_id: str, instance_id: str) -> bool:
    return await _instance_status(http_control, host_id, instance_id) == "running"


async def _instance_status(http_control, host_id: str, instance_id: str) -> str:
    r = await http_control.get(f"/api/hosts/{host_id}/instances")
    if r.status_code != 200:
        return ""
    for i in r.json():
        if i.get("id") == instance_id:
            return i.get("status", "")
    return ""

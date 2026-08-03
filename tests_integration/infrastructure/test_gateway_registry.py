"""infrastructure: gateway registry alias lifecycle (marker: infrastructure)."""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import BACKEND_CLASSIFICATION, MODEL_SOURCE_URI
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


async def _alias_visible(http_control, alias: str) -> bool:
    resp = await http_control.get("/v1/models")
    if resp.status_code != 200:
        return False
    body = resp.json()
    names = {m.get("name") for m in body.get("models", [])} | {
        m.get("id") for m in body.get("data", [])
    }
    return alias in names


async def _alias_gone(http_control, alias: str) -> bool:
    return not await _alias_visible(http_control, alias)


async def test_alias_lifecycle(http_control, clean_state):
    """Registry entry appears when the instance runs, disappears on stop."""
    hosts = {h["name"]: h for h in (await http_control.get("/api/hosts")).json()}
    host = hosts["host-a"]
    alias = f"life-{uuid.uuid4().hex[:8]}"

    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances", json=_instance_payload(alias)
    )
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance"]["id"]

    # Not visible while stopped.
    await wait_for(
        lambda: _alias_gone(http_control, alias),
        timeout=15.0,
        interval=0.5,
        description="alias absent while stopped",
    )

    # Visible once running (registry refresh picks it up).
    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances/{instance_id}/start"
    )
    assert resp.status_code == 200, resp.text
    await wait_for(
        lambda: _alias_visible(http_control, alias),
        timeout=15.0,
        interval=0.5,
        description="alias visible while running",
    )

    # Gone again after stop.
    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances/{instance_id}/stop"
    )
    assert resp.status_code == 200, resp.text
    await wait_for(
        lambda: _alias_gone(http_control, alias),
        timeout=60.0,
        interval=0.5,
        description="alias gone after stop",
    )

"""repo_path: instance creation + inference through the normal Solar route.

The money path: POST /api/hosts/{id}/instances with a ``repo://`` source
→ control resolves via data-repo, the host pulls from Harbor, the instance
starts, and a ``/v1/classify`` request through control's OpenAI-compatible
gateway returns real scores.
"""

from __future__ import annotations

import pytest

from fixtures.constants import (
    BACKEND_CLASSIFICATION,
    MANAGEMENT_API_KEY,
    MODEL_ALIAS,
    MODEL_SOURCE_URI,
)
from fixtures.helpers import wait_for
from fixtures.intents import classify_until_ok

pytestmark = pytest.mark.repo_path


async def _host_a(http_control):
    hosts = (await http_control.get("/api/hosts")).json()
    return next(h for h in hosts if h["name"] == "host-a")


def _instance_payload(alias: str = MODEL_ALIAS) -> dict:
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


async def test_create_instance_via_control_and_classify(
    http_control, stack, clean_state
):
    """Imperative flow: repo:// instance via control -> running -> classify."""
    host = await _host_a(http_control)
    stack.stub_harbor.reset()

    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances", json=_instance_payload()
    )
    assert resp.status_code == 200, resp.text
    instance = resp.json()["instance"]
    instance_id = instance["id"]

    # Control resolved the repo:// source and instructed the host to pull.
    assert (
        stack.stub_harbor.count_requests("GET", "/v2/supernova/test-model/manifests/")
        >= 1
    )
    assert instance["config"]["model_source"] == MODEL_SOURCE_URI
    # model_id holds the resolved local path on the host (slug dir).
    assert instance["config"]["model_id"].endswith("repo--test-model--v1"), instance[
        "config"
    ]["model_id"]

    # Created stopped; start it (through control's proxy).
    assert instance["status"] == "stopped"
    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances/{instance_id}/start"
    )
    assert resp.status_code == 200, resp.text

    await wait_for(
        lambda: _instance_state(http_control, host["id"], instance_id),
        timeout=90.0,
        interval=0.5,
        description=f"instance {instance_id} running",
    )

    # The host flips the instance to RUNNING as soon as the process is
    # alive, but the HF backend takes a few seconds to import torch and
    # bind its port. Gate on the alias appearing in the gateway registry
    # first. NOTE: /v1/models through control does NOT prove the upstream
    # is alive — the gateway fabricates a fallback entry for the alias
    # whenever the upstream query fails (gateway.py get_available_models),
    # so a dead server keeps the alias listed. The classify retry below
    # is what actually absorbs the remaining startup window.
    await wait_for(
        lambda: _registry_has_alias(http_control, MODEL_ALIAS),
        timeout=15.0,
        interval=0.5,
        description=f"gateway routes to {MODEL_ALIAS}",
    )

    # ── Inference through the normal Solar route ──
    # The registry gate above only proves the alias is listed; the
    # hf_server can still be importing torch when the first request
    # lands — retry the classify briefly (the backend binds within a
    # few seconds).
    body = await classify_until_ok(http_control, MODEL_ALIAS, timeout=20.0)
    assert body["model"] == MODEL_ALIAS
    assert len(body["choices"]) == 1
    assert body["choices"][0]["score"] > 0.0
    assert body["choices"][0]["label"].startswith("LABEL_")

    # Consistent on repeat (same fixed-seed model -> deterministic logits).
    resp2 = await http_control.post(
        "/v1/classify",
        json={"model": MODEL_ALIAS, "input": "hello integration world"},
        headers={"X-API-Key": MANAGEMENT_API_KEY},
    )
    assert resp2.status_code == 200
    assert resp2.json()["choices"][0]["label"] == body["choices"][0]["label"]


async def _instance_state(http_control, host_id: str, instance_id: str) -> bool:
    resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    if resp.status_code != 200:
        return False
    for inst in resp.json():
        if inst.get("id") == instance_id and inst.get("status") == "running":
            return True
    return False


async def test_instance_registered_in_gateway_registry(
    http_control, stack, clean_state
):
    """Running instance appears in the gateway registry with its alias."""
    host = await _host_a(http_control)
    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances", json=_instance_payload()
    )
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance"]["id"]
    await http_control.post(f"/api/hosts/{host['id']}/instances/{instance_id}/start")

    await wait_for(
        lambda: _registry_has_alias(http_control, MODEL_ALIAS),
        timeout=90.0,
        interval=0.5,
        description=f"alias {MODEL_ALIAS} in gateway registry",
    )

    # /v1/models (the normal Solar route) lists the alias.
    resp = await http_control.get("/v1/models")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {m.get("name") for m in body.get("models", [])} | {
        m.get("id") for m in body.get("data", [])
    }
    assert MODEL_ALIAS in names, f"alias missing from /v1/models: {body}"

    # Control's view of the host instance shows the resolved model path.
    resp = await http_control.get(f"/api/hosts/{host['id']}/instances")
    instances = resp.json()
    inst = next(i for i in instances if i["id"] == instance_id)
    assert inst["status"] == "running"
    # WS flat cache shape: config nested, alias + model_source present.
    assert inst["config"]["model_source"] == MODEL_SOURCE_URI


async def _registry_has_alias(http_control, alias: str) -> bool:
    resp = await http_control.get("/v1/models")
    if resp.status_code != 200:
        return False
    body = resp.json()
    names = {m.get("name") for m in body.get("models", [])} | {
        m.get("id") for m in body.get("data", [])
    }
    return alias in names

"""intent_path: intent reconciliation to ready (marker: intent_path).

Proves the full declarative path: intent submission -> reconciler creates
managed instances (repo:// resolve + host pull) -> starts them -> gateway
registry sees the alias -> inference works.
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import MANAGEMENT_API_KEY, MODEL_ALIAS
from fixtures.helpers import wait_for
from fixtures.intents import (
    create_intent,
    get_intent,
    replica_hosts,
    replica_states,
    wait_intent_phase,
    wait_intent_ready,
)

pytestmark = pytest.mark.intent_path


def _alias(prefix: str = "ready") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def test_intent_reaches_ready(http_control, clean_state):
    """Submit intent -> phase ready, 1 running replica, registry + inference."""
    intent = await create_intent(http_control, alias=_alias(), replicas=1)
    ready = await wait_intent_ready(http_control, intent["id"])

    status = ready["status"]
    assert status["observed_replicas"] == 1
    assert status["ready_replicas"] == 1
    assert status["shortfall"] == 0
    assert status["reconcile"] == "succeeded"

    # One managed replica, running, on the intent's model_source.
    states = replica_states(ready)
    assert len(states) == 1
    assert list(states.values()) == ["running"]
    replica = ready["status"]["replica_set"][0]
    assert replica["model_source"] == intent["model_source"]
    assert replica["healthy"] is True

    # Inference through the normal Solar route.
    await wait_for(
        lambda: _alias_visible(http_control),
        timeout=60.0,
        interval=1.0,
        description="alias in gateway registry",
    )
    resp = await http_control.post(
        "/v1/classify",
        json={"model": ready["alias"], "input": "declarative path"},
        headers={"X-API-Key": MANAGEMENT_API_KEY},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == ready["alias"]
    assert len(resp.json()["choices"]) == 1
    assert resp.json()["choices"][0]["score"] > 0.0


async def _alias_visible(http_control) -> bool:
    resp = await http_control.get("/v1/models")
    if resp.status_code != 200:
        return False
    body = resp.json()
    names = {m.get("name") for m in body.get("models", [])} | {
        m.get("id") for m in body.get("data", [])
    }
    return MODEL_ALIAS in names or any("ready-" in (n or "") for n in names)


async def test_phase_transitions(http_control, clean_state):
    """Observed phases: pending -> reconciling -> ready; reconcile succeeded."""
    intent = await create_intent(http_control, alias=_alias())
    # The create response is the deterministic "pending" observation (the
    # wake-driven first pass flips it to reconciling within milliseconds).
    assert intent["status"]["phase"] == "pending"
    seen: list[str] = [intent["status"]["phase"]]

    async def collect() -> bool:
        current = await get_intent(http_control, intent["id"])
        if current is None:
            return False
        phase = current["status"]["phase"]
        if phase not in seen:
            seen.append(phase)
        return phase == "ready"

    await wait_for(collect, timeout=120.0, interval=0.5,
                   description="phase transitions to ready")
    assert "reconciling" in seen, f"never observed reconciling: {seen}"
    assert seen[-1] == "ready", seen

    ready = await get_intent(http_control, intent["id"])
    assert ready is not None
    assert ready["status"]["reconcile"] == "succeeded"
    assert ready["status"]["observed_replicas"] == 1


async def test_replicas_two_uses_two_hosts(http_control, clean_state):
    """replicas: 2 -> exactly 2 managed instances on distinct hosts."""
    intent = await create_intent(http_control, alias=_alias(), replicas=2)
    ready = await wait_intent_ready(http_control, intent["id"], ready_replicas=2)

    hosts = replica_hosts(ready)
    assert len(hosts) == 2, f"expected 2 distinct hosts, got {hosts}"
    states = replica_states(ready)
    assert sorted(states.values()) == ["running", "running"]
    assert ready["status"]["observed_replicas"] == 2

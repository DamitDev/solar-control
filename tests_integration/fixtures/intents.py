"""Shared helpers for intent-path tests (S-040/S-041/S-042)."""

from __future__ import annotations

from typing import Any

from fixtures.constants import BACKEND_CLASSIFICATION, MODEL_SOURCE_URI
from fixtures.helpers import wait_for


def intent_payload(
    alias: str,
    *,
    model_source: str = MODEL_SOURCE_URI,
    replicas: int = 1,
    priority: str = "production",
    strategy: str = "rolling",
    backend: dict[str, Any] | None = None,
    placement: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a valid POST /api/intents payload (backend = HF classification)."""
    return {
        "alias": alias,
        "model_source": model_source,
        "replicas": replicas,
        "priority": priority,
        "strategy": strategy,
        "backend": backend or dict(BACKEND_CLASSIFICATION),
        "placement": placement or {},
        "resources": resources or {},
        "metadata": metadata or {},
    }


async def create_intent(http_control: Any, **overrides: Any) -> dict[str, Any]:
    """POST /api/intents and return the created intent (asserts 201)."""
    payload = intent_payload(**overrides)
    resp = await http_control.post("/api/intents", json=payload)
    assert resp.status_code == 201, f"intent create failed: {resp.status_code} {resp.text}"
    return resp.json()


async def get_intent(http_control: Any, intent_id: str) -> dict[str, Any] | None:
    resp = await http_control.get(f"/api/intents/{intent_id}")
    if resp.status_code == 404:
        return None
    assert resp.status_code == 200, resp.text
    return resp.json()


async def wait_intent_phase(
    http_control: Any,
    intent_id: str,
    phase: str,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Poll until the intent reaches ``phase``; returns the final intent."""

    async def reached() -> bool:
        intent = await get_intent(http_control, intent_id)
        return intent is not None and intent["status"]["phase"] == phase

    await wait_for(reached, timeout=timeout, interval=0.5,
                   description=f"intent {intent_id} phase={phase}")
    intent = await get_intent(http_control, intent_id)
    assert intent is not None
    return intent


async def wait_intent_ready(
    http_control: Any,
    intent_id: str,
    *,
    ready_replicas: int = 1,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Poll until the intent is fully ready (phase, replicas, conditions)."""

    async def ready() -> bool:
        intent = await get_intent(http_control, intent_id)
        if intent is None:
            return False
        status = intent["status"]
        if status["phase"] != "ready":
            return False
        if status["ready_replicas"] != ready_replicas:
            return False
        conditions = {c["type"]: c["status"] for c in status.get("conditions", [])}
        return conditions.get("Available") is True

    await wait_for(ready, timeout=timeout, interval=0.5,
                   description=f"intent {intent_id} ready")
    intent = await get_intent(http_control, intent_id)
    assert intent is not None
    return intent


def replica_hosts(intent: dict[str, Any]) -> list[str]:
    """Distinct host ids in the intent's replica_set."""
    return sorted(
        {
            r["host_id"]
            for r in intent["status"].get("replica_set", [])
            if r.get("host_id")
        }
    )


def replica_states(intent: dict[str, Any]) -> dict[str, str]:
    """instance_id -> state from the replica_set."""
    return {
        r["instance_id"]: r["state"]
        for r in intent["status"].get("replica_set", [])
        if r.get("instance_id")
    }

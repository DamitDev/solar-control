"""Shared helpers for intent-path tests (S-040/S-041/S-042)."""

from __future__ import annotations

import asyncio
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
    assert (
        resp.status_code == 201
    ), f"intent create failed: {resp.status_code} {resp.text}"
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
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Poll until the intent reaches ``phase``; returns the final intent."""

    async def reached() -> bool:
        intent = await get_intent(http_control, intent_id)
        return intent is not None and intent["status"]["phase"] == phase

    await wait_for(
        reached,
        timeout=timeout,
        interval=0.5,
        description=f"intent {intent_id} phase={phase}",
    )
    intent = await get_intent(http_control, intent_id)
    assert intent is not None
    return intent


async def wait_intent_ready(
    http_control: Any,
    intent_id: str,
    *,
    ready_replicas: int = 1,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Poll until the intent is fully ready (phase, replicas, conditions).

    The 30s default is a *convergence* budget, not a fast-path assertion:
    a replica can crash ~2-5s after spawn (host reports RUNNING before
    torch/transformers finish loading) and the reconciler's §8.2 Recreate
    converges in ~10-20s (first retry at the next 0.5s tick + spawn ~2s +
    torch load ~5-15s).  All gates here are registry-derived
    (ready_replicas, Available condition), so they can regress mid-run —
    the caller's classify retry absorbs the flip-back.
    """

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

    await wait_for(
        ready, timeout=timeout, interval=0.5, description=f"intent {intent_id} ready"
    )
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


_ROUTING_TRAP_DETAIL = "not found or no instances available"


async def classify_until_ok(
    http_control: Any,
    alias: str,
    *,
    timeout: float = 30.0,
    interval: float = 1.0,
    stack: Any = None,
    input_text: str = "hello integration world",
) -> dict[str, Any]:
    """POST /v1/classify, retrying while the backend warms up / is recreated.

    The host flips instances to RUNNING ~2s after spawn, before
    torch/transformers finish loading (documented startup race). When a
    replica crashes outright, the reconciler's §8.2 Recreate converges in
    ~10-20s — the 30s budget rides that convergence (transient-crash case
    only: a failed first recreate backs off >=10s and will not converge
    within the budget; that case produces the evidence dump, not a pass).

    **Fail fast on the routing trap:** a 404 whose detail is the
    "not found or no instances available" variant means the gateway's
    ``attempted`` set was empty — either a dead server the registry has
    already dropped, or an instance whose ``supported_endpoints`` never
    included ``/v1/classify``. The latter cannot be fixed by the
    reconciler; checking the registry once and failing immediately saves
    the full retry budget. With ``stack`` provided, the failure paths
    dump full instance evidence (pulled-file hashes, server logs, direct
    upstream probes) next to the pytest logs.
    """
    import time

    from fixtures.constants import MANAGEMENT_API_KEY as _KEY

    last_status = 0
    last_text = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = await http_control.post(
            "/v1/classify",
            json={"model": alias, "input": input_text},
            headers={"X-API-Key": _KEY},
        )
        last_status, last_text = resp.status_code, resp.text
        if resp.status_code == 200:
            return resp.json()
        if (
            resp.status_code == 404
            and _ROUTING_TRAP_DETAIL in resp.text
            and stack is not None
        ):
            from fixtures.helpers import (
                dump_instance_evidence,
                registry_entries_for_alias,
            )

            entries = await registry_entries_for_alias(stack.db_env["redis"], alias)
            if entries and not any(
                "/v1/classify" in (e.get("supported_endpoints") or []) for e in entries
            ):
                evidence = await dump_instance_evidence(
                    stack, alias, registry_entries=entries
                )
                raise AssertionError(
                    f"Routing trap: alias {alias!r} has {len(entries)} registry "
                    f"entries but none supports /v1/classify — the reconciler "
                    f"cannot fix this. Evidence in {evidence}."
                )
        await asyncio.sleep(interval)

    tail = f"last response: HTTP {last_status} {last_text[:300]}"
    if stack is not None:
        from fixtures.helpers import dump_instance_evidence

        evidence = await dump_instance_evidence(stack, alias)
        raise AssertionError(
            f"classify {alias!r} never succeeded within {timeout}s; evidence in "
            f"{evidence}. {tail}"
        )
    raise AssertionError(
        f"classify {alias!r} never succeeded within {timeout}s. {tail}"
    )

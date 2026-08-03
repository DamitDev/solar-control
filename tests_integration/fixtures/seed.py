"""Seed helpers for the D-017 integration suite.

DB access uses plain psycopg2 (the pytest process runs in solar-control's
venv, which has psycopg2-binary) — no SQLAlchemy/app imports needed, which
keeps collection-time imports free of app.* side effects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


# ── DB connection helpers ──────────────────────────────────────────


def control_db_conn(database_url: str):
    """Open a psycopg2 connection to the control DB (autocommit)."""
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    return conn


def truncate_intents(database_url: str) -> None:
    """Delete every intent row (and gateway logs) — NOT the hosts table."""
    with control_db_conn(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "TRUNCATE intents, gateway_requests, gateway_events RESTART IDENTITY"
        )


def update_intent_in_db(database_url: str, intent_id: str, **fields: Any) -> None:
    """Directly UPDATE the ``intents`` row (no PUT endpoint exists — spec §12.5).

    Used by strategy/scale tests to mutate desired state and then observe
    reconciliation. Only the given columns are touched; ``updated_at`` is
    bumped so the reconciler's ordering stays sane.
    """
    if not fields:
        return
    allowed = {
        "alias",
        "model_source",
        "replicas",
        "priority",
        "strategy",
        "backend",
        "placement",
        "resources",
        "metadata",
        "phase",
        "reconcile",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown intent columns: {sorted(unknown)}")
    sets = ", ".join(f"{col} = %s" for col in fields)
    values = list(fields.values())
    with control_db_conn(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE intents SET {sets}, updated_at = now() WHERE id = %s",
            (*values, intent_id),
        )
        if cur.rowcount != 1:
            raise AssertionError(
                f"intent {intent_id} not updated (rowcount={cur.rowcount})"
            )


def count_intents(database_url: str, alias: str | None = None) -> int:
    with control_db_conn(database_url) as conn, conn.cursor() as cur:
        if alias:
            cur.execute(
                "SELECT count(*) FROM intents WHERE deleted_at IS NULL AND alias = %s",
                (alias,),
            )
        else:
            cur.execute("SELECT count(*) FROM intents WHERE deleted_at IS NULL")
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ── API seed helpers ───────────────────────────────────────────────


async def register_host_via_api(
    http_control: Any,
    name: str,
    url: str,
    api_key: str,
    roles: list[str] | None = None,
    gpu_type: str | None = None,
) -> str:
    """Register a host row through the real management API.

    Returns the host id. ``roles``/``gpu_type`` are stored in the DB row;
    the WS registration event later refreshes them from the host itself.
    """

    payload: dict[str, Any] = {"name": name, "url": url, "api_key": api_key}
    if roles is not None:
        payload["roles"] = roles
    if gpu_type is not None:
        payload["gpu_type"] = gpu_type
    resp = await http_control.post("/api/hosts", json=payload)
    assert (
        resp.status_code == 200
    ), f"host registration failed: {resp.status_code} {resp.text}"
    body = resp.json()
    host_id = body["host"]["id"]
    logger.info("registered host %s id=%s", name, host_id)
    return host_id


# ── Test model fixture helpers ─────────────────────────────────────


def read_test_model_files(model_dir: Path) -> dict[str, bytes]:
    """Read the committed tiny HF classification model as {filename: bytes}."""
    files: dict[str, bytes] = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".txt"):
            files[path.name] = path.read_bytes()
    assert "model.safetensors" in files, "fixture model.safetensors missing"
    assert "tokenizer.json" in files, "fixture tokenizer.json missing"
    return files


async def register_model_in_data_repo(
    http_data_repo: Any,
    *,
    name: str,
    harbor_ref: str,
    version: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a model version through data-repo's API (verifies Harbor)."""
    payload: dict[str, Any] = {"harbor_ref": harbor_ref, "version": version}
    if metadata is not None:
        payload["metadata"] = metadata
    resp = await http_data_repo.post(f"/api/models/{name}/versions", json=payload)
    assert (
        resp.status_code == 201
    ), f"data-repo registration failed: {resp.status_code} {resp.text}"
    return resp.json()


def update_host_api_key(database_url: str, host_id: str, api_key: str) -> None:
    """Rotate a host's API key directly in the DB.

    Used to make control's HTTP calls to that host fail deterministically
    (401/403) while the WS channel stays up — a clean way to force a fast
    start failure for the RECREATE backoff path.
    """
    with control_db_conn(database_url) as conn, conn.cursor() as cur:
        cur.execute("UPDATE hosts SET api_key = %s WHERE id = %s", (api_key, host_id))


def update_host_roles(database_url: str, host_id: str, roles: list[str]) -> None:
    """Set a host's roles directly in the DB.

    Mirrors what the WS registration event should persist (it currently
    does not fire reliably in the test environment — see the suite README
    "Platform findings"). The migration guard validate_target_fitness
    requires the ``inference`` role on the target.
    """
    with control_db_conn(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE hosts SET roles = %s WHERE id = %s",
            (json.dumps(roles), host_id),
        )


def count_host_requests(
    stack: Any, host_letter: str, method: str, path_part: str
) -> int:
    """Count requests matching ``method`` + ``path_part`` in a host's log.

    Used for drift/stop-spam assertions: the host access log is the only
    place that records every reconciler-driven call.
    """
    svc = stack.service(host_letter)
    if svc is None:
        return 0
    log = svc.tail(100000)
    count = 0
    for line in log.splitlines():
        if f'"{method} ' in line and path_part in line:
            count += 1
    return count


def redis_cache_instances(redis_url: str, host_id: str) -> list[dict[str, Any]]:
    """Read control's Redis instances cache for a host (the reconciler's view).

    The management routes reflect the *host's* live config (which retains
    ownership markers for disowned instances — there is no host-side PATCH
    for running instances), so marker-clearing assertions must read the
    Redis cache directly.
    """
    import json

    import redis as redis_lib

    r = redis_lib.from_url(redis_url)
    raw = r.hget("solar:hosts:instances", host_id)
    if raw is None:
        return []
    return json.loads(raw)

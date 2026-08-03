"""intent_path: deployment strategies + delete semantics (marker: intent_path).

Version changes (via direct intents-row UPDATE — no PUT endpoint exists,
spec §12.5) trigger REPLACE; delete stops managed instances; delete with
?orphan=true keeps them running with markers cleared.
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import MODEL_NAME
from fixtures.helpers import wait_for
from fixtures.intents import (
    classify_until_ok,
    create_intent,
    get_intent,
    replica_states,
    wait_intent_ready,
)
from fixtures.seed import read_test_model_files, update_intent_in_db

pytestmark = pytest.mark.intent_path


def _alias(prefix: str = "strat") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _register_v2(stack, http_data_repo) -> str:
    """Register test-model:v2 (different artifact) in stub Harbor + data-repo.

    Idempotent: v2 is shared across tests in the module. The v2 artifact
    is a *valid* safetensors file (tensors bit-identical to v1, only the
    header metadata differs -> different sha256 -> different artifact
    identity), so v2 replicas actually serve inference instead of
    crashing at startup.
    """
    from fixtures.constants import FIXTURE_MODEL_DIR, harbor_port
    from fixtures.helpers import rewrite_safetensors_with_metadata

    v2_ref = f"127.0.0.1:{harbor_port(stack.harbor_ref)}/supernova/{MODEL_NAME}:v2"
    resp = await http_data_repo.get(f"/api/models/{MODEL_NAME}/versions")
    if resp.status_code == 200 and any(
        v["version"] == "v2" for v in resp.json().get("versions", [])
    ):
        return f"repo://{MODEL_NAME}:v2"
    files = read_test_model_files(FIXTURE_MODEL_DIR)
    # Different but VALID content: re-save the same tensors with a
    # "version: v2" header entry. Appending bytes to a safetensors file
    # (the old construction) makes it longer than the header's
    # data_offsets coverage -> every v2 instance crashed on start with
    # SafetensorError ("incomplete metadata, file not fully covered").
    files = dict(files)
    files["model.safetensors"] = rewrite_safetensors_with_metadata(
        files["model.safetensors"], {"version": "v2"}
    )
    stack.stub_harbor.register_model(v2_ref, files)
    resp = await http_data_repo.post(
        f"/api/models/{MODEL_NAME}/versions",
        json={"harbor_ref": v2_ref, "version": "v2"},
    )
    assert resp.status_code == 201, resp.text
    return f"repo://{MODEL_NAME}:v2"


async def _wait_model_source(http_control, intent_id: str, source: str) -> dict:
    """Poll until every running replica carries the new model_source."""

    async def migrated() -> bool:
        intent = await get_intent(http_control, intent_id)
        if intent is None or intent["status"]["phase"] != "ready":
            return False
        replicas = intent["status"].get("replica_set", [])
        if not replicas:
            return False
        return all(r.get("model_source") == source for r in replicas)

    await wait_for(
        migrated, timeout=180.0, interval=0.5, description=f"intent on {source}"
    )
    intent = await get_intent(http_control, intent_id)
    assert intent is not None
    return intent


async def test_rolling_version_change(http_control, http_data_repo, stack, clean_state):
    """DB model_source change -> old replica replaced by new, intent ready again."""
    v2_source = await _register_v2(stack, http_data_repo)
    intent = await create_intent(http_control, alias=_alias())
    await wait_intent_ready(http_control, intent["id"])
    ready = await get_intent(http_control, intent["id"])
    assert ready is not None
    old_instance_id = next(iter(replica_states(ready)))

    update_intent_in_db(
        stack.db_env["control_db"], intent["id"], model_source=v2_source
    )

    final = await _wait_model_source(http_control, intent["id"], v2_source)
    status = final["status"]
    assert status["phase"] == "ready"
    assert status["ready_replicas"] == 1
    assert status["updated_replicas"] == 1
    # The old instance is gone; the new one runs the v2 source.
    new_instance_ids = set(replica_states(final))
    assert old_instance_id not in new_instance_ids
    assert len(new_instance_ids) == 1
    assert final["status"]["replica_set"][0]["model_source"] == v2_source

    # The v2 replica must actually serve inference (the old fixture's
    # corrupt safetensors made every v2 replica a dead server).
    body = await classify_until_ok(
        http_control, final["alias"], stack=stack, timeout=30.0
    )
    assert body["model"] == final["alias"]
    assert len(body["choices"]) == 1
    assert body["choices"][0]["score"] > 0.0


async def test_immediate_version_change(
    http_control, http_data_repo, stack, clean_state
):
    """Same via strategy=immediate: converges to the new version, ready."""
    v2_source = await _register_v2(stack, http_data_repo)
    intent = await create_intent(http_control, alias=_alias(), strategy="immediate")
    await wait_intent_ready(http_control, intent["id"])

    update_intent_in_db(
        stack.db_env["control_db"], intent["id"], model_source=v2_source
    )

    final = await _wait_model_source(http_control, intent["id"], v2_source)
    assert final["status"]["phase"] == "ready"
    assert final["status"]["ready_replicas"] == 1
    assert final["status"]["updated_replicas"] == 1

    # Liveness of the v2 replica (positive assertion per house rules).
    body = await classify_until_ok(
        http_control, final["alias"], stack=stack, timeout=30.0
    )
    assert body["model"] == final["alias"]
    assert body["choices"][0]["score"] > 0.0


async def test_delete_intent_cleans_up(http_control, clean_state):
    """Delete -> managed instances stopped+deleted, phase deleted, alias gone."""
    intent = await create_intent(http_control, alias=_alias())
    ready = await wait_intent_ready(http_control, intent["id"])
    instance_id = next(iter(replica_states(ready)))
    host_id = ready["status"]["replica_set"][0]["host_id"]

    resp = await http_control.delete(f"/api/intents/{intent['id']}")
    assert resp.status_code == 202, resp.text

    # Soft-deleted intents are hidden by the API (404) once the reconciler
    # finishes cleanup — poll for that, then assert the host-side cleanup.
    await wait_for(
        lambda: _soft_deleted(http_control, intent["id"]),
        timeout=15.0,
        interval=0.5,
        description="intent soft-deleted (API 404)",
    )

    # Instance removed from the host.
    hosts_resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    assert hosts_resp.status_code == 200
    assert instance_id not in [i["id"] for i in hosts_resp.json()]

    # Alias gone from the gateway registry.
    await wait_for(
        lambda: _alias_gone(http_control, ready["alias"]),
        timeout=60.0,
        interval=0.5,
        description="alias removed from registry",
    )


async def _soft_deleted(http_control, intent_id: str) -> bool:
    return (await get_intent(http_control, intent_id)) is None


async def _alias_gone(http_control, alias: str) -> bool:
    return not await _alias_visible(http_control, alias)


async def _alias_visible(http_control, alias: str) -> bool:
    resp = await http_control.get("/v1/models")
    if resp.status_code != 200:
        return False
    body = resp.json()
    names = {m.get("name") for m in body.get("models", [])} | {
        m.get("id") for m in body.get("data", [])
    }
    return alias in names


async def test_delete_orphan_keeps_instances(http_control, stack, clean_state):
    """DELETE ?orphan=true -> instances keep running, markers cleared in cache."""
    from fixtures.seed import redis_cache_instances

    intent = await create_intent(http_control, alias=_alias())
    ready = await wait_intent_ready(http_control, intent["id"])
    instance_id = next(iter(replica_states(ready)))
    host_id = ready["status"]["replica_set"][0]["host_id"]

    resp = await http_control.delete(f"/api/intents/{intent['id']}?orphan=true")
    assert resp.status_code == 202, resp.text

    # The reconciler disowns the managed instance (markers cleared in the
    # Redis cache — the reconciler's view) and then transitions the intent
    # to 'deleted' (deleted_at set) — the API 404s. The disown chain spans
    # several passes, and the sequential loop may be held by a strategy
    # health gate, so this wait gets headroom beyond the fast-path 15s.
    await wait_for(
        lambda: _soft_deleted(http_control, intent["id"]),
        timeout=45.0,
        interval=0.5,
        description="intent soft-deleted (API 404)",
    )
    cached = redis_cache_instances(stack.db_env["redis"], host_id)
    assert cached, "instance still present in the cache"
    inst = next(
        i for i in cached if (i.get("id") or i.get("instance_id")) == instance_id
    )
    assert inst.get("managed_by") not in ("intent",)
    assert inst.get("intent_id") not in (intent["id"],)

    # The instance is still there and still running (orphaned). The host's
    # own config retains the markers (no host-side PATCH for running
    # instances) — only the control-side view is cleared.
    hosts_resp = await http_control.get(f"/api/hosts/{host_id}/instances")
    assert hosts_resp.status_code == 200
    instances = hosts_resp.json()
    inst = next(i for i in instances if i["id"] == instance_id)
    assert inst["status"] == "running"

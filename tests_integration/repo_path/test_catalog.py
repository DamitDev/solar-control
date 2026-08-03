"""repo_path: D-018 model catalog (marker: repo_path).

Proves ``GET /api/catalog/models`` through the live stack (data-repo +
control + host-a + host-b):

* the D-013 listing is proxied faithfully — totals match data-repo's own
  ``GET /api/models``, pagination (``limit``) and ``search`` are
  forwarded, and Data Repository remains the metadata authority;
* Solar enrichment reflects real deployment state — after a distribute
  the model appears in ``solar.deployed_hosts`` (joined on the host
  manifest's authoritative ``model_name``), and a running instance
  drives ``solar.status == "available"`` with ``running_instances``
  counted from the WS-pushed instance cache.

Unit-level failure modes (Data Repository unreachable, S-020 source down)
are covered in ``tests/test_catalog.py`` — killing hosts here would
disturb the session-scoped stack for every later module.
"""

from __future__ import annotations

import pytest

from fixtures.constants import MODEL_ALIAS, MODEL_NAME, MODEL_SOURCE_URI
from fixtures.helpers import wait_for

pytestmark = pytest.mark.repo_path


async def _host_a(http_control):
    hosts = (await http_control.get("/api/hosts")).json()
    return next(h for h in hosts if h["name"] == "host-a")


def _instance_payload(alias: str = MODEL_ALIAS) -> dict:
    """Imperative instance payload (same shape as repo_path/test_inference)."""
    return {
        "config": {
            "backend_type": "huggingface_classification",
            "alias": alias,
            "model_source": MODEL_SOURCE_URI,
            "device": "cpu",
            "dtype": "float32",
            "max_length": 128,
            "labels": [f"LABEL_{i}" for i in range(5)],
        },
        "priority": "staging",
    }


async def _catalog_item(http_control, name: str) -> dict | None:
    """Return one catalog item by name (None when missing/unreachable)."""
    resp = await http_control.get("/api/catalog/models")
    if resp.status_code != 200:
        return None
    for item in resp.json()["items"]:
        if item["name"] == name:
            return item
    return None


async def _running_instances_for(http_control, name: str) -> bool:
    """True once the catalog reports >=1 running instance for the model."""
    item = await _catalog_item(http_control, name)
    if item is None:
        return False
    return item["solar"]["running_instances"] >= 1


async def test_catalog_proxies_data_repository_listing(
    http_control, http_data_repo, stack, clean_state
):
    """D-013 passthrough: totals match, metadata round-trips, enrichment ok."""
    repo_resp = await http_data_repo.get("/api/models")
    assert repo_resp.status_code == 200, repo_resp.text
    repo_total = repo_resp.json()["total"]

    resp = await http_control.get("/api/catalog/models")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == repo_total
    assert len(body["items"]) == repo_total

    item = next((i for i in body["items"] if i["name"] == MODEL_NAME), None)
    assert item is not None, f"fixture model missing from catalog: {body}"
    assert item["category"] == "model"
    assert item["versions_count"] >= 1
    assert item["latest_version"] == "v1"

    # Clean state: hosts hold no models and no instances -> the model is
    # honestly unavailable and the availability source answered (ok).
    assert item["solar"]["status"] == "unavailable"
    assert item["solar"]["running_instances"] == 0
    assert item["solar"]["deployed_hosts"] == []
    assert item["solar"]["instances"] == []
    assert body["meta"]["enrichment"] == "ok"


async def test_catalog_pagination_and_search_passthrough(
    http_control, http_data_repo, stack, clean_state
):
    """limit/offset and search are forwarded to data-repo, not re-implemented."""
    repo_resp = await http_data_repo.get("/api/models")
    assert repo_resp.status_code == 200, repo_resp.text
    repo_total = repo_resp.json()["total"]

    resp = await http_control.get("/api/catalog/models", params={"limit": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == repo_total
    assert len(body["items"]) == 1

    resp = await http_control.get("/api/catalog/models", params={"search": MODEL_NAME})
    assert resp.status_code == 200, resp.text
    names = [i["name"] for i in resp.json()["items"]]
    assert (
        MODEL_NAME in names
    ), f"search={MODEL_NAME!r} missed the fixture model: {names}"


async def test_catalog_reports_deployed_hosts_after_distribute(
    http_control, stack, clean_state
):
    """A distributed model appears in deployed_hosts with status deployed."""
    host = await _host_a(http_control)
    resp = await http_control.post(
        "/api/models/distribute",
        json={"target_host_id": host["id"], "source_uri": MODEL_SOURCE_URI},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()) == 1

    item = await _catalog_item(http_control, MODEL_NAME)
    assert item is not None, "catalog missing the just-distributed model"
    assert item["solar"]["status"] == "deployed"
    assert item["solar"]["running_instances"] == 0

    deployed = item["solar"]["deployed_hosts"]
    assert len(deployed) == 1, f"expected host-a only: {deployed}"
    assert deployed[0]["host_id"] == host["id"]
    assert deployed[0]["host_name"] == "host-a"
    # Join went through the manifest's authoritative model_name: the slug
    # dir path proves it is the repo pull, not a name-collision.
    assert deployed[0]["path"].endswith(f"repo--{MODEL_NAME}--v1"), deployed[0]["path"]


async def test_catalog_reports_running_instance(http_control, stack, clean_state):
    """A running instance drives status=available + running_instances=1."""
    host = await _host_a(http_control)
    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances", json=_instance_payload()
    )
    assert resp.status_code == 200, resp.text
    instance_id = resp.json()["instance"]["id"]
    resp = await http_control.post(
        f"/api/hosts/{host['id']}/instances/{instance_id}/start"
    )
    assert resp.status_code == 200, resp.text

    # The catalog counts instances from the WS-pushed host cache; the host
    # flips to running once the backend process is alive (a few seconds).
    await wait_for(
        lambda: _running_instances_for(http_control, MODEL_NAME),
        timeout=60.0,
        interval=0.5,
        description=f"catalog shows a running instance for {MODEL_NAME}",
    )

    item = await _catalog_item(http_control, MODEL_NAME)
    assert item is not None
    assert item["solar"]["status"] == "available"
    assert item["solar"]["running_instances"] == 1
    assert len(item["solar"]["instances"]) == 1
    assert item["solar"]["instances"][0]["host_id"] == host["id"]
    assert item["solar"]["instances"][0]["instance_id"] == instance_id
    # The instance was created via control -> host pulled the model too.
    assert item["solar"]["deployed_hosts"], "expected the model on host-a"

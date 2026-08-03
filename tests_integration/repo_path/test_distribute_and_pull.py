"""repo_path: distribution + host pull + cache (marker: repo_path).

Proves POST /api/models/distribute instructs the host to pull directly
from Harbor via ORAS, the manifest caches by source_uri (second distribute
is a cache hit with no second Harbor pull), and data-repo metadata
round-trips into the host manifest entry.
"""

from __future__ import annotations

import json

import pytest

from fixtures.constants import MODEL_NAME, MODEL_SOURCE_URI, MODEL_VERSION

pytestmark = pytest.mark.repo_path


async def _host_a(http_control):
    hosts = (await http_control.get("/api/hosts")).json()
    return next(h for h in hosts if h["name"] == "host-a")


async def test_distribute_pulls_once_and_caches(
    http_control, http_host, stack, clean_state
):
    """First distribute pulls from Harbor; second is a cache hit, no re-pull."""
    host = await _host_a(http_control)
    stack.stub_harbor.reset()

    resp = await http_control.post(
        "/api/models/distribute",
        json={"target_host_id": host["id"], "source_uri": MODEL_SOURCE_URI},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()
    assert len(results) == 1
    assert results[0]["source_uri"] == MODEL_SOURCE_URI
    assert results[0]["cached"] is False
    # Path is the absolute pulled directory on the host (slug dir).
    assert results[0]["path"].endswith(f"repo--{MODEL_NAME}--{MODEL_VERSION}"), results[
        0
    ]["path"]

    # Host pulled from (stub) Harbor via ORAS: manifests + blob GETs.
    assert (
        stack.stub_harbor.count_requests(
            "GET", f"/v2/supernova/{MODEL_NAME}/manifests/"
        )
        >= 1
    )
    blob_pulls = stack.stub_harbor.count_requests("GET", "/blobs/")
    assert blob_pulls >= 1, "host never downloaded blobs"

    # Manifest entry keyed by source_uri slug.
    manifest = json.loads((stack.models_dir_a / "manifest.json").read_text())
    entry = next(
        (m for m in manifest["models"] if m["source_uri"] == MODEL_SOURCE_URI), None
    )
    assert entry is not None, f"no manifest entry for {MODEL_SOURCE_URI}: {manifest}"
    assert entry["slug"] == f"repo--{MODEL_NAME}--{MODEL_VERSION}"
    # digest is carried through from data-repo's resolve (checksum field).
    assert (entry.get("digest") or "").startswith("sha256:")
    assert (stack.models_dir_a / entry["slug"] / "model.safetensors").exists()

    # Second distribute: cache hit, NO additional Harbor traffic.
    harbor_before = len(stack.stub_harbor.received_requests())
    resp = await http_control.post(
        "/api/models/distribute",
        json={"target_host_id": host["id"], "source_uri": MODEL_SOURCE_URI},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()
    assert results[0]["cached"] is True
    assert (
        len(stack.stub_harbor.received_requests()) == harbor_before
    ), "second distribute re-pulled from Harbor despite cached manifest"


async def test_manifest_round_trips_repo_metadata(
    http_control, http_host, stack, clean_state
):
    """Data-repo metadata (name, version, checksum) lands in the host manifest."""
    host = await _host_a(http_control)
    resp = await http_control.post(
        "/api/models/distribute",
        json={"target_host_id": host["id"], "source_uri": MODEL_SOURCE_URI},
    )
    assert resp.status_code == 200, resp.text

    manifest = json.loads((stack.models_dir_a / "manifest.json").read_text())
    entry = next(m for m in manifest["models"] if m["source_uri"] == MODEL_SOURCE_URI)
    assert entry["name"] == MODEL_NAME
    assert entry["version"] == MODEL_VERSION
    assert entry["category"] == "model"
    assert entry["checksum"].startswith("sha256:")
    assert isinstance(entry["metadata"], dict)

    # GET /models on the host surfaces the metadata too.
    resp = await http_host.get("/models")
    assert resp.status_code == 200, resp.text
    models = resp.json()
    listed = next((m for m in models if m.get("source_uri") == MODEL_SOURCE_URI), None)
    assert listed is not None, f"host GET /models missing {MODEL_SOURCE_URI}: {models}"
    assert listed["model_name"] == MODEL_NAME
    assert listed["version"] == MODEL_VERSION

"""repo_path: Data Repository registration (marker: repo_path).

Proves the model registration flow: a ``harbor_ref`` registered through
data-repo's API triggers a real ``verify_artifact`` call against (stub)
Harbor, and unknown refs are rejected with nothing persisted.
"""

from __future__ import annotations

import uuid

import pytest

from fixtures.constants import FIXTURE_MODEL_DIR
from fixtures.seed import read_test_model_files, register_model_in_data_repo

pytestmark = pytest.mark.repo_path


async def _unique_name(prefix: str = "reg-model") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def test_register_model_version(http_data_repo, stack, clean_state):
    """POST /api/models/{name}/versions with a known harbor_ref -> 201 + listed.

    Mirrors the real flow: the artifact is pushed to Harbor *first*, then
    registered. Also proves data-repo called stub Harbor (verify_artifact)
    during registration — the first link in the repository->host chain.
    """
    from fixtures.constants import harbor_port

    name = await _unique_name()
    harbor_ref = f"127.0.0.1:{harbor_port(stack.harbor_ref)}/supernova/{name}:v1"
    # Push the artifact into (stub) Harbor, then register metadata.
    stack.stub_harbor.register_model(
        harbor_ref, read_test_model_files(FIXTURE_MODEL_DIR)
    )
    stack.stub_harbor.reset()

    body = await register_model_in_data_repo(
        http_data_repo, name=name, harbor_ref=harbor_ref, version="v1"
    )
    assert body["name"] == name
    assert body["version"] == "v1"
    assert body["category"] == "model"

    # Listed under the model's versions
    resp = await http_data_repo.get(f"/api/models/{name}/versions")
    assert resp.status_code == 200
    versions = resp.json()["versions"]
    assert [v["version"] for v in versions] == ["v1"]
    assert versions[0]["harbor_ref"] == harbor_ref

    # data-repo verified the artifact against (stub) Harbor on registration
    manifest_calls = stack.stub_harbor.count_requests(
        "HEAD", f"/v2/supernova/{name}/manifests/v1"
    )
    assert manifest_calls >= 1, (
        "data-repo never called Harbor verify_artifact during registration; "
        f"requests seen: {stack.stub_harbor.received_paths()}"
    )


async def test_register_version_404_unknown_harbor_ref(
    http_data_repo, stack, clean_state
):
    """Registration with a ref not present in Harbor -> 4xx, nothing persisted."""
    from fixtures.constants import harbor_port

    name = await _unique_name()
    missing_ref = f"127.0.0.1:{harbor_port(stack.harbor_ref)}/supernova/{name}:v9"

    resp = await http_data_repo.post(
        f"/api/models/{name}/versions",
        json={"harbor_ref": missing_ref, "version": "v1"},
    )
    assert resp.status_code in (
        404,
        422,
    ), f"expected 4xx for unknown Harbor ref, got {resp.status_code}: {resp.text}"

    # Nothing persisted
    resp = await http_data_repo.get(f"/api/models/{name}")
    assert resp.status_code == 404

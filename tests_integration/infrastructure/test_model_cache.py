"""infrastructure: model cache — one pull per host per artifact (marker: infrastructure)."""

from __future__ import annotations

import pytest

from fixtures.constants import MODEL_SOURCE_URI

pytestmark = pytest.mark.infrastructure


async def test_shared_artifact_pulled_once_per_host(http_control, stack, clean_state):
    """Two hosts serving the same repo:// -> exactly one Harbor pull each."""
    hosts = {h["name"]: h for h in (await http_control.get("/api/hosts")).json()}
    stack.stub_harbor.reset()

    for name in ("host-a", "host-b"):
        resp = await http_control.post(
            "/api/models/distribute",
            json={"target_host_id": hosts[name]["id"], "source_uri": MODEL_SOURCE_URI},
        )
        assert resp.status_code == 200, resp.text
        results = resp.json()
        assert len(results) == 1
        assert results[0]["cached"] is False, f"{name}: expected a real pull"

    # Exactly one pull per host. The OCI token dance means each pull issues
    # two manifest GETs (401 challenge + authed bearer), so count the token
    # requests instead (one per pull).
    token_gets = stack.stub_harbor.count_requests("GET", "/service/token")
    assert token_gets == 2, (
        f"expected 2 pulls (one per host), got {token_gets}: "
        f"{stack.stub_harbor.received_paths()}"
    )

    # Second distribute per host is a cache hit with no additional traffic.
    for name in ("host-a", "host-b"):
        resp = await http_control.post(
            "/api/models/distribute",
            json={"target_host_id": hosts[name]["id"], "source_uri": MODEL_SOURCE_URI},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["cached"] is True

    assert stack.stub_harbor.count_requests("GET", "/service/token") == 2

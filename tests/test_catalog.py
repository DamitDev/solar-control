"""Tests for the D-018 model catalog endpoint (GET /api/catalog/models)."""

import pytest
from unittest.mock import AsyncMock, patch

import aiohttp
from fastapi import HTTPException

from app.models import Host, HostStatus
from app.routes.management.catalog import (
    _collect_availability,
    _collect_running_instances,
    _derive_status,
    _model_name_from_source,
    get_catalog_models,
)

REPO_MODELS = [
    {
        "name": "llama-3.1-8b",
        "category": "model",
        "description": "Meta Llama 3.1 8B",
        "versions_count": 2,
        "latest_version": "v2",
        "created_at": "2026-01-01T00:00:00Z",
    },
    {
        "name": "qwen-7b",
        "category": "model",
        "description": "Qwen 7B",
        "versions_count": 1,
        "latest_version": "v1",
        "created_at": "2026-02-01T00:00:00Z",
    },
    {
        "name": "mistral-7b",
        "category": "model",
        "description": "Mistral 7B",
        "versions_count": 3,
        "latest_version": "v3",
        "created_at": "2026-03-01T00:00:00Z",
    },
]


def _cm_response(status: int, payload):
    """AsyncMock context manager whose __aenter__ yields an HTTP response."""
    resp = AsyncMock()
    resp.status = status
    resp.json.return_value = payload
    cm = AsyncMock()
    cm.__aenter__.return_value = resp
    return cm


@pytest.fixture
def catalog_settings():
    with patch("app.routes.management.catalog.settings") as mock_settings:
        mock_settings.data_repository_url = "http://data-repo:8000"
        mock_settings.data_repository_api_key = ""
        mock_settings.data_repository_timeout_s = 10.0
        yield mock_settings


@pytest.fixture
def mock_host():
    return Host(
        id="host-1",
        name="Test Host",
        url="http://test-host:8000",
        api_key="test-key",
        status=HostStatus.ONLINE,
    )


@pytest.fixture
def mock_host_2():
    return Host(
        id="host-2",
        name="Test Host 2",
        url="http://test-host-2:8000",
        api_key="test-key-2",
        status=HostStatus.ONLINE,
    )


# ── Proxy behaviour (D-013 passthrough) ───────────────────────


@pytest.mark.anyio
async def test_catalog_proxies_pagination_and_search(catalog_settings):
    listing = {"total": 2, "items": REPO_MODELS[:2]}
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("app.database.hosts.host_db.get_all_hosts", return_value=[]),
    ):
        mock_get.return_value = _cm_response(200, listing)

        resp = await get_catalog_models(search="llama", limit=10, offset=20)

    call = mock_get.call_args
    assert call.args[0] == "http://data-repo:8000/api/models"
    assert call.kwargs["params"] == {"limit": 10, "offset": 20, "search": "llama"}

    assert resp.total == 2
    assert len(resp.items) == 2
    assert resp.items[0].name == "llama-3.1-8b"
    assert resp.items[0].versions_count == 2
    assert resp.items[0].latest_version == "v2"
    assert resp.items[0].description == "Meta Llama 3.1 8B"
    assert resp.items[1].name == "qwen-7b"
    # No hosts configured -> every model is truly unavailable, enrichment ok.
    assert resp.items[0].solar.status == "unavailable"
    assert resp.items[0].solar.running_instances == 0
    assert resp.items[0].solar.deployed_hosts == []
    assert resp.meta.enrichment == "ok"


@pytest.mark.anyio
async def test_catalog_proxies_default_pagination(catalog_settings):
    listing = {"total": 0, "items": []}
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("app.database.hosts.host_db.get_all_hosts", return_value=[]),
    ):
        mock_get.return_value = _cm_response(200, listing)

        resp = await get_catalog_models(search=None, limit=50, offset=0)

    call = mock_get.call_args
    assert call.kwargs["params"] == {"limit": 50, "offset": 0}
    assert "search" not in call.kwargs["params"]
    assert resp.total == 0
    assert resp.items == []


@pytest.mark.anyio
async def test_catalog_forwards_data_repository_api_key(catalog_settings):
    catalog_settings.data_repository_api_key = "repo-key"
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("app.database.hosts.host_db.get_all_hosts", return_value=[]),
    ):
        mock_get.return_value = _cm_response(200, {"total": 0, "items": []})
        await get_catalog_models()

    assert mock_get.call_args.kwargs["headers"]["X-API-Key"] == "repo-key"


# ── Enrichment ────────────────────────────────────────────────


@pytest.mark.anyio
async def test_catalog_enriches_with_deployment_and_running_instances(
    catalog_settings, mock_host, mock_host_2
):
    """Repo models joined to host availability (model_name) and instances."""
    hosts = [mock_host, mock_host_2]
    listing = {"total": 3, "items": REPO_MODELS}
    host1_models = [
        {
            "name": "repo--llama-3.1-8b--v2",
            "model_name": "llama-3.1-8b",
            "size_bytes": 1000,
            "path": "/models/repo--llama-3.1-8b--v2",
        },
        {
            "name": "repo--qwen-7b--v1",
            "model_name": "qwen-7b",
            "size_bytes": 2000,
            "path": "/models/repo--qwen-7b--v1",
        },
    ]
    host2_models = [
        {
            "name": "repo--llama-3.1-8b--v2",
            "model_name": "llama-3.1-8b",
            "size_bytes": 1000,
            "path": "/models/repo--llama-3.1-8b--v2",
        }
    ]
    instances = [
        {"id": "i1", "status": "running", "model_source": "repo://llama-3.1-8b:v2"},
        {"id": "i2", "status": "stopped", "model_source": "repo://qwen-7b:v1"},
    ]

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch(
            "app.socketio_app.host_handlers.get_host_instances",
            side_effect=[instances, []],
        ),
    ):
        mock_get.side_effect = [
            _cm_response(200, listing),
            _cm_response(200, host1_models),
            _cm_response(200, host2_models),
        ]

        resp = await get_catalog_models()

    by_name = {item.name: item for item in resp.items}

    # llama: deployed on both hosts, one running instance
    llama = by_name["llama-3.1-8b"]
    assert llama.solar.status == "available"
    assert llama.solar.running_instances == 1
    assert len(llama.solar.deployed_hosts) == 2
    assert {h.host_id for h in llama.solar.deployed_hosts} == {"host-1", "host-2"}
    assert llama.solar.deployed_hosts[0].host_name == "Test Host"
    assert llama.solar.instances[0].instance_id == "i1"
    assert llama.solar.instances[0].host_id == "host-1"

    # qwen: deployed on host-1 only, nothing running
    qwen = by_name["qwen-7b"]
    assert qwen.solar.status == "deployed"
    assert qwen.solar.running_instances == 0
    assert len(qwen.solar.deployed_hosts) == 1
    assert qwen.solar.deployed_hosts[0].host_id == "host-1"

    # mistral: nowhere
    mistral = by_name["mistral-7b"]
    assert mistral.solar.status == "unavailable"
    assert mistral.solar.running_instances == 0
    assert mistral.solar.deployed_hosts == []
    assert mistral.solar.instances == []

    assert resp.meta.enrichment == "ok"


@pytest.mark.anyio
async def test_catalog_joins_legacy_host_entries_by_name(catalog_settings, mock_host):
    """Pre-D-016 host entries (no model_name) join on the manifest name."""
    listing = {"total": 1, "items": [REPO_MODELS[0]]}
    host_models = [
        {
            "name": "llama-3.1-8b",
            "size_bytes": 1000,
            "path": "/models/llama-3.1-8b",
        }
    ]
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("app.database.hosts.host_db.get_all_hosts", return_value=[mock_host]),
        patch("app.socketio_app.host_handlers.get_host_instances", return_value=[]),
    ):
        mock_get.side_effect = [
            _cm_response(200, listing),
            _cm_response(200, host_models),
        ]

        resp = await get_catalog_models()

    assert len(resp.items) == 1
    assert resp.items[0].solar.status == "deployed"
    assert resp.items[0].solar.deployed_hosts[0].host_id == "host-1"


@pytest.mark.anyio
async def test_catalog_counts_running_instances_from_source_variants(
    catalog_settings, mock_host, mock_host_2
):
    """Instances joined via config-nested and top-level model_source."""
    hosts = [mock_host, mock_host_2]
    qwen_repo = {
        "name": "org/qwen-7b",
        "category": "model",
        "description": "Qwen 7B",
        "versions_count": 1,
        "latest_version": "v1",
        "created_at": "2026-02-01T00:00:00Z",
    }
    listing = {"total": 2, "items": [REPO_MODELS[0], qwen_repo]}
    host1_instances = [
        # config-nested model_source (raw solar-host REST shape)
        {
            "id": "i1",
            "status": "running",
            "config": {"model_source": "repo://llama-3.1-8b:v2"},
        },
        # non-running instances are not counted
        {"id": "i2", "status": "stopped", "model_source": "repo://llama-3.1-8b:v2"},
        # top-level model_source, huggingface scheme
        {"id": "i3", "status": "running", "model_source": "huggingface://org/qwen-7b"},
        # unparsable source is ignored
        {"id": "i4", "status": "running", "model_source": "local:///tmp/x"},
    ]
    host2_instances = [
        {"id": "i5", "status": "running", "model_source": "repo://llama-3.1-8b:v1"}
    ]

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch(
            "app.socketio_app.host_handlers.get_host_instances",
            side_effect=[host1_instances, host2_instances],
        ),
    ):
        mock_get.side_effect = [
            _cm_response(200, listing),
            _cm_response(200, []),
            _cm_response(200, []),
        ]

        resp = await get_catalog_models()

    by_name = {item.name: item for item in resp.items}
    llama = by_name["llama-3.1-8b"]
    assert llama.solar.status == "available"
    assert llama.solar.running_instances == 2
    assert {i.instance_id for i in llama.solar.instances} == {"i1", "i5"}

    qwen = by_name["org/qwen-7b"]
    assert qwen.solar.status == "available"
    assert qwen.solar.running_instances == 1
    assert qwen.solar.instances[0].instance_id == "i3"


# ── Data Repository failures ──────────────────────────────────


@pytest.mark.anyio
async def test_catalog_data_repository_unreachable(catalog_settings):
    with patch(
        "aiohttp.ClientSession.get",
        side_effect=aiohttp.ClientConnectionError("Refused"),
    ):
        with pytest.raises(HTTPException) as exc:
            await get_catalog_models()
    assert exc.value.status_code == 502
    assert "unreachable" in str(exc.value.detail)


@pytest.mark.anyio
async def test_catalog_data_repository_not_configured(catalog_settings):
    catalog_settings.data_repository_url = ""
    with pytest.raises(HTTPException) as exc:
        await get_catalog_models()
    assert exc.value.status_code == 500
    assert "DATA_REPOSITORY_URL" in str(exc.value.detail)


@pytest.mark.anyio
async def test_catalog_data_repository_upstream_error(catalog_settings):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value = _cm_response(500, {"detail": "boom"})
        with pytest.raises(HTTPException) as exc:
            await get_catalog_models()
    assert exc.value.status_code == 502
    assert "boom" in str(exc.value.detail)


# ── Missing/unavailable S-020 enrichment ──────────────────────


@pytest.mark.anyio
async def test_catalog_availability_source_down_marks_unknown(
    catalog_settings, mock_host, mock_host_2
):
    """All hosts unreachable -> no false 'unavailable'; statuses stay unknown."""
    listing = {"total": 1, "items": [REPO_MODELS[0]]}
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch(
            "app.database.hosts.host_db.get_all_hosts",
            return_value=[mock_host, mock_host_2],
        ),
        patch("app.socketio_app.host_handlers.get_host_instances", return_value=[]),
    ):
        mock_get.side_effect = [
            _cm_response(200, listing),
            aiohttp.ClientConnectionError("down"),
            aiohttp.ClientConnectionError("down"),
        ]

        resp = await get_catalog_models()

    assert resp.meta.enrichment == "unavailable"
    item = resp.items[0]
    assert item.solar.status == "unknown"
    assert item.solar.deployed_hosts == []
    assert item.solar.running_instances == 0


@pytest.mark.anyio
async def test_catalog_availability_partial_failure(
    catalog_settings, mock_host, mock_host_2
):
    """Some hosts down -> partial meta; seen evidence still reported."""
    listing = {"total": 2, "items": REPO_MODELS[:2]}
    host2_models = [
        {
            "name": "repo--llama-3.1-8b--v2",
            "model_name": "llama-3.1-8b",
            "size_bytes": 1000,
            "path": "/models/repo--llama-3.1-8b--v2",
        }
    ]
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch(
            "app.database.hosts.host_db.get_all_hosts",
            return_value=[mock_host, mock_host_2],
        ),
        patch("app.socketio_app.host_handlers.get_host_instances", return_value=[]),
    ):
        mock_get.side_effect = [
            _cm_response(200, listing),
            aiohttp.ClientConnectionError("down"),
            _cm_response(200, host2_models),
        ]

        resp = await get_catalog_models()

    assert resp.meta.enrichment == "partial"
    by_name = {item.name: item for item in resp.items}
    # Deployment evidence on the surviving host is still reported.
    assert by_name["llama-3.1-8b"].solar.status == "deployed"
    assert len(by_name["llama-3.1-8b"].solar.deployed_hosts) == 1
    # No evidence + partial source -> unknown, not unavailable.
    assert by_name["qwen-7b"].solar.status == "unknown"


# ── Unit-level helpers ────────────────────────────────────────


@pytest.mark.anyio
async def test_collect_availability_reports_failures(
    catalog_settings, mock_host, mock_host_2
):
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch(
            "app.database.hosts.host_db.get_all_hosts",
            return_value=[mock_host, mock_host_2],
        ),
    ):
        mock_get.side_effect = [
            _cm_response(200, []),
            aiohttp.ClientConnectionError("down"),
        ]
        by_name, status = await _collect_availability()
    assert by_name == {}
    assert status == "partial"


@pytest.mark.anyio
async def test_collect_running_instances_skips_non_running(mock_host, mock_host_2):
    with (
        patch(
            "app.database.hosts.host_db.get_all_hosts",
            return_value=[mock_host, mock_host_2],
        ),
        patch(
            "app.socketio_app.host_handlers.get_host_instances",
            side_effect=[
                [{"id": "i1", "status": "running", "model_source": "repo://a:v1"}],
                [{"id": "i2", "status": "stopped", "model_source": "repo://a:v1"}],
            ],
        ),
    ):
        by_name = await _collect_running_instances()
    assert len(by_name["a"]) == 1
    assert by_name["a"][0].instance_id == "i1"


def test_model_name_from_source():
    assert (
        _model_name_from_source("repo://llama-3.1-8b:v2/model.gguf") == "llama-3.1-8b"
    )
    assert _model_name_from_source("huggingface://org/qwen-7b") == "org/qwen-7b"
    assert _model_name_from_source("local:///tmp/x") is None
    assert _model_name_from_source("not-a-uri") is None
    assert _model_name_from_source(None) is None
    assert _model_name_from_source("") is None


def test_derive_status_evidence_based():
    from app.routes.management.catalog import DeployedHostInfo, RunningInstanceInfo

    running = [RunningInstanceInfo(host_id="h", host_name="H", instance_id="i")]
    deployed = [DeployedHostInfo(host_id="h", host_name="H")]

    assert (
        _derive_status(running=[], deployed=[], availability_ok=True) == "unavailable"
    )
    assert _derive_status(running=[], deployed=[], availability_ok=False) == "unknown"
    assert (
        _derive_status(running=[], deployed=deployed, availability_ok=False)
        == "deployed"
    )
    assert (
        _derive_status(running=running, deployed=[], availability_ok=False)
        == "available"
    )

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from app.models import Host, HostStatus
from app.routes.management.models import (
    _pull_on_host,
    _check_disk_space,
    distribute_model,
    DistributeRequest,
    _StructuredPullError,
)
from app.model_resolvers.parser import RepoURI, parse


@pytest.fixture
def mock_host():
    return Host(
        id="host-1",
        name="Test Host",
        url="http://test-host:8000",
        api_key="test-key",
        status=HostStatus.ONLINE,
    )


@pytest.mark.anyio
async def test_check_disk_space_success(mock_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"disk": {"available_gb": 10.5}}
        mock_get.return_value.__aenter__.return_value = mock_resp

        available = await _check_disk_space(mock_host)
        assert available == 10.5


@pytest.mark.anyio
async def test_check_disk_space_failure(mock_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")
        available = await _check_disk_space(mock_host)
        assert available is None


@pytest.mark.anyio
async def test_pull_on_host_success(mock_host):
    uri = "huggingface://microsoft/phi-3"
    parsed = parse(uri)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"path": "/models/hf--phi3", "cached": True}
        mock_post.return_value.__aenter__.return_value = mock_resp

        path, cached = await _pull_on_host(parsed, uri, mock_host)
        assert path == "/models/hf--phi3"
        assert cached is True

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["source_uri"] == uri
        assert kwargs["json"]["model_id"] == "microsoft/phi-3"


@pytest.mark.anyio
async def test_pull_on_host_404_propagated(mock_host):
    uri = "huggingface://microsoft/phi-3"
    parsed = parse(uri)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.json.return_value = {"detail": "Model not found"}
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await _pull_on_host(parsed, uri, mock_host)
        assert isinstance(result, _StructuredPullError)
        assert result.status_code == 404
        assert "Model not found" in result.detail


@pytest.mark.anyio
async def test_pull_on_host_507_propagated(mock_host):
    uri = "huggingface://microsoft/phi-3"
    parsed = parse(uri)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 507
        mock_resp.json.return_value = {"error": "Disk full"}
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await _pull_on_host(parsed, uri, mock_host)
        assert isinstance(result, _StructuredPullError)
        assert result.status_code == 507
        assert "Disk full" in result.detail


@pytest.mark.anyio
async def test_pull_on_host_connection_error(mock_host):
    uri = "huggingface://microsoft/phi-3"
    parsed = parse(uri)

    with patch("aiohttp.ClientSession.post") as mock_post:
        import aiohttp

        mock_post.side_effect = aiohttp.ClientConnectionError("Refused")

        with pytest.raises(HTTPException) as exc:
            await _pull_on_host(parsed, uri, mock_host)
        assert exc.value.status_code == 502
        assert "is unreachable" in exc.value.detail


@pytest.mark.anyio
async def test_pull_on_host_local_uri(mock_host):
    uri = "local:///path/to/model"
    parsed = parse(uri)

    result = await _pull_on_host(parsed, uri, mock_host)
    assert isinstance(result, _StructuredPullError)
    assert result.status_code == 400
    assert "Cannot distribute local://" in result.detail
    assert result.source_uri == "local:///path/to/model"
    assert result.error == "validation_error"


@pytest.mark.anyio
async def test_pull_on_host_repo_uri(mock_host, repo_settings):
    uri = "repo://model:v1"
    parsed = parse(uri)
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = {
            "category": "model",
            "name": "model",
            "version": "v1",
            "harbor_ref": "imgrepo.damit.hu/supernova/model:v1",
            "size_bytes": 123,
            "checksum": "sha256:abc",
            "metadata": {},
            "created_at": "2026-04-29T10:00:00Z",
        }
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {"path": "/models/repo--model--v1"}
        mock_post.return_value.__aenter__.return_value = pull_resp

        result = await _pull_on_host(parsed, uri, mock_host)
        assert result == ("/models/repo--model--v1", False)


@pytest.mark.anyio
async def test_pull_on_host_repo_uri_subpath_stripped_for_data_repo(
    mock_host, repo_settings
):
    """``repo://name:version/file.gguf`` queries Data Repo with the base URI
    but forwards the full URI (subpath included) to the host pull (D-017)."""
    uri = "repo://model:v1/model.gguf"
    parsed = parse(uri)
    assert isinstance(parsed, RepoURI)
    assert parsed.subpath == "model.gguf"
    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = {
            "category": "model",
            "name": "model",
            "version": "v1",
            "harbor_ref": "imgrepo.damit.hu/supernova/model:v1",
            "size_bytes": 123,
            "checksum": "sha256:abc",
            "metadata": {},
            "created_at": "2026-04-29T10:00:00Z",
        }
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {"path": "/models/repo--model--v1/model.gguf"}
        mock_post.return_value.__aenter__.return_value = pull_resp

        result = await _pull_on_host(parsed, uri, mock_host)
        assert result == ("/models/repo--model--v1/model.gguf", False)

        # Data Repo sees the base URI only; the host gets the full URI.
        _, get_kwargs = mock_get.call_args
        assert get_kwargs["params"] == {"uri": "repo://model:v1"}
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["json"]["source_uri"] == "repo://model:v1/model.gguf"


@pytest.mark.anyio
async def test_pull_on_host_repo_uri_resolve_5xx_raises(mock_host, repo_settings):
    uri = "repo://model:v1"
    parsed = parse(uri)
    with patch("aiohttp.ClientSession.get") as mock_get:
        resolve_resp = AsyncMock()
        resolve_resp.status = 500
        resolve_resp.json.return_value = {"detail": "internal error"}
        mock_get.return_value.__aenter__.return_value = resolve_resp

        with pytest.raises(HTTPException) as exc:
            await _pull_on_host(parsed, uri, mock_host)

        assert exc.value.status_code == 502
        assert "Data Repository resolution failed [500]" in exc.value.detail
        assert "internal error" in exc.value.detail


@pytest.mark.anyio
async def test_pull_on_host_repo_uri_missing_harbor_ref_is_partial(
    mock_host, repo_settings
):
    """A Data Repository response missing harbor_ref must not abort the batch."""
    uri = "repo://model:v1"
    parsed = parse(uri)
    with patch("aiohttp.ClientSession.get") as mock_get:
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = {
            "category": "model",
            "name": "model",
            "version": "v1",
            # harbor_ref intentionally omitted
            "size_bytes": 123,
            "checksum": "sha256:abc",
            "metadata": {},
        }
        mock_get.return_value.__aenter__.return_value = resolve_resp

        result = await _pull_on_host(parsed, uri, mock_host)

        assert isinstance(result, _StructuredPullError)
        assert result.status_code == 422
        assert result.error == "resolve_failed"
        assert "missing harbor_ref" in result.detail
        assert result.source_uri == uri


@pytest.mark.anyio
async def test_distribute_model_route_success(mock_host):
    req = DistributeRequest(target_host_id="host-1", source_uri="huggingface://phi-3")

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host),
        patch("app.routes.management.models._check_disk_space", return_value=10.0),
        patch(
            "app.routes.management.models._pull_on_host", return_value=("/path", False)
        ),
    ):

        results = await distribute_model(req)
        assert len(results) == 1
        assert results[0].source_uri == "huggingface://phi-3"
        assert results[0].path == "/path"
        assert results[0].cached is False


@pytest.mark.anyio
async def test_distribute_model_route_batch(mock_host):
    req = DistributeRequest(
        target_host_id="host-1",
        source_uri=["huggingface://phi-3", "huggingface://phi-4"],
    )

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host),
        patch("app.routes.management.models._check_disk_space", return_value=10.0),
        patch("app.routes.management.models._pull_on_host") as mock_pull,
    ):

        mock_pull.side_effect = [("/path1", True), ("/path2", False)]

        results = await distribute_model(req)
        assert len(results) == 2
        assert results[0].source_uri == "huggingface://phi-3"
        assert results[0].path == "/path1"
        assert results[1].source_uri == "huggingface://phi-4"
        assert results[1].path == "/path2"


@pytest.mark.anyio
async def test_distribute_model_partial_results(mock_host):
    """Test that partial results are returned when some items in array fail."""
    req = DistributeRequest(
        target_host_id="host-1",
        source_uri=[
            "huggingface://phi-3",
            "repo://test-model:v1",
            "huggingface://phi-4",
        ],
    )

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host),
        patch("app.routes.management.models._check_disk_space", return_value=10.0),
        patch("app.routes.management.models._pull_on_host") as mock_pull,
    ):

        mock_pull.side_effect = [
            ("/path1", False),
            _StructuredPullError(
                error="resolve_failed",
                detail="Data Repository is unreachable",
                source_uri="repo://test-model:v1",
                status_code=502,
            ),
            ("/path3", True),
        ]

        results = await distribute_model(req)
        # Only successful results returned
        assert len(results) == 2
        assert results[0].source_uri == "huggingface://phi-3"
        assert results[0].path == "/path1"
        assert results[1].source_uri == "huggingface://phi-4"
        assert results[1].path == "/path3"
        assert results[1].cached is True


@pytest.mark.anyio
async def test_distribute_model_single_success(mock_host):
    """Test that a single source_uri returns result even when parse raises (bad URI)."""
    req = DistributeRequest(
        target_host_id="host-1",
        source_uri="huggingface://phi-3",
    )

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host),
        patch("app.routes.management.models._check_disk_space", return_value=10.0),
        patch(
            "app.routes.management.models._pull_on_host", return_value=("/path", False)
        ),
    ):

        results = await distribute_model(req)
        assert len(results) == 1
        assert results[0].source_uri == "huggingface://phi-3"
        assert results[0].path == "/path"
        assert results[0].cached is False


@pytest.mark.anyio
async def test_distribute_model_host_not_found():
    req = DistributeRequest(
        target_host_id="non-existent", source_uri="huggingface://phi-3"
    )

    with patch("app.database.hosts.host_db.get_host", return_value=None):
        with pytest.raises(HTTPException) as exc:
            await distribute_model(req)
        assert exc.value.status_code == 404


@pytest.mark.anyio
async def test_distribute_model_insufficient_disk(mock_host):
    req = DistributeRequest(target_host_id="host-1", source_uri="huggingface://phi-3")

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host),
        patch("app.routes.management.models._check_disk_space", return_value=2.0),
    ):  # Below 5.0 GB

        with pytest.raises(HTTPException) as exc:
            await distribute_model(req)
        assert exc.value.status_code == 507
        assert "Insufficient disk" in exc.value.detail


@pytest.mark.anyio
async def test_distribute_model_disk_check_failure_proceeds(mock_host):
    req = DistributeRequest(target_host_id="host-1", source_uri="huggingface://phi-3")

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host),
        patch("app.routes.management.models._check_disk_space", return_value=None),
        patch(
            "app.routes.management.models._pull_on_host", return_value=("/path", True)
        ),
    ):

        results = await distribute_model(req)
        assert len(results) == 1
        assert results[0].path == "/path"


@pytest.mark.anyio
async def test_pull_on_host_404_structured_error(mock_host):
    """Test that 404 errors from host are propagated with structured format (B-008)."""
    uri = "huggingface://nonexistent/repo"
    parsed = parse(uri)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 404
        # Host returns structured error format
        mock_resp.json.return_value = {
            "error": "not_found",
            "detail": "HuggingFace repository not found: 404 Client Error...",
            "source_uri": "huggingface://nonexistent/repo",
            "status_code": 404,
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await _pull_on_host(parsed, uri, mock_host)
        assert isinstance(result, _StructuredPullError)
        assert result.status_code == 404
        assert result.error == "not_found"
        assert result.detail == "HuggingFace repository not found: 404 Client Error..."
        assert result.source_uri == "huggingface://nonexistent/repo"

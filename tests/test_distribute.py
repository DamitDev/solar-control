import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from app.models import Host, HostStatus
from app.routes.management.models import (
    _pull_on_host,
    _check_disk_space,
    distribute_model,
    DistributeRequest,
)
from app.model_resolvers.parser import parse


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

        with pytest.raises(HTTPException) as exc:
            await _pull_on_host(parsed, uri, mock_host)
        assert exc.value.status_code == 404
        assert "Model not found" in exc.value.detail


@pytest.mark.anyio
async def test_pull_on_host_507_propagated(mock_host):
    uri = "huggingface://microsoft/phi-3"
    parsed = parse(uri)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 507
        mock_resp.json.return_value = {"error": "Disk full"}
        mock_post.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await _pull_on_host(parsed, uri, mock_host)
        assert exc.value.status_code == 507
        assert "Disk full" in exc.value.detail


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

    with pytest.raises(HTTPException) as exc:
        await _pull_on_host(parsed, uri, mock_host)
    assert exc.value.status_code == 400
    assert "Cannot distribute local:// URIs" in exc.value.detail


@pytest.mark.anyio
async def test_pull_on_host_repo_uri(mock_host):
    uri = "repo://model:v1"
    parsed = parse(uri)

    with pytest.raises(HTTPException) as exc:
        await _pull_on_host(parsed, uri, mock_host)
    assert exc.value.status_code == 501
    assert "repo:// distribution requires Data Repository" in exc.value.detail


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

        with pytest.raises(HTTPException) as exc:
            await _pull_on_host(parsed, uri, mock_host)
        assert exc.value.status_code == 404
        # The HTTPException should have the structured error as detail
        assert exc.value.detail == {
            "error": "not_found",
            "detail": "HuggingFace repository not found: 404 Client Error...",
            "source_uri": "huggingface://nonexistent/repo",
            "status_code": 404,
        }

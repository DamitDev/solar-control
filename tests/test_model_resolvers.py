import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from app.model_resolvers.parser import parse, RepoURI, HuggingFaceURI, LocalURI
from app.model_resolvers.dispatcher import resolve


def test_parser_local():
    # Absolute path
    p = parse("local:///opt/models/iris.gguf")
    assert isinstance(p, LocalURI)
    assert p.path == "/opt/models/iris.gguf"

    # Relative path
    p = parse("local://relative/path")
    assert isinstance(p, LocalURI)
    assert p.path == "relative/path"


def test_parser_huggingface():
    p = parse("huggingface://microsoft/phi-3")
    assert isinstance(p, HuggingFaceURI)
    assert p.model_id == "microsoft/phi-3"

    p = parse("huggingface://phi-3")
    assert isinstance(p, HuggingFaceURI)
    assert p.model_id == "phi-3"


def test_parser_repo():
    p = parse("repo://iris-osl:v3")
    assert isinstance(p, RepoURI)
    assert p.name == "iris-osl"
    assert p.version == "v3"


def test_parser_errors():
    with pytest.raises(HTTPException) as exc:
        parse("invalid://something")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        parse("repo://iris-osl")  # missing version
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        parse("huggingface://")  # missing model_id
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        parse("local://")  # missing path
    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_resolve_local():
    uri = "local:///opt/models/iris.gguf"
    resolved = await resolve(uri, "http://host:8000", "key")
    assert resolved == uri


@pytest.mark.anyio
async def test_resolve_huggingface_success():
    uri = "huggingface://microsoft/phi-3"
    host_url = "http://host:8000"
    host_key = "key"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"path": "/opt/solar/models/hf--microsoft--phi-3"}
        mock_post.return_value.__aenter__.return_value = mock_resp

        resolved = await resolve(uri, host_url, host_key)

        assert resolved == "local:///opt/solar/models/hf--microsoft--phi-3"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{host_url}/models/pull"
        assert kwargs["json"]["model_id"] == "microsoft/phi-3"
        assert kwargs["json"]["source_uri"] == uri


@pytest.mark.anyio
async def test_resolve_huggingface_404():
    uri = "huggingface://microsoft/phi-3"
    host_url = "http://host:8000"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.json.side_effect = Exception("Not JSON")
        mock_resp.text.return_value = "Not Found"
        mock_post.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, host_url, "key")

        assert exc.value.status_code == 404
        assert (
            f"Model pull failed on host '{host_url}' [404]: Not Found"
            in exc.value.detail
        )


@pytest.mark.anyio
async def test_resolve_huggingface_507():
    uri = "huggingface://microsoft/phi-3"
    host_url = "http://host:8000"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 507
        mock_resp.json.return_value = {"error": "Insufficient disk space"}
        mock_post.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, host_url, "key")

        assert exc.value.status_code == 507
        assert "Insufficient disk space" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_huggingface_no_path():
    uri = "huggingface://microsoft/phi-3"
    host_url = "http://host:8000"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {}  # Missing "path"
        mock_post.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, host_url, "key")

        assert exc.value.status_code == 502
        assert "no path for model pull" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_huggingface_connection_error():
    uri = "huggingface://microsoft/phi-3"
    import aiohttp

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = aiohttp.ClientConnectionError("Connection refused")

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, "http://host:8000", "key")

        assert exc.value.status_code == 502
        assert "is unreachable" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_huggingface_timeout():
    uri = "huggingface://microsoft/phi-3"
    import asyncio

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = asyncio.TimeoutError()

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, "http://host:8000", "key")

        assert exc.value.status_code == 502
        assert "is unreachable" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_huggingface_structured_error():
    uri = "huggingface://microsoft/phi-3"
    host_url = "http://host:8000"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.json.return_value = {"detail": "Invalid API Key"}
        mock_post.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, host_url, "key")

        assert exc.value.status_code == 502  # 401 is not in PROPAGATED_CODES
        assert "Invalid API Key" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_stub():
    uri = "repo://iris-osl:v3"
    with pytest.raises(HTTPException) as exc:
        await resolve(uri, "http://host:8000", "key")

    assert exc.value.status_code == 501
    assert (
        "repo:// resolver not yet available. Data Repository integration will be completed in Phase 1."
        in exc.value.detail
    )


@pytest.mark.anyio
async def test_resolve_repo_stub_includes_uri():
    uri = "repo://my-model:v1.2.3"
    with pytest.raises(HTTPException) as exc:
        await resolve(uri, "http://host:8000", "key")

    assert exc.value.status_code == 501
    assert f"URI: {uri}" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_invalid_missing_version():
    # Dispatcher calls parse first, which raises 400
    uri = "repo://iris-osl"
    with pytest.raises(HTTPException) as exc:
        await resolve(uri, "http://host:8000", "key")

    assert exc.value.status_code == 400
    assert "Missing version" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_invalid_empty_name():
    uri = "repo://:v3"
    with pytest.raises(HTTPException) as exc:
        await resolve(uri, "http://host:8000", "key")

    assert exc.value.status_code == 400
    assert "Name and version must be non-empty" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_invalid_empty_version():
    uri = "repo://iris-osl:"
    with pytest.raises(HTTPException) as exc:
        await resolve(uri, "http://host:8000", "key")

    assert exc.value.status_code == 400
    assert "Name and version must be non-empty" in exc.value.detail

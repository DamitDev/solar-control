import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from app.model_resolvers.parser import parse, RepoURI, HuggingFaceURI, LocalURI
from app.model_resolvers.dispatcher import resolve


def _repo_resolve_payload(**overrides):
    """Build a canonical Data Repository /api/resolve response."""
    payload = {
        "category": "model",
        "name": "iris-osl",
        "version": "v3",
        "harbor_ref": "imgrepo.damit.hu/supernova/iris-osl:v3",
        "size_bytes": 123,
        "checksum": "sha256:abc",
        "metadata": {},
        "created_at": "2026-04-29T10:00:00Z",
    }
    payload.update(overrides)
    return payload


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
    assert p.subpath == ""


def test_parser_repo_with_subpath():
    p = parse("repo://iris-osl:v3/model.gguf")
    assert isinstance(p, RepoURI)
    assert p.name == "iris-osl"
    assert p.version == "v3"
    assert p.subpath == "model.gguf"


def test_parser_repo_with_nested_subpath():
    p = parse("repo://iris-osl:v3/subdir/model.gguf")
    assert isinstance(p, RepoURI)
    assert p.name == "iris-osl"
    assert p.version == "v3"
    assert p.subpath == "subdir/model.gguf"


def test_parser_repo_empty_subpath_errors():
    with pytest.raises(HTTPException) as exc:
        parse("repo://iris-osl:v3/")
    assert exc.value.status_code == 400


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
async def test_resolve_repo_success(repo_settings):
    uri = "repo://iris-osl:v3"
    host_url = "http://host:8000"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload()
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {
            "path": "/opt/solar/models/repo--iris-osl--v3",
            "cached": True,
        }
        mock_post.return_value.__aenter__.return_value = pull_resp

        resolved = await resolve(uri, host_url, "key")

        assert resolved == "local:///opt/solar/models/repo--iris-osl--v3"
        mock_get.assert_called_once()
        mock_post.assert_called_once()
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["json"]["source"] == "harbor"
        assert (
            post_kwargs["json"]["harbor_ref"]
            == "imgrepo.damit.hu/supernova/iris-osl:v3"
        )
        assert post_kwargs["json"]["source_uri"] == uri
        assert post_kwargs["json"]["digest"] == "sha256:abc"
        assert post_kwargs["json"]["category"] == "model"
        assert post_kwargs["json"]["name"] == "iris-osl"
        assert post_kwargs["json"]["version"] == "v3"
        assert post_kwargs["json"]["size_bytes"] == 123
        assert post_kwargs["json"]["checksum"] == "sha256:abc"
        assert post_kwargs["json"]["metadata"] == {}


@pytest.mark.anyio
async def test_resolve_repo_forwards_backend_type(repo_settings):
    """llamacpp backend is forwarded in the pull payload for GGUF selection."""
    uri = "repo://iris-osl:v3"
    host_url = "http://host:8000"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload()
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {
            "path": "/opt/solar/models/repo--iris-osl--v3/model.gguf",
            "cached": True,
        }
        mock_post.return_value.__aenter__.return_value = pull_resp

        resolved = await resolve(uri, host_url, "key", backend_type="llamacpp")

        assert resolved == "local:///opt/solar/models/repo--iris-osl--v3/model.gguf"
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["json"]["backend_type"] == "llamacpp"


@pytest.mark.anyio
async def test_resolve_repo_default_backend_type_is_none(repo_settings):
    """No backend declared -> payload carries None (no GGUF selection)."""
    uri = "repo://iris-osl:v3"
    host_url = "http://host:8000"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload()
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {
            "path": "/opt/solar/models/repo--iris-osl--v3",
            "cached": True,
        }
        mock_post.return_value.__aenter__.return_value = pull_resp

        await resolve(uri, host_url, "key")

        _, post_kwargs = mock_post.call_args
        assert post_kwargs["json"]["backend_type"] is None


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


@pytest.mark.anyio
async def test_resolve_repo_not_found(repo_settings):
    uri = "repo://iris-osl:v99"

    with patch("aiohttp.ClientSession.get") as mock_get:
        resolve_resp = AsyncMock()
        resolve_resp.status = 404
        resolve_resp.json.return_value = {"detail": "Version not found"}
        mock_get.return_value.__aenter__.return_value = resolve_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, "http://host:8000", "key")

        assert exc.value.status_code == 404
        assert "Version not found" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_pull_failed(repo_settings):
    uri = "repo://iris-osl:v3"
    host_url = "http://host:8000"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload()
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 404
        pull_resp.json.side_effect = Exception("Not JSON")
        pull_resp.text.return_value = "Not Found"
        mock_post.return_value.__aenter__.return_value = pull_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, host_url, "key")

        assert exc.value.status_code == 404
        assert (
            f"Model pull failed on host '{host_url}' [404]: Not Found"
            in exc.value.detail
        )


@pytest.mark.anyio
async def test_resolve_repo_pull_507_propagates(repo_settings):
    """Insufficient-disk responses from the host propagate as 507, not 502."""
    uri = "repo://iris-osl:v3"
    host_url = "http://host:8000"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload()
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 507
        pull_resp.json.return_value = {"detail": "Insufficient disk space"}
        mock_post.return_value.__aenter__.return_value = pull_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, host_url, "key")

        assert exc.value.status_code == 507
        assert "Insufficient disk space" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_data_repo_unreachable(repo_settings):
    uri = "repo://iris-osl:v3"

    with patch("aiohttp.ClientSession.get") as mock_get:
        import aiohttp

        mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, "http://host:8000", "key")

        assert exc.value.status_code == 502
        assert "Data Repository is unreachable" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_data_repo_url_unconfigured(repo_settings):
    """An unset DATA_REPOSITORY_URL surfaces as 500, not a silent failure."""
    repo_settings.data_repository_url = ""

    with pytest.raises(HTTPException) as exc:
        await resolve("repo://iris-osl:v3", "http://host:8000", "key")

    assert exc.value.status_code == 500
    assert "DATA_REPOSITORY_URL is not configured" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_rejects_dataset(repo_settings):
    uri = "repo://iris-tickets:2026-03"

    with patch("aiohttp.ClientSession.get") as mock_get:
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload(
            category="dataset",
            name="iris-tickets",
            version="2026-03",
            harbor_ref="imgrepo.damit.hu/supernova/iris-tickets:2026-03",
        )
        mock_get.return_value.__aenter__.return_value = resolve_resp

        with pytest.raises(HTTPException) as exc:
            await resolve(uri, "http://host:8000", "key")

        assert exc.value.status_code == 422
        assert "not a deployable model" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_missing_harbor_ref_is_422(repo_settings):
    """Resolve responses without a harbor_ref are a per-item 422, not 502.

    A 502 here would abort the whole /distribute batch because the route
    re-raises 5xx. 422 keeps it as a structured per-item failure.
    """
    with patch("aiohttp.ClientSession.get") as mock_get:
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload(harbor_ref=None)
        mock_get.return_value.__aenter__.return_value = resolve_resp

        with pytest.raises(HTTPException) as exc:
            await resolve("repo://iris-osl:v3", "http://host:8000", "key")

        assert exc.value.status_code == 422
        assert "missing harbor_ref" in exc.value.detail


@pytest.mark.anyio
async def test_resolve_repo_latest_forwarded_verbatim(repo_settings):
    """``repo://name:latest`` must be passed through unchanged to Data Repository.

    Per the issue: "do not invent separate latest behavior in Solar Control".
    """
    uri = "repo://iris-osl:latest"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        # Data Repository resolves "latest" -> a concrete version itself.
        resolve_resp.json.return_value = _repo_resolve_payload(version="v7")
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {"path": "/opt/solar/models/repo--iris-osl--v7"}
        mock_post.return_value.__aenter__.return_value = pull_resp

        await resolve(uri, "http://host:8000", "key")

        _, get_kwargs = mock_get.call_args
        assert get_kwargs["params"] == {"uri": "repo://iris-osl:latest"}
        # And the original source_uri is what the host sees, not a rewritten one.
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["json"]["source_uri"] == "repo://iris-osl:latest"
        assert post_kwargs["json"]["version"] == "v7"


@pytest.mark.anyio
async def test_resolve_repo_subpath_stripped_for_data_repo(repo_settings):
    """``repo://name:version/file.gguf`` queries Data Repo without the subpath
    but forwards the full URI (subpath included) to the host pull."""
    uri = "repo://iris-osl:v3/model.gguf"

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        resolve_resp = AsyncMock()
        resolve_resp.status = 200
        resolve_resp.json.return_value = _repo_resolve_payload()
        mock_get.return_value.__aenter__.return_value = resolve_resp

        pull_resp = AsyncMock()
        pull_resp.status = 200
        pull_resp.json.return_value = {
            "path": "/opt/solar/models/repo--iris-osl--v3/model.gguf"
        }
        mock_post.return_value.__aenter__.return_value = pull_resp

        resolved = await resolve(uri, "http://host:8000", "key")

        # Data Repo sees the base URI only.
        _, get_kwargs = mock_get.call_args
        assert get_kwargs["params"] == {"uri": "repo://iris-osl:v3"}
        # Host pull receives the full URI so it can select the file.
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["json"]["source_uri"] == "repo://iris-osl:v3/model.gguf"
        # Resolved local:// URI points at the file inside the directory.
        assert resolved == "local:///opt/solar/models/repo--iris-osl--v3/model.gguf"

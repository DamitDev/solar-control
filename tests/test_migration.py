"""Tests for instance migration (S-037)."""

import copy
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.models import Host, HostStatus
from app.models.migration import MigrationResult
from app.services.migration import (
    capture_instance_config,
    check_one_replica_per_host,
    create_instance_on_host,
    disown_source_instance,
    ensure_model_on_target,
    execute_migration,
    stop_source_instance,
    validate_target_fitness,
)

# ── Helpers ──────────────────────────────────────────────────────


def _async_mock_host(host: Host):
    """Return an AsyncMock that resolves to *host*."""
    return AsyncMock(return_value=host)


def _async_mock_instances(instances: list[dict]):
    """Return an AsyncMock that resolves to *instances*."""
    return AsyncMock(return_value=instances)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def source_host() -> Host:
    return Host(
        id="host-src",
        name="Source Host",
        url="http://source:8000",
        api_key="key-src",
        status=HostStatus.ONLINE,
    )


@pytest.fixture
def target_host() -> Host:
    return Host(
        id="host-tgt",
        name="Target Host",
        url="http://target:8000",
        api_key="key-tgt",
        status=HostStatus.ONLINE,
        roles=["inference"],
    )


@pytest.fixture
def instance_config() -> dict:
    return {
        "instance_id": "inst-1",
        "config": {
            "alias": "test-model:v1",
            "model_source": "repo://test-model:v1",
            "backend_type": "huggingface_classification",
            "priority": "staging",
            "max_length": 512,
            "labels": ["osl"],
        },
    }


# ── create_instance_on_host ──────────────────────────────────────


@pytest.mark.anyio
async def test_create_instance_on_host_success(target_host):
    with (
        patch("app.services.migration.resolve") as mock_resolve,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        mock_resolve.return_value = "repo://test-model:v1"
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"instance_id": "new-inst", "status": "running"}
        )
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await create_instance_on_host(
            target_host,
            {"model_source": "repo://test-model:v1", "priority": "staging"},
        )
        assert result["instance_id"] == "new-inst"


@pytest.mark.anyio
async def test_create_instance_on_host_invalid_priority(target_host):
    with pytest.raises(HTTPException) as exc:
        await create_instance_on_host(target_host, {"priority": "invalid"})
    assert exc.value.status_code == 422
    assert "Invalid priority" in exc.value.detail


@pytest.mark.anyio
async def test_create_instance_on_host_unreachable(target_host):
    with (
        patch("app.services.migration.resolve"),
        patch(
            "aiohttp.ClientSession.post",
            side_effect=Exception("Connection refused"),
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await create_instance_on_host(target_host, {})
        assert exc.value.status_code == 502


# ── capture_instance_config ──────────────────────────────────────


@pytest.mark.anyio
async def test_capture_config_from_redis(source_host, instance_config):
    with patch("app.services.migration.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[instance_config])

        result = await capture_instance_config(source_host, "inst-1")
        assert result["instance_id"] == "inst-1"


@pytest.mark.anyio
async def test_capture_config_fallback_http(source_host, instance_config):
    with (
        patch("app.services.migration.host_store") as mock_store,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_store.get_host_instances = AsyncMock(return_value=[])
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[instance_config])
        mock_get.return_value.__aenter__.return_value = mock_resp

        result = await capture_instance_config(source_host, "inst-1")
        assert result["instance_id"] == "inst-1"


@pytest.mark.anyio
async def test_capture_config_not_found(source_host):
    with (
        patch("app.services.migration.host_store") as mock_store,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_store.get_host_instances = AsyncMock(return_value=[])
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[])
        mock_get.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await capture_instance_config(source_host, "inst-1")
        assert exc.value.status_code == 404


# ── check_one_replica_per_host ───────────────────────────────────


@pytest.mark.anyio
async def test_check_one_replica_no_conflict(target_host):
    with patch("app.services.migration.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(
            return_value=[{"config": {"alias": "other-model:v1"}}]
        )
        # Should not raise
        await check_one_replica_per_host(target_host, "test-model:v1")


@pytest.mark.anyio
async def test_check_one_replica_conflict(target_host):
    with patch("app.services.migration.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(
            return_value=[{"config": {"alias": "test-model:v1"}}]
        )
        with pytest.raises(HTTPException) as exc:
            await check_one_replica_per_host(target_host, "test-model:v1")
        assert exc.value.status_code == 409


# ── validate_target_fitness ──────────────────────────────────────


@pytest.mark.anyio
async def test_validate_target_no_inference_role():
    host = Host(
        id="h1",
        name="No Role",
        url="http://h:8000",
        api_key="k",
        roles=["training"],
        status=HostStatus.ONLINE,
    )
    with pytest.raises(HTTPException) as exc:
        await validate_target_fitness(host, {"priority": "staging"})
    assert exc.value.status_code == 422
    assert "inference" in exc.value.detail


@pytest.mark.anyio
async def test_validate_target_production_refused(target_host):
    with pytest.raises(HTTPException) as exc:
        await validate_target_fitness(
            target_host,
            {"priority": "production"},
            allow_production=False,
        )
    assert exc.value.status_code == 422
    assert "production" in exc.value.detail.lower()


@pytest.mark.anyio
async def test_validate_target_production_allowed(target_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        # Should not raise
        await validate_target_fitness(
            target_host,
            {"priority": "production"},
            allow_production=True,
        )


@pytest.mark.anyio
async def test_validate_target_insufficient_disk(target_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 1.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await validate_target_fitness(target_host, {"priority": "staging"})
        assert exc.value.status_code == 507


@pytest.mark.anyio
async def test_validate_target_gpu_type_mismatch():
    """GPU type mismatch is logged but not rejected (GGUF is portable)."""
    host = Host(
        id="h-cuda",
        name="CUDA Host",
        url="http://h:8000",
        api_key="k",
        roles=["inference"],
        gpu_type="nvidia_cuda",
        status=HostStatus.ONLINE,
    )
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        # Should not raise — GPU type difference is logged, not blocked
        await validate_target_fitness(
            host,
            {"priority": "staging"},
            source_gpu_type="apple_silicon",
        )


@pytest.mark.anyio
async def test_validate_target_gpu_type_match():
    """Same GPU type on both hosts passes."""
    host = Host(
        id="h-cuda",
        name="CUDA Host",
        url="http://h:8000",
        api_key="k",
        roles=["inference"],
        gpu_type="nvidia_cuda",
        status=HostStatus.ONLINE,
    )
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        # Should not raise
        await validate_target_fitness(
            host,
            {"priority": "staging"},
            source_gpu_type="nvidia_cuda",
        )


@pytest.mark.anyio
async def test_validate_target_insufficient_vram(target_host):
    """Target host has enough disk but VRAM is too low."""
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "disk": {"available_gb": 100.0},
                "memory": [
                    {"memory_type": "VRAM", "total_gb": 8.0, "available_gb": 0.5},
                ],
            }
        )
        mock_get.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await validate_target_fitness(target_host, {"priority": "staging"})
        assert exc.value.status_code == 507
        assert "vram" in exc.value.detail.lower()


# ── ensure_model_on_target ───────────────────────────────────────


@pytest.mark.anyio
async def test_ensure_model_on_target_success(target_host):
    mock_pull = AsyncMock(return_value=("/models/repo--test--v1", True))
    with patch("app.routes.management.models._pull_on_host", mock_pull):
        path, cached = await ensure_model_on_target(target_host, "repo://test-model:v1")
        assert path == "/models/repo--test--v1"
        assert cached is True


@pytest.mark.anyio
async def test_ensure_model_on_target_failure(target_host):
    from app.routes.management.models import _StructuredPullError

    mock_pull = AsyncMock(
        return_value=_StructuredPullError(
            error="not_found",
            detail="Artifact not found",
            source_uri="repo://bad:v1",
            status_code=404,
        )
    )
    with patch("app.routes.management.models._pull_on_host", mock_pull):
        with pytest.raises(HTTPException) as exc:
            await ensure_model_on_target(target_host, "repo://bad:v1")
        assert exc.value.status_code == 404


# ── stop_source_instance ─────────────────────────────────────────


@pytest.mark.anyio
async def test_stop_source_instance_success(source_host):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"status": "stopped"})
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await stop_source_instance(source_host, "inst-1")
        assert result["status"] == "stopped"


@pytest.mark.anyio
async def test_disown_source_redis_failure_raises_http_exception(
    source_host, instance_config
):
    """Redis marker-clear failure must surface as HTTPException, not escape."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    with (
        patch("aiohttp.ClientSession.put") as mock_put,
        patch("app.services.migration.host_store") as mock_store,
    ):
        mock_put.return_value.__aenter__.return_value = mock_resp
        mock_store.get_host_instances = AsyncMock(
            side_effect=Exception("Redis connection lost")
        )

        with pytest.raises(HTTPException) as exc_info:
            await disown_source_instance(
                source_host, "inst-1", instance_config["config"]
            )

        assert exc_info.value.status_code == 502
        assert "Redis" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_stop_source_instance_unreachable(source_host):
    with patch(
        "aiohttp.ClientSession.post",
        side_effect=Exception("Connection refused"),
    ):
        with pytest.raises(HTTPException) as exc:
            await stop_source_instance(source_host, "inst-1")
        assert exc.value.status_code == 502


# ── disown_source_instance ───────────────────────────────────────


@pytest.mark.anyio
async def test_disown_source_instance_success(source_host, instance_config):
    """Disown clears markers on the host and in the Redis cache."""
    with (
        patch("aiohttp.ClientSession.put") as mock_put,
        patch("app.services.migration.host_store") as mock_store,
    ):
        put_resp = AsyncMock()
        put_resp.status = 200
        mock_put.return_value.__aenter__.return_value = put_resp

        instances = [
            {
                "instance_id": "inst-1",
                "config": {
                    "alias": "test-model:v1",
                    "model_source": "repo://test-model:v1",
                },
                "managed_by": "intent",
                "intent_id": "intent-1",
            }
        ]
        mock_store.get_host_instances = AsyncMock(return_value=instances)
        mock_store.set_host_instances = AsyncMock()

        await disown_source_instance(source_host, "inst-1", instance_config["config"])

        # Host PUT clears markers
        _, put_kwargs = mock_put.call_args
        assert put_kwargs["json"]["managed_by"] is None
        assert put_kwargs["json"]["intent_id"] is None

        # Redis cache updated
        mock_store.set_host_instances.assert_awaited_once()
        stored = mock_store.set_host_instances.call_args[0][1]
        assert stored[0].get("managed_by") is None
        assert stored[0].get("intent_id") is None
        assert stored[0]["config"].get("managed_by") is None
        assert stored[0]["config"].get("intent_id") is None


@pytest.mark.anyio
async def test_disown_source_flat_ws_format_no_circular_reference(
    source_host, instance_config
):
    """Flat WS-format Redis entries (no nested 'config' key) must not crash.

    Regression: ``inst.get("config", inst)`` fell back to the instance dict
    itself, then ``inst["config"] = cfg`` created a self-reference that blew
    up ``json.dumps`` in ``set_host_instances`` with "Circular reference
    detected", leaving intent markers in Redis and causing the reconciler
    to fight the stopped source instance (RECREATE -> /stop spam).
    """
    with (
        patch("aiohttp.ClientSession.put") as mock_put,
        patch("app.services.migration.host_store") as mock_store,
    ):
        put_resp = AsyncMock()
        put_resp.status = 200
        mock_put.return_value.__aenter__.return_value = put_resp

        # Flat WS format: markers at top level, NO nested "config" key
        instances = [
            {
                "instance_id": "inst-1",
                "alias": "test-model:v1",
                "model_source": "repo://test-model:v1",
                "status": "stopped",
                "port": 3500,
                "managed_by": "intent",
                "intent_id": "intent-1",
            }
        ]
        mock_store.get_host_instances = AsyncMock(return_value=instances)
        mock_store.set_host_instances = AsyncMock()

        await disown_source_instance(source_host, "inst-1", instance_config["config"])

        # No self-reference: stored entry must remain JSON-serializable
        import json

        stored = mock_store.set_host_instances.call_args[0][1]
        json.dumps(stored)  # must not raise Circular reference detected

        # Top-level markers cleared (flat format has no nested config)
        assert stored[0].get("managed_by") is None
        assert stored[0].get("intent_id") is None
        assert "config" not in stored[0] or isinstance(stored[0]["config"], dict)


@pytest.mark.anyio
async def test_disown_source_instance_unreachable(source_host, instance_config):
    with patch(
        "aiohttp.ClientSession.put",
        side_effect=Exception("Connection refused"),
    ):
        with pytest.raises(HTTPException) as exc:
            await disown_source_instance(
                source_host, "inst-1", instance_config["config"]
            )
        assert exc.value.status_code == 502


# ── execute_migration (integration) ───────────────────────────────


@pytest.mark.anyio
async def test_migrate_happy_path(source_host, target_host, instance_config):
    """Full happy-path migration: all steps succeed."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
        patch("app.services.migration.ensure_model_on_target") as mock_ensure,
        patch("app.services.migration.stop_source_instance") as mock_stop,
        patch("app.services.migration.create_instance_on_host") as mock_create,
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.put") as mock_put,
    ):
        # DB: both hosts exist
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )

        # Redis: source has the instance, target has no conflicts
        mock_store.get_host_instances = AsyncMock(
            side_effect=lambda hid: [instance_config] if hid == "host-src" else []
        )
        mock_store.set_host_instances = AsyncMock()
        mock_train_check.return_value = None

        # Disk check passes
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        # Disown PUT succeeds
        put_resp = AsyncMock()
        put_resp.status = 200
        mock_put.return_value.__aenter__.return_value = put_resp

        mock_ensure.return_value = (
            "/models/repo--test--v1",
            True,
        )
        mock_stop.return_value = {"status": "stopped"}
        mock_create.return_value = {
            "instance": {"id": "new-inst", "status": "stopped"},
            "message": "created",
        }

        result = await execute_migration(
            instance_id="inst-1",
            source_host_id="host-src",
            target_host_id="host-tgt",
        )

        assert isinstance(result, MigrationResult)
        assert result.status == "completed"
        assert result.alias == "test-model:v1"
        assert result.source_host_id == "host-src"
        assert result.target_host_id == "host-tgt"
        assert result.target_instance_id == "new-inst"

        # All 9 steps should be ok
        assert len(result.steps) == 9
        for step in result.steps:
            assert step.status == "ok", f"Step '{step.step}' failed"
        step_names = [s.step for s in result.steps]
        assert "disown_source" in step_names

        # Disown PUT clears markers on the source host
        _, put_kwargs = mock_put.call_args
        assert put_kwargs["json"]["managed_by"] is None
        assert put_kwargs["json"]["intent_id"] is None


@pytest.mark.anyio
async def test_migrate_target_keeps_managed_markers(
    source_host, target_host, instance_config
):
    """Target is created managed (G3): managed_by/intent_id preserved in
    the create wrapper so the reconciler adopts and restarts it (§8.2)."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
        patch("app.services.migration.ensure_model_on_target") as mock_ensure,
        patch("app.services.migration.stop_source_instance") as mock_stop,
        patch("app.services.migration.create_instance_on_host") as mock_create,
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.put") as mock_put,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )
        # Source instance carries the ownership markers (managed by intent)
        captured = {
            **instance_config,
            "managed_by": "intent",
            "intent_id": "intent-1",
        }
        # Fresh copy per fetch: disown_source_instance pops markers from
        # its Redis fetch; production Redis returns new objects per call,
        # so the captured dict must not be aliased by the mock.
        mock_store.get_host_instances = AsyncMock(
            side_effect=lambda hid: (
                [copy.deepcopy(captured)] if hid == "host-src" else []
            )
        )
        mock_store.set_host_instances = AsyncMock()
        mock_train_check.return_value = None

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        put_resp = AsyncMock()
        put_resp.status = 200
        mock_put.return_value.__aenter__.return_value = put_resp

        mock_ensure.return_value = ("/models/repo--test--v1", True)
        mock_stop.return_value = {"status": "stopped"}
        mock_create.return_value = {
            "instance": {"id": "new-inst", "status": "stopped"},
            "message": "created",
        }

        result = await execute_migration(
            instance_id="inst-1",
            source_host_id="host-src",
            target_host_id="host-tgt",
        )

        assert result.status == "completed"
        # No disown_target step — target stays managed
        step_names = [s.step for s in result.steps]
        assert "disown_target" not in step_names
        # Create wrapper carries the ownership markers at top level (G3)
        create_wrapper = mock_create.call_args[0][1]
        assert create_wrapper["managed_by"] == "intent"
        assert create_wrapper["intent_id"] == "intent-1"
        assert create_wrapper["priority"] == "staging"
        # Source disown PUT still happened (source stays stopped + unmanaged)
        assert len(result.steps) == 9


@pytest.mark.anyio
async def test_migrate_source_host_not_found(target_host):
    with patch("app.services.migration.host_db") as mock_db:
        mock_db.get_host = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-1",
                source_host_id="host-src",
                target_host_id="host-tgt",
            )
        assert exc.value.status_code == 404
        assert "Source host" in exc.value.detail


@pytest.mark.anyio
async def test_migrate_target_host_not_found(source_host):
    with patch("app.services.migration.host_db") as mock_db:
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: source_host if hid == "host-src" else None
        )

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-1",
                source_host_id="host-src",
                target_host_id="host-tgt",
            )
        assert exc.value.status_code == 404
        assert "Target host" in exc.value.detail


@pytest.mark.anyio
async def test_migrate_same_host_rejected(source_host):
    """Same source and target host is rejected with 422."""
    with patch("app.services.migration.host_db") as mock_db:
        mock_db.get_host = AsyncMock(return_value=source_host)

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-1",
                source_host_id="host-src",
                target_host_id="host-src",
            )
        assert exc.value.status_code == 422
        assert "same" in exc.value.detail.lower()


@pytest.mark.anyio
async def test_migrate_alias_conflict(source_host, target_host, instance_config):
    """Target host already runs an instance with the same alias."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )

        # Source: instance exists; Target: already has conflict
        mock_store.get_host_instances = AsyncMock(
            side_effect=lambda hid: (
                [instance_config]
                if hid == "host-src"
                else [{"config": {"alias": "test-model:v1"}}]
            )
        )
        mock_train_check.return_value = None

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-1",
                source_host_id="host-src",
                target_host_id="host-tgt",
            )
        assert exc.value.status_code == 409


@pytest.mark.anyio
async def test_migrate_production_refused(source_host, target_host):
    """Production instance migration refused without allow_production."""
    prod_config = {
        "instance_id": "inst-prod",
        "config": {
            "alias": "prod-model:v1",
            "model_source": "repo://prod:v1",
            "backend_type": "huggingface_classification",
            "priority": "production",
        },
    }

    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )
        mock_store.get_host_instances = AsyncMock(return_value=[prod_config])
        mock_train_check.return_value = None

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-prod",
                source_host_id="host-src",
                target_host_id="host-tgt",
                allow_production=False,
            )
        assert exc.value.status_code == 422
        assert "production" in exc.value.detail.lower()


@pytest.mark.anyio
async def test_migrate_model_pull_fails(source_host, target_host, instance_config):
    """Model pull on target host fails — returns failed MigrationResult."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
        patch("app.services.migration.ensure_model_on_target") as mock_ensure,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )
        mock_store.get_host_instances = AsyncMock(
            side_effect=lambda hid: [instance_config] if hid == "host-src" else []
        )
        mock_train_check.return_value = None

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        mock_ensure.side_effect = HTTPException(
            status_code=404,
            detail="Artifact not found",
        )

        result = await execute_migration(
            instance_id="inst-1",
            source_host_id="host-src",
            target_host_id="host-tgt",
        )

        assert isinstance(result, MigrationResult)
        assert result.status == "failed"
        assert result.error is not None
        assert "Ensure model failed" in result.error
        step_statuses = {s.step: s.status for s in result.steps}
        assert step_statuses.get("ensure_model") == "failed"
        # Steps 1–4 should be ok, step 5 failed, steps 6–7 not reached
        assert step_statuses.get("validate_hosts") == "ok"
        assert step_statuses.get("check_training_jobs") == "ok"
        assert step_statuses.get("capture_config") == "ok"
        assert step_statuses.get("validate_target") == "ok"
        assert step_statuses.get("check_anti_affinity") == "ok"
        assert "stop_source" not in step_statuses
        assert "create_target" not in step_statuses


@pytest.mark.anyio
async def test_migrate_create_target_fails(source_host, target_host, instance_config):
    """Target creation fails after source is stopped — partial result returned."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
        patch("app.services.migration.ensure_model_on_target") as mock_ensure,
        patch("app.services.migration.stop_source_instance") as mock_stop,
        patch("app.services.migration.create_instance_on_host") as mock_create,
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.put") as mock_put,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )
        mock_store.get_host_instances = AsyncMock(
            side_effect=lambda hid: [instance_config] if hid == "host-src" else []
        )
        mock_store.set_host_instances = AsyncMock()
        mock_train_check.return_value = None

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"disk": {"available_gb": 100.0}})
        mock_get.return_value.__aenter__.return_value = mock_resp

        # Disown PUT succeeds
        put_resp = AsyncMock()
        put_resp.status = 200
        mock_put.return_value.__aenter__.return_value = put_resp

        mock_ensure.return_value = (
            "/models/repo--test--v1",
            True,
        )
        mock_stop.return_value = {"status": "stopped"}
        mock_create.side_effect = HTTPException(
            status_code=500, detail="Host internal error"
        )

        result = await execute_migration(
            instance_id="inst-1",
            source_host_id="host-src",
            target_host_id="host-tgt",
        )
        assert isinstance(result, MigrationResult)
        assert result.status == "failed"
        assert result.error is not None
        assert "Create target failed" in result.error
        # Verify steps: stop_source ok, create_target failed
        step_statuses = {s.step: s.status for s in result.steps}
        assert step_statuses.get("stop_source") == "ok"
        assert step_statuses.get("create_target") == "failed"


@pytest.mark.anyio
async def test_migrate_rejected_source_has_training_jobs(
    source_host, target_host, instance_config
):
    """Migration is rejected when source host has active training jobs."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.check_no_active_training") as mock_check,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )

        mock_check.side_effect = HTTPException(
            status_code=409,
            detail=(
                "Source host 'Source Host' has 2 active training job(s): "
                "job-1, job-2. Training jobs are non-migratable workloads."
            ),
        )

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-1",
                source_host_id="host-src",
                target_host_id="host-tgt",
            )
        assert exc.value.status_code == 409
        assert "training" in exc.value.detail.lower()


@pytest.mark.anyio
async def test_migrate_invalid_priority_in_captured_config(source_host, target_host):
    """Migration fails early when captured config has an invalid priority.

    This prevents the bug where a legacy instance with priority='dev'
    would be stopped on the source but fail at create_target step.
    """
    invalid_config = {
        "instance_id": "inst-legacy",
        "config": {
            "alias": "legacy-model:v1",
            "model_source": "repo://legacy:v1",
            "backend_type": "huggingface_classification",
            "priority": "dev",  # invalid — not in {production, staging, ephemeral}
        },
    }

    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
        patch("app.services.migration.check_no_active_training") as mock_train_check,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )
        mock_store.get_host_instances = AsyncMock(return_value=[invalid_config])
        mock_train_check.return_value = None

        with pytest.raises(HTTPException) as exc:
            await execute_migration(
                instance_id="inst-legacy",
                source_host_id="host-src",
                target_host_id="host-tgt",
            )
        assert exc.value.status_code == 422
        assert "invalid priority" in exc.value.detail.lower()
        assert "dev" in exc.value.detail


@pytest.mark.anyio
async def test_training_check_unreachable_rejected(
    source_host, target_host, instance_config
):
    """Migration is rejected when source host is unreachable for training check."""
    with (
        patch("app.services.migration.host_db") as mock_db,
        patch("app.services.migration.host_store") as mock_store,
    ):
        mock_db.get_host = AsyncMock(
            side_effect=lambda hid: (
                source_host
                if hid == "host-src"
                else target_host if hid == "host-tgt" else None
            )
        )
        mock_store.get_host_instances = AsyncMock(return_value=[instance_config])

        # Simulate the training job check hitting a connection error.
        # The real check_no_active_training now raises HTTPException(502)
        # on connectivity failures instead of silently proceeding.
        with patch(
            "app.services.migration.check_no_active_training",
            side_effect=HTTPException(
                status_code=502,
                detail=(
                    "Source host 'Source Host' is unreachable for training "
                    "job check at http://source:8000. Cannot verify no active "
                    "training jobs – migration rejected."
                ),
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await execute_migration(
                    instance_id="inst-1",
                    source_host_id="host-src",
                    target_host_id="host-tgt",
                )
            assert exc.value.status_code == 502
            assert "training" in exc.value.detail.lower()

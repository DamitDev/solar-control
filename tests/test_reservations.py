"""Tests for reservation coordinator (S-038)."""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.models import Host, HostStatus
from app.models.reservation import (
    ReservationRequest,
    ReservationResponse,
)
from app.services.reservation import reserve_resources, release_reservation

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def host_gpu() -> Host:
    return Host(
        id="h-gpu",
        name="GPU Node",
        url="http://gpu:8000",
        api_key="k",
        status=HostStatus.ONLINE,
        gpu_type="nvidia_cuda",
        roles=["training"],
    )


@pytest.fixture
def host_gpu2() -> Host:
    return Host(
        id="h-gpu2",
        name="GPU Node 2",
        url="http://gpu2:8000",
        api_key="k",
        status=HostStatus.ONLINE,
        gpu_type="nvidia_cuda",
        roles=["training"],
    )


@pytest.fixture
def reservation_request() -> ReservationRequest:
    return ReservationRequest(
        requester="supernova",
        job_id="job-001",
        vram_gb=20.0,
        ram_gb=64.0,
        workload_type="training",
        priority="production",
        host_roles=["training"],
    )


def _snap(
    host: Host,
    *,
    vram_avail=80.0,
    ram_avail=200.0,
    disk_avail=500.0,
):
    from app.models import HostResourceSnapshot

    return HostResourceSnapshot(
        host_id=host.id,
        host_name=host.name,
        url=host.url,
        status=host.status,
        roles=host.roles or [],
        gpu_type=host.gpu_type,
        reachable=True,
        vram_total_gb=80.0,
        vram_available_gb=vram_avail,
        ram_total_gb=256.0,
        ram_available_gb=ram_avail,
        disk_total_gb=1000.0,
        disk_available_gb=disk_avail,
        running_instance_count=0,
    )


# ── reserve_resources ───────────────────────────────────────────


@pytest.mark.anyio
async def test_reserve_happy_path(host_gpu, reservation_request):
    """Simple reservation succeeds when capacity is available."""
    with (
        patch("app.services.reservation.host_db") as mock_db,
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
        patch("app.services.reservation._call_host_reserve") as mock_reserve,
        patch("app.services.reservation._store_reservation") as mock_store,
        patch("app.services.placement.host_store") as mock_host_store,
    ):
        mock_db.get_all_hosts = AsyncMock(return_value=[host_gpu])
        mock_fetch.return_value = _snap(host_gpu)
        mock_reserve.return_value = {
            "reservation_id": "host-res-1",
            "expiration": "2026-08-01T00:00:00Z",
        }
        mock_store.return_value = None
        mock_host_store.get_host_instances = AsyncMock(return_value=[])

        result = await reserve_resources(reservation_request)

        assert isinstance(result, ReservationResponse)
        assert result.host_id == "h-gpu"
        assert result.host_name == "GPU Node"
        assert result.migrated is False
        assert result.vram_gb == 20.0
        assert result.ram_gb == 64.0
        mock_reserve.assert_called_once()


@pytest.mark.anyio
async def test_reserve_no_hosts():
    """No hosts registered returns deterministic failure."""
    with patch("app.services.reservation.host_db") as mock_db:
        mock_db.get_all_hosts = AsyncMock(return_value=[])

        with pytest.raises(HTTPException) as exc:
            await reserve_resources(
                ReservationRequest(
                    requester="test",
                    job_id="j1",
                    vram_gb=10.0,
                    host_roles=["training"],
                )
            )

        assert exc.value.status_code == 409
        failure = exc.value.detail
        assert failure["reason"] == "no_hosts"


@pytest.mark.anyio
async def test_reserve_insufficient_capacity(host_gpu, reservation_request):
    """Host exists but doesn't have enough VRAM."""
    with (
        patch("app.services.reservation.host_db") as mock_db,
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
        patch("app.services.placement.host_store") as mock_host_store,
    ):
        mock_db.get_all_hosts = AsyncMock(return_value=[host_gpu])
        mock_fetch.return_value = _snap(host_gpu, vram_avail=5.0)
        mock_host_store.get_host_instances = AsyncMock(return_value=[])

        with pytest.raises(HTTPException) as exc:
            await reserve_resources(reservation_request)

        assert exc.value.status_code == 409
        failure = exc.value.detail
        assert failure["reason"] == "insufficient_capacity"
        assert failure["hosts_checked"] == 1


@pytest.mark.anyio
async def test_reserve_with_migration(host_gpu, host_gpu2, reservation_request):
    """When no host has capacity, migration frees capacity."""
    with (
        patch("app.services.reservation.host_db") as mock_db,
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
        patch("app.services.reservation._call_host_reserve") as mock_reserve,
        patch("app.services.reservation._store_reservation") as mock_store,
        patch("app.services.placement.host_store") as mock_host_store,
        patch("app.services.reservation.execute_migration") as mock_migrate,
    ):
        # Both hosts have low VRAM initially — triggers migration path
        # host_gpu2 has VRAM for displaced instance but not enough RAM for the request
        mock_db.get_all_hosts = AsyncMock(return_value=[host_gpu, host_gpu2])
        mock_fetch.side_effect = [
            _snap(host_gpu, vram_avail=5.0, ram_avail=10.0),
            _snap(host_gpu2, vram_avail=30.0, ram_avail=10.0),
            # After migration re-fetch: now host_gpu has capacity
            _snap(host_gpu, vram_avail=90.0, ram_avail=200.0),
        ]

        # host_gpu has a displaceable ephemeral instance
        mock_host_store.get_host_instances = AsyncMock(
            return_value=[
                {
                    "instance_id": "inst-ephem",
                    "config": {"alias": "old-model", "priority": "ephemeral"},
                    "vram_gb": 6,
                }
            ]
        )

        from app.models.migration import MigrationResult

        mock_migrate.return_value = MigrationResult(
            migration_id="mig-1",
            status="completed",
            source_host_id="h-gpu",
            source_host_name="GPU Node",
            target_host_id="h-gpu2",
            target_host_name="GPU Node 2",
            source_instance_id="inst-ephem",
            target_instance_id="new-inst",
            alias="old-model",
            model_source="repo://old:v1",
            priority="ephemeral",
        )

        mock_reserve.return_value = {"reservation_id": "host-res-1"}
        mock_store.return_value = None

        result = await reserve_resources(reservation_request)

        assert result.host_id == "h-gpu"
        assert result.migrated is True
        assert len(result.migrations) == 1
        assert result.migrations[0]["status"] == "completed"


# ── release_reservation ─────────────────────────────────────────


@pytest.mark.anyio
async def test_release_reservation_success(host_gpu):
    """Release finds the reservation and proxies to the host."""
    with (
        patch("app.services.reservation._get_reservation") as mock_get,
        patch("app.services.reservation.host_db") as mock_db,
        patch("app.services.reservation._call_host_release") as mock_release,
        patch("app.services.reservation._remove_reservation") as mock_remove,
    ):
        mock_get.return_value = {
            "reservation_id": "res-1",
            "host_id": "h-gpu",
            "host_reservation_id": "host-res-1",
        }
        mock_db.get_host = AsyncMock(return_value=host_gpu)
        mock_release.return_value = {"status": "released"}
        mock_remove.return_value = None

        result = await release_reservation("res-1")

        assert result.released is True
        assert result.host_id == "h-gpu"


@pytest.mark.anyio
async def test_release_not_found():
    """Release with unknown reservation ID returns 404."""
    with patch(
        "app.services.reservation._get_reservation",
        return_value=None,
    ):
        with pytest.raises(HTTPException) as exc:
            await release_reservation("nonexistent")
        assert exc.value.status_code == 404


@pytest.mark.anyio
async def test_release_host_not_found():
    """Release when host no longer exists returns 404."""
    with (
        patch("app.services.reservation._get_reservation") as mock_get,
        patch("app.services.reservation.host_db") as mock_db,
    ):
        mock_get.return_value = {
            "reservation_id": "res-1",
            "host_id": "h-missing",
            "host_reservation_id": "host-res-1",
        }
        mock_db.get_host = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc:
            await release_reservation("res-1")
        assert exc.value.status_code == 404


# ── With GPU type filtering ─────────────────────────────────────


@pytest.mark.anyio
async def test_reserve_gpu_type_filter(reservation_request):
    """Only hosts with matching GPU type are candidates."""
    host_cuda = Host(
        id="h-cuda",
        name="CUDA",
        url="http://cuda:8000",
        api_key="k",
        status=HostStatus.ONLINE,
        gpu_type="nvidia_cuda",
        roles=["training"],
    )
    host_mps = Host(
        id="h-mps",
        name="MPS",
        url="http://mps:8000",
        api_key="k",
        status=HostStatus.ONLINE,
        gpu_type="apple_mps",
        roles=["training"],
    )

    req = ReservationRequest(
        requester="test",
        job_id="j1",
        vram_gb=10.0,
        host_roles=["training"],
        gpu_type="nvidia_cuda",
    )

    with (
        patch("app.services.reservation.host_db") as mock_db,
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
        patch("app.services.reservation._call_host_reserve") as mock_reserve,
        patch("app.services.reservation._store_reservation") as mock_store,
        patch("app.services.placement.host_store") as mock_host_store,
    ):
        mock_db.get_all_hosts = AsyncMock(return_value=[host_cuda, host_mps])
        mock_fetch.side_effect = [
            _snap(host_cuda),
            _snap(host_mps),
        ]
        mock_reserve.return_value = {"reservation_id": "hr-1"}
        mock_store.return_value = None
        mock_host_store.get_host_instances = AsyncMock(return_value=[])

        result = await reserve_resources(req)

        assert result.host_id == "h-cuda"

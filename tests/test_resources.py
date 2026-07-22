"""Tests for GET /api/resources (S-035)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.models import (
    Host,
    HostStatus,
    HostResourceSnapshot,
    AggregatedResourceResponse,
)
from app.routes.management.resources import _fetch_host_resource_snapshot


@pytest.fixture
def mock_host_online():
    return Host(
        id="host-1",
        name="GPU Node 1",
        url="http://host-1:8000",
        api_key="key-1",
        status=HostStatus.ONLINE,
        gpu_type="A100-80GB",
        roles=["inference", "training"],
    )


@pytest.fixture
def mock_host_offline():
    return Host(
        id="host-2",
        name="GPU Node 2",
        url="http://host-2:8000",
        api_key="key-2",
        status=HostStatus.OFFLINE,
        gpu_type="A10-24GB",
        roles=["inference"],
    )


@pytest.fixture
def live_resource_payload():
    """Mimics a solar-host GET /resources response."""
    return {
        "memory_type": "VRAM",
        "vram": {
            "total_gb": 80.0,
            "system_used_gb": 10.0,
            "reserved_headroom_gb": 20.0,
            "reported_used_gb": 30.0,
            "available_gb": 50.0,
        },
        "ram": {
            "total_gb": 256.0,
            "system_used_gb": 32.0,
            "reserved_headroom_gb": 64.0,
            "reported_used_gb": 96.0,
            "available_gb": 160.0,
        },
        "disk": {
            "total_gb": 1000.0,
            "system_used_gb": 200.0,
            "reserved_headroom_gb": 50.0,
            "reported_used_gb": 250.0,
            "available_gb": 750.0,
        },
        "reservations": [
            {
                "id": "res-1",
                "job_id": "job-1",
                "workload_type": "training",
                "status": "pending",
                "vram_gb": 20.0,
                "ram_gb": 64.0,
                "disk_gb": 50.0,
            }
        ],
    }


# ── _fetch_host_resource_snapshot tests ────────────────────────────


@pytest.mark.anyio
async def test_fetch_host_resource_snapshot_success(
    mock_host_online, live_resource_payload
):
    """Live host returns valid resource snapshot."""
    with (
        patch("app.routes.management.resources.host_store") as mock_store,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_store.get_host_instances = AsyncMock(
            return_value=[
                {"id": "inst-1", "status": "running"},
                {"id": "inst-2", "status": "stopped"},
            ]
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=live_resource_payload)
        mock_get.return_value.__aenter__.return_value = mock_resp

        snap = await _fetch_host_resource_snapshot(mock_host_online)

        assert snap.reachable is True
        assert snap.host_id == "host-1"
        assert snap.host_name == "GPU Node 1"
        assert snap.gpu_type == "A100-80GB"
        assert snap.roles == ["inference", "training"]
        assert snap.vram_total_gb == 80.0
        assert snap.vram_available_gb == 50.0
        assert snap.ram_total_gb == 256.0
        assert snap.ram_available_gb == 160.0
        assert snap.disk_total_gb == 1000.0
        assert snap.disk_available_gb == 750.0
        assert snap.instance_count == 2
        assert snap.running_instance_count == 1
        assert snap.reservation_count == 1
        assert snap.reservation_vram_total_gb == 20.0
        assert snap.reservation_ram_total_gb == 64.0
        assert snap.reservation_disk_total_gb == 50.0


@pytest.mark.anyio
async def test_fetch_host_resource_snapshot_unreachable(mock_host_online):
    """Connection error marks host as unreachable with error message."""
    with (
        patch("app.routes.management.resources.host_store") as mock_store,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_store.get_host_instances = AsyncMock(return_value=[])
        import aiohttp

        mock_get.side_effect = aiohttp.ClientConnectionError("Refused")

        snap = await _fetch_host_resource_snapshot(mock_host_online)

        assert snap.reachable is False
        assert snap.error is not None
        assert "unreachable" in snap.error
        assert snap.host_name == "GPU Node 1"  # DB data still present
        assert snap.gpu_type == "A100-80GB"


@pytest.mark.anyio
async def test_fetch_host_resource_snapshot_timeout(mock_host_online):
    """Timeout marks host as unreachable."""
    with (
        patch("app.routes.management.resources.host_store") as mock_store,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_store.get_host_instances = AsyncMock(return_value=[])
        mock_get.side_effect = asyncio.TimeoutError()

        snap = await _fetch_host_resource_snapshot(mock_host_online)

        assert snap.reachable is False
        assert "timed out" in snap.error.lower()


@pytest.mark.anyio
async def test_fetch_host_resource_snapshot_non_200(mock_host_online):
    """Non-200 response marks host as unreachable."""
    with (
        patch("app.routes.management.resources.host_store") as mock_store,
        patch("aiohttp.ClientSession.get") as mock_get,
    ):
        mock_store.get_host_instances = AsyncMock(return_value=[])
        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_get.return_value.__aenter__.return_value = mock_resp

        snap = await _fetch_host_resource_snapshot(mock_host_online)

        assert snap.reachable is False
        assert "HTTP 503" in (snap.error or "")


# ── get_resources integration tests ─────────────────────────────────


# Helper: build a simple reachable snapshot per host
def _make_snap(h, *, reachable=True, **kw):
    return HostResourceSnapshot(
        host_id=h.id,
        host_name=h.name,
        url=h.url,
        status=h.status,
        roles=h.roles or [],
        gpu_type=h.gpu_type,
        reachable=reachable,
        **kw,
    )


@pytest.mark.anyio
async def test_get_resources_all_hosts(mock_host_online, mock_host_offline):
    """Aggregated endpoint returns all hosts with correct counts."""
    from app.routes.management.resources import get_resources

    hosts = [mock_host_online, mock_host_offline]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
    ):
        mock_fetch.side_effect = [
            _make_snap(
                mock_host_online,
                vram_total_gb=80.0,
                vram_available_gb=50.0,
                ram_total_gb=256.0,
                ram_available_gb=160.0,
                instance_count=3,
                running_instance_count=2,
                reservation_count=1,
            ),
            _make_snap(mock_host_offline, reachable=False, error="Host unreachable"),
        ]

        resp = await get_resources()

        assert isinstance(resp, AggregatedResourceResponse)
        assert resp.total_hosts == 2
        assert resp.reachable_hosts == 1
        assert resp.unreachable_hosts == 1
        assert len(resp.hosts) == 2
        assert resp.hosts[0].reachable is True
        assert resp.hosts[1].reachable is False


@pytest.mark.anyio
async def test_get_resources_filter_by_gpu_type(mock_host_online, mock_host_offline):
    """gpu_type filter returns only matching hosts."""
    from app.routes.management.resources import get_resources

    hosts = [mock_host_online, mock_host_offline]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
    ):
        mock_fetch.side_effect = [
            _make_snap(mock_host_online),
            _make_snap(mock_host_offline),
        ]

        resp = await get_resources(gpu_type="A100-80GB")
        assert len(resp.hosts) == 1
        assert resp.hosts[0].host_id == "host-1"


@pytest.mark.anyio
async def test_get_resources_filter_by_role(mock_host_online):
    """role filter passed to host_db.get_all_hosts."""
    from app.routes.management.resources import get_resources

    hosts = [mock_host_online]

    with (
        patch(
            "app.database.hosts.host_db.get_all_hosts", return_value=hosts
        ) as mock_db,
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
    ):
        mock_fetch.side_effect = [_make_snap(mock_host_online)]

        resp = await get_resources(role="training")
        mock_db.assert_called_once_with(role="training")
        assert len(resp.hosts) == 1


@pytest.mark.anyio
async def test_get_resources_filter_by_host_id(mock_host_online):
    """host_id filter returns single host or 404."""
    from app.routes.management.resources import get_resources
    from fastapi import HTTPException

    with (
        patch("app.database.hosts.host_db.get_host", return_value=mock_host_online),
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
    ):
        mock_fetch.return_value = _make_snap(mock_host_online)

        resp = await get_resources(host_id="host-1")
        assert len(resp.hosts) == 1
        assert resp.hosts[0].host_id == "host-1"

    # 404 case
    with patch("app.database.hosts.host_db.get_host", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            await get_resources(host_id="nonexistent")
        assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_get_resources_min_available_vram(mock_host_online):
    """min_available_vram_gb filter excludes hosts below threshold."""
    from app.routes.management.resources import get_resources

    hosts = [mock_host_online]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch(
            "app.routes.management.resources._fetch_host_resource_snapshot"
        ) as mock_fetch,
    ):
        # Host has 50 GB available VRAM — use return_value for repeated calls
        mock_fetch.return_value = _make_snap(
            mock_host_online,
            vram_total_gb=80.0,
            vram_available_gb=50.0,
            ram_total_gb=256.0,
            ram_available_gb=160.0,
        )

        # Filter: min 40 GB — should include
        resp = await get_resources(min_available_vram_gb=40)
        assert len(resp.hosts) == 1

        # Filter: min 60 GB — should exclude
        resp = await get_resources(min_available_vram_gb=60)
        assert len(resp.hosts) == 0


@pytest.mark.anyio
async def test_get_resources_no_hosts():
    """Empty host list returns empty response."""
    from app.routes.management.resources import get_resources

    with patch("app.database.hosts.host_db.get_all_hosts", return_value=[]):
        resp = await get_resources()
        assert resp.total_hosts == 0
        assert resp.hosts == []

"""Tests for shared placement policy (S-038 / S-041)."""

import pytest
from unittest.mock import AsyncMock, patch

from app.models import Host, HostStatus, HostResourceSnapshot
from app.services.placement import (
    find_candidates,
    can_displace,
    find_displaceable_instances,
    fits_resources,
)

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def host_a100() -> Host:
    return Host(
        id="h-a100",
        name="A100 Node",
        url="http://a100:8000",
        api_key="k",
        status=HostStatus.ONLINE,
        gpu_type="nvidia_cuda",
        roles=["inference", "training"],
    )


@pytest.fixture
def host_mps() -> Host:
    return Host(
        id="h-mps",
        name="MPS Node",
        url="http://mps:8000",
        api_key="k",
        status=HostStatus.ONLINE,
        gpu_type="apple_mps",
        roles=["inference"],
    )


@pytest.fixture
def host_offline() -> Host:
    return Host(
        id="h-off",
        name="Offline Node",
        url="http://off:8000",
        api_key="k",
        status=HostStatus.OFFLINE,
        gpu_type="nvidia_cuda",
        roles=["inference"],
    )


def _make_snap(
    host: Host,
    *,
    vram_available=80.0,
    ram_available=200.0,
    instances=0,
) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        host_id=host.id,
        host_name=host.name,
        url=host.url,
        status=host.status,
        roles=host.roles or [],
        gpu_type=host.gpu_type,
        reachable=True,
        vram_available_gb=vram_available,
        ram_available_gb=ram_available,
        running_instance_count=instances,
    )


# ── fits_resources ──────────────────────────────────────────────


def test_fits_resources_sufficient():
    snap = HostResourceSnapshot(
        host_id="h1",
        host_name="test",
        url="http://h:8000",
        status=HostStatus.ONLINE,
        reachable=True,
        vram_available_gb=80.0,
        ram_available_gb=200.0,
        disk_available_gb=500.0,
    )
    assert fits_resources(snap, 20.0, 64.0, 50.0) is True


def test_fits_resources_insufficient_vram():
    snap = HostResourceSnapshot(
        host_id="h1",
        host_name="test",
        url="http://h:8000",
        status=HostStatus.ONLINE,
        reachable=True,
        vram_available_gb=4.0,
    )
    assert fits_resources(snap, 20.0, None, None) is False


def test_fits_resources_unreachable():
    snap = HostResourceSnapshot(
        host_id="h1",
        host_name="test",
        url="http://h:8000",
        status=HostStatus.OFFLINE,
        reachable=False,
        vram_available_gb=80.0,
    )
    assert fits_resources(snap, 10.0, None, None) is False


def test_fits_resources_none_optional():
    """None values for optional resources should skip those checks."""
    snap = HostResourceSnapshot(
        host_id="h1",
        host_name="test",
        url="http://h:8000",
        status=HostStatus.ONLINE,
        reachable=True,
        vram_available_gb=80.0,
        ram_available_gb=None,  # unknown
        disk_available_gb=None,  # unknown
    )
    assert fits_resources(snap, 20.0, None, None) is True


# ── find_candidates ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_find_candidates_basic(host_a100, host_mps):
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: _make_snap(host_a100, vram_available=80.0),
        host_mps.id: _make_snap(host_mps, vram_available=40.0),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
        )

        assert len(candidates) == 2
        # A100 should rank first (more VRAM)
        assert candidates[0][0].id == "h-a100"


@pytest.mark.anyio
async def test_find_candidates_role_filter(host_a100, host_mps):
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: _make_snap(host_a100),
        host_mps.id: _make_snap(host_mps),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["training"],
            vram_gb=6.0,
        )

        # Only host_a100 has "training" role
        assert len(candidates) == 1
        assert candidates[0][0].id == "h-a100"


@pytest.mark.anyio
async def test_find_candidates_gpu_filter(host_a100, host_mps):
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: _make_snap(host_a100),
        host_mps.id: _make_snap(host_mps),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            gpu_type="apple_mps",
            vram_gb=6.0,
        )

        assert len(candidates) == 1
        assert candidates[0][0].id == "h-mps"


@pytest.mark.anyio
async def test_find_candidates_insufficient_vram(host_a100, host_mps):
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: _make_snap(host_a100, vram_available=4.0),
        host_mps.id: _make_snap(host_mps, vram_available=2.0),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=20.0,
        )

        assert len(candidates) == 0


@pytest.mark.anyio
async def test_find_candidates_allow_deny(host_a100, host_mps):
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: _make_snap(host_a100),
        host_mps.id: _make_snap(host_mps),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        # host_allow restricts to a100
        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
            host_allow=["h-a100"],
        )
        assert len(candidates) == 1
        assert candidates[0][0].id == "h-a100"

        # host_deny excludes a100
        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
            host_deny=["h-a100"],
        )
        assert len(candidates) == 1
        assert candidates[0][0].id == "h-mps"


@pytest.mark.anyio
async def test_find_candidates_alias_conflict(host_a100, host_mps):
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: _make_snap(host_a100),
        host_mps.id: _make_snap(host_mps),
    }

    with patch("app.services.placement.host_store") as mock_store:
        # A100 already runs the alias
        mock_store.get_host_instances = AsyncMock(
            side_effect=lambda hid: (
                [{"config": {"alias": "test-model:v1"}}] if hid == "h-a100" else []
            )
        )

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
            exclude_alias="test-model:v1",
        )

        assert len(candidates) == 1
        assert candidates[0][0].id == "h-mps"


@pytest.mark.anyio
async def test_find_candidates_offline_excluded(host_a100, host_offline):
    """Offline hosts should not be candidates even with resources."""
    hosts = [host_a100, host_offline]
    snapshots = {
        host_a100.id: _make_snap(host_a100),
        host_offline.id: HostResourceSnapshot(
            host_id=host_offline.id,
            host_name=host_offline.name,
            url=host_offline.url,
            status=host_offline.status,
            reachable=False,
            vram_available_gb=999.0,
        ),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
        )

        assert len(candidates) == 1
        assert candidates[0][0].id == "h-a100"


@pytest.mark.anyio
async def test_find_candidates_ranking(host_a100, host_mps):
    """Rank by VRAM (desc), then instances (asc), then host_id."""
    hosts = [host_mps, host_a100]  # Deliberately out of order
    snapshots = {
        host_a100.id: _make_snap(host_a100, vram_available=80.0, instances=3),
        host_mps.id: _make_snap(host_mps, vram_available=80.0, instances=1),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
        )

        # Same VRAM, fewer instances wins
        assert candidates[0][0].id == "h-mps"
        assert candidates[1][0].id == "h-a100"


# ── can_displace ────────────────────────────────────────────────


def test_can_displace_production_over_staging():
    assert can_displace("production", "staging") is True


def test_can_displace_production_over_ephemeral():
    assert can_displace("production", "ephemeral") is True


def test_can_displace_staging_over_ephemeral():
    assert can_displace("staging", "ephemeral") is True


def test_can_displace_equal_not_allowed():
    assert can_displace("staging", "staging") is False
    assert can_displace("production", "production") is False


def test_can_displace_lower_not_allowed():
    assert can_displace("staging", "production") is False
    assert can_displace("ephemeral", "staging") is False


def test_can_displace_unknown_priority():
    assert can_displace("unknown", "staging") is False
    assert can_displace("staging", "unknown") is False


# ── find_displaceable_instances ─────────────────────────────────


@pytest.mark.anyio
async def test_find_displaceable_basic():
    instances = [
        {"config": {"alias": "m1", "priority": "ephemeral"}, "vram_gb": 4},
        {"config": {"alias": "m2", "priority": "staging"}, "vram_gb": 6},
        {"config": {"alias": "m3", "priority": "production"}, "vram_gb": 8},
    ]

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=instances)

        result = await find_displaceable_instances("h1", "production")

        # Both ephemeral and staging are displaceable by production
        assert len(result) == 2
        # ephemeral should sort first (lowest priority)
        assert result[0].get("config", {}).get("priority") == "ephemeral"


@pytest.mark.anyio
async def test_find_displaceable_staging_only():
    """Staging can only displace ephemeral."""
    instances = [
        {"config": {"alias": "m1", "priority": "ephemeral"}, "vram_gb": 4},
        {"config": {"alias": "m2", "priority": "staging"}, "vram_gb": 6},
    ]

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=instances)

        result = await find_displaceable_instances("h1", "staging")

        assert len(result) == 1
        assert result[0].get("config", {}).get("priority") == "ephemeral"


@pytest.mark.anyio
async def test_find_displaceable_preserve_alias():
    instances = [
        {
            "config": {"alias": "iris:v1", "priority": "ephemeral"},
            "vram_gb": 4,
        },
        # Only one replica of "iris:v1" — should be preserved
    ]

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=instances)

        result = await find_displaceable_instances(
            "h1", "production", preserve_alias="iris:v1"
        )

        # Must preserve at least one — and there's only one
        assert len(result) == 0


@pytest.mark.anyio
async def test_find_displaceable_preserve_alias_extra_replica():
    """With two replicas, one can be displaced while one is preserved."""
    instances = [
        {
            "config": {"alias": "iris:v1", "priority": "ephemeral"},
            "vram_gb": 4,
        },
        {
            "config": {"alias": "iris:v1", "priority": "ephemeral"},
            "vram_gb": 4,
        },
    ]

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=instances)

        result = await find_displaceable_instances(
            "h1", "production", preserve_alias="iris:v1"
        )

        # Both are displaceable (more than one replica exists)
        assert len(result) == 2


# ── §8.4 free-disk tiebreak ─────────────────────────────────────


@pytest.mark.anyio
async def test_find_candidates_disk_tiebreak(host_a100, host_mps):
    """Equal VRAM → more free disk ranks first (§8.4)."""
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: HostResourceSnapshot(
            host_id=host_a100.id,
            host_name=host_a100.name,
            url=host_a100.url,
            status=host_a100.status,
            reachable=True,
            vram_available_gb=80.0,
            disk_available_gb=5.0,
        ),
        host_mps.id: HostResourceSnapshot(
            host_id=host_mps.id,
            host_name=host_mps.name,
            url=host_mps.url,
            status=host_mps.status,
            reachable=True,
            vram_available_gb=80.0,
            disk_available_gb=50.0,
        ),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
        )

        # Same VRAM → more free disk wins
        assert candidates[0][0].id == "h-mps"
        assert candidates[1][0].id == "h-a100"


@pytest.mark.anyio
async def test_find_candidates_vram_beats_disk(host_a100, host_mps):
    """More free VRAM still outranks more free disk (§8.4 ordering)."""
    hosts = [host_a100, host_mps]
    snapshots = {
        host_a100.id: HostResourceSnapshot(
            host_id=host_a100.id,
            host_name=host_a100.name,
            url=host_a100.url,
            status=host_a100.status,
            reachable=True,
            vram_available_gb=80.0,
            disk_available_gb=1.0,
        ),
        host_mps.id: HostResourceSnapshot(
            host_id=host_mps.id,
            host_name=host_mps.name,
            url=host_mps.url,
            status=host_mps.status,
            reachable=True,
            vram_available_gb=40.0,
            disk_available_gb=500.0,
        ),
    }

    with patch("app.services.placement.host_store") as mock_store:
        mock_store.get_host_instances = AsyncMock(return_value=[])

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=["inference"],
            vram_gb=6.0,
        )

        assert candidates[0][0].id == "h-a100"
        assert candidates[1][0].id == "h-mps"

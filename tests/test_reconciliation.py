"""Tests for reconciliation engine (S-041)."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from app.models.intent import (
    IntentPhase,
    IntentResponse,
    IntentStatus,
    PlacementConstraints,
    ReconcileState,
    ResourceRequirements,
)
from app.services.reconciliation import ActionType, Reconciler

# ── Simple host stub ────────────────────────────────────────────


@dataclass
class _HostStub:
    id: str
    name: str = ""
    status: str = "online"
    url: str = "http://localhost:8080"
    api_key: str = "test-key"
    roles: list | None = None
    gpu_type: str | None = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = ["inference"]


# ── Helpers ────────────────────────────────────────────────────


def _make_intent(**overrides) -> IntentResponse:
    """Build a minimal IntentResponse for testing."""
    defaults = {
        "id": "intent-001",
        "alias": "test-model",
        "model_source": "repo://test:v1",
        "replicas": 2,
        "priority": "production",
        "strategy": "rolling",
        "backend": {"backend_type": "huggingface_classification", "max_length": 512},
        "placement": PlacementConstraints(),
        "resources": ResourceRequirements(),
        "metadata": {},
        "status": IntentStatus(
            phase=IntentPhase.RECONCILING,
            reconcile=ReconcileState.IN_PROGRESS,
            desired_replicas=2,
        ),
    }
    defaults.update(overrides)
    return IntentResponse(**defaults)


def _make_managed_instance(
    instance_id: str,
    host_id: str = "host-1",
    host_name: str = "host1",
    alias: str = "test-model",
    model_source: str = "repo://test:v1",
    status: str = "running",
) -> dict:
    """Build a managed instance dict (as stored in Redis)."""
    return {
        "instance_id": instance_id,
        "id": instance_id,
        "status": status,
        "config": {
            "alias": alias,
            "model_source": model_source,
            "managed_by": "intent",
            "intent_id": "intent-001",
            "backend_type": "huggingface_classification",
        },
        "_host_id": host_id,
        "_host_name": host_name,
    }


def _make_observed(
    managed: list | None = None,
    alias_instances: list | None = None,
    hosts: list | None = None,
    gateway_aliases: set | None = None,
) -> dict:
    """Build an observed state dict for testing."""
    if managed is None:
        managed = []
    if alias_instances is None:
        alias_instances = list(managed)
    return {
        "managed_instances": managed,
        "alias_instances": alias_instances,
        "hosts": hosts or [],
        "snapshots": {},
        "gateway_aliases": gateway_aliases or set(),
    }


# ── Diff tests ─────────────────────────────────────────────────


class TestDiff:
    """Test the _diff method's action planning."""

    def test_noop_when_desired_matches_observed(self):
        """When replicas match, no actions needed."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=2)
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", host_id="h1"),
                _make_managed_instance("inst-2", host_id="h2"),
            ]
        )
        actions = reconciler._diff(intent, observed)
        assert actions == []

    def test_create_on_shortfall(self):
        """When observed < desired, create actions are generated."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=3)
        observed = _make_observed(
            managed=[_make_managed_instance("inst-1", host_id="h1")],
            hosts=[_HostStub(id="h2"), _HostStub(id="h3")],
        )
        actions = reconciler._diff(intent, observed)
        creates = [a for a in actions if a.type == ActionType.CREATE]
        assert len(creates) == 2  # shortfall of 2

    def test_stop_surplus(self):
        """When observed > desired, stop actions are generated."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", host_id="h1"),
                _make_managed_instance("inst-2", host_id="h2"),
            ]
        )
        actions = reconciler._diff(intent, observed)
        stops = [a for a in actions if a.type == ActionType.STOP]
        assert len(stops) == 1

    def test_replace_on_model_source_drift(self):
        """When instance model_source differs, replace action is generated."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1, model_source="repo://test:v2")
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", model_source="repo://test:v1"),
            ]
        )
        actions = reconciler._diff(intent, observed)
        replaces = [a for a in actions if a.type == ActionType.REPLACE]
        assert len(replaces) == 1

    def test_recreate_on_failed_instance(self):
        """Failed instances trigger recreate actions."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", status="failed"),
            ]
        )
        actions = reconciler._diff(intent, observed)
        recreates = [a for a in actions if a.type == ActionType.RECREATE]
        assert len(recreates) == 1

    def test_stop_all_on_delete(self):
        """Deleting intents get stop actions for all managed instances."""
        reconciler = Reconciler()
        intent = _make_intent(
            replicas=2,
            status=IntentStatus(phase=IntentPhase.DELETING),
        )
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", host_id="h1"),
                _make_managed_instance("inst-2", host_id="h2"),
            ]
        )
        actions = reconciler._diff(intent, observed)
        stops = [a for a in actions if a.type == ActionType.STOP]
        assert len(stops) == 2

    def test_stop_all_on_zero_replicas(self):
        """replicas=0 means stop all managed instances."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=0)
        observed = _make_observed(managed=[_make_managed_instance("inst-1")])
        actions = reconciler._diff(intent, observed)
        stops = [a for a in actions if a.type == ActionType.STOP]
        assert len(stops) == 1

    def test_actions_sorted_by_priority(self):
        """Actions are sorted: stops first (p0), recreate (p15), create (p50)."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", host_id="h1", status="failed"),
                _make_managed_instance("inst-2", host_id="h2"),
            ],
            hosts=[_HostStub(id="h3")],
        )
        # Surplus of 1 → stop, failed → recreate, shortfall → create
        actions = reconciler._diff(intent, observed)
        priorities = [a.priority for a in actions]
        assert priorities == sorted(
            priorities
        ), f"Expected sorted priorities, got {priorities}"

    def test_no_create_when_hosts_occupied(self):
        """Creates skip hosts already serving the alias."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=2)
        observed = _make_observed(
            managed=[_make_managed_instance("inst-1", host_id="h1")],
            alias_instances=[
                _make_managed_instance("inst-1", host_id="h1"),
                # Manual instance on h2 also serving the same alias
                {
                    "instance_id": "inst-manual",
                    "config": {"alias": "test-model"},
                    "_host_id": "h2",
                },
            ],
            hosts=[_HostStub(id="h2"), _HostStub(id="h3")],
        )
        actions = reconciler._diff(intent, observed)
        creates = [a for a in actions if a.type == ActionType.CREATE]
        # Only h3 is eligible (h2 occupied by manual instance)
        assert len(creates) == 1
        assert creates[0].host_id == "h3"


# ── Build instance config test ──────────────────────────────────


class TestBuildInstanceConfig:
    """Test _build_instance_config method."""

    def test_maps_fields_correctly(self):
        """Config contains alias, model_source, priority, managed_by, intent_id.

        Per deployment-intent.md §6, backend_type maps to the instance config.
        """
        reconciler = Reconciler()
        intent = _make_intent(
            alias="iris-osl:110m",
            model_source="repo://iris-osl:v3",
            priority="production",
            backend={
                "backend_type": "huggingface_classification",
                "max_length": 512,
                "labels": ["osl"],
            },
        )
        host = _HostStub(id="h1")
        config = reconciler._build_instance_config(intent, host)

        assert config["config"]["alias"] == "iris-osl:110m"
        assert config["config"]["model_source"] == "repo://iris-osl:v3"
        assert config["config"]["priority"] == "production"
        assert config["config"]["managed_by"] == "intent"
        assert config["config"]["intent_id"] == "intent-001"
        assert config["config"]["max_length"] == 512
        assert config["config"]["backend_type"] == "huggingface_classification"

    def test_copies_backend_runtime_params(self):
        """Backend params are copied to config, backend_type included."""
        reconciler = Reconciler()
        intent = _make_intent(
            backend={"backend_type": "llamacpp", "dtype": "float16"},
        )
        host = _HostStub(id="h1")
        config = reconciler._build_instance_config(intent, host)
        assert config["config"]["backend_type"] == "llamacpp"
        assert config["config"]["dtype"] == "float16"


# ── Integration tests ──────────────────────────────────────────


class TestReconcileOne:
    """Integration tests for the full observe→diff→act→status pipeline."""

    @pytest.mark.anyio
    async def test_noop_when_already_healthy(self):
        """When desired state matches, status is updated but no actions taken."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=2)
        managed = [
            _make_managed_instance("inst-1", host_id="h1"),
            _make_managed_instance("inst-2", host_id="h2"),
        ]

        with (
            patch.object(
                reconciler,
                "_observe",
                new=AsyncMock(
                    return_value=_make_observed(
                        managed=managed, gateway_aliases={"test-model"}
                    )
                ),
            ),
            patch.object(reconciler, "_update_status", new=AsyncMock()) as mock_status,
        ):
            await reconciler._reconcile_one(intent)
            mock_status.assert_called_once()

    @pytest.mark.anyio
    async def test_create_action_executed(self):
        """Shortfall triggers create action on eligible host."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        host = _HostStub(id="h1")

        observed = _make_observed(managed=[], hosts=[host])

        with (
            patch.object(reconciler, "_observe", new=AsyncMock(return_value=observed)),
            patch.object(
                reconciler,
                "_act",
                new=AsyncMock(return_value={"instance_id": "new-inst"}),
            ) as mock_act,
            patch.object(reconciler, "_update_status", new=AsyncMock()) as mock_status,
        ):
            await reconciler._reconcile_one(intent)
            mock_act.assert_called_once()
            mock_status.assert_called_once()

    @pytest.mark.anyio
    async def test_error_reported_in_status(self):
        """When action fails, last_error is populated."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        host = _HostStub(id="h1")

        observed = _make_observed(managed=[], hosts=[host])

        with (
            patch.object(reconciler, "_observe", new=AsyncMock(return_value=observed)),
            patch.object(
                reconciler,
                "_act",
                new=AsyncMock(side_effect=RuntimeError("host unreachable")),
            ),
            patch.object(reconciler, "_update_status", new=AsyncMock()) as mock_status,
        ):
            await reconciler._reconcile_one(intent)

            # Verify last_error was passed to _update_status
            call_kwargs = mock_status.call_args
            last_error = call_kwargs[1].get("last_error")
            assert last_error is not None
            assert last_error["code"] == "RuntimeError"
            assert "host unreachable" in last_error["message"]

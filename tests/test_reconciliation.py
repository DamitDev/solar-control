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
from app.services.reconciliation import (
    Action,
    ActionType,
    Reconciler,
    _detect_backend_drift,
    _intent_orphan,
    _intent_phase,
)

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


@dataclass
class _SnapshotStub:
    """Minimal snapshot stub for placement policy."""

    host_id: str
    reachable: bool = True
    vram_available_gb: float = 10.0
    ram_available_gb: float | None = None
    disk_available_gb: float | None = None
    running_instance_count: int = 0


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
    **extra_config,
) -> dict:
    """Build a managed instance dict (as stored in Redis).

    Default config matches the default intent's backend so drift
    detection doesn't fire on tests that don't care about drift.
    """
    config = {
        "alias": alias,
        "model_source": model_source,
        "managed_by": "intent",
        "intent_id": "intent-001",
        "backend_type": "huggingface_classification",
        "max_length": 512,
    }
    config.update(extra_config)
    return {
        "instance_id": instance_id,
        "id": instance_id,
        "status": status,
        "config": config,
        "_host_id": host_id,
        "_host_name": host_name,
    }


def _make_observed(
    managed: list | None = None,
    alias_instances: list | None = None,
    hosts: list | None = None,
    snapshots: dict | None = None,
    gateway_aliases: set | None = None,
    candidates: list | None = None,
    displaceable_map: dict | None = None,
    manual_conflicts: list | None = None,
) -> dict:
    """Build an observed state dict for testing.

    For CREATE tests, provide *candidates* as a list of (host, snapshot) tuples.
    For tests that don't test CREATE, leave candidates empty.
    """
    if managed is None:
        managed = []
    if alias_instances is None:
        alias_instances = list(managed)
    if candidates is None:
        candidates = []
    return {
        "managed_instances": managed,
        "alias_instances": alias_instances,
        "hosts": hosts or [],
        "snapshots": snapshots or {},
        "gateway_aliases": gateway_aliases or set(),
        "candidates": candidates,
        "displaceable_map": displaceable_map or {},
        "manual_conflicts": manual_conflicts or [],
    }


# ── Helper function tests ──────────────────────────────────────


class TestHelpers:
    """Test standalone helper functions."""

    def test_intent_phase_extracts_correctly(self):
        intent = _make_intent(status=IntentStatus(phase=IntentPhase.READY))
        assert _intent_phase(intent) == "ready"

    def test_intent_phase_fallback(self):
        """Fallback for objects without status.phase."""

        class StubIntent:
            phase = "pending"

        assert _intent_phase(StubIntent()) == "pending"

    def test_intent_orphan_true(self):
        intent = _make_intent(metadata={"orphan": "true"})
        assert _intent_orphan(intent) is True

    def test_intent_orphan_false(self):
        intent = _make_intent(metadata={})
        assert _intent_orphan(intent) is False

    def test_detect_backend_drift_no_change(self):
        intent = _make_intent(backend={"backend_type": "hf", "max_length": 512})
        instance_config = {"backend_type": "hf", "max_length": 512}
        assert _detect_backend_drift(intent, instance_config) is False

    def test_detect_backend_drift_changed(self):
        intent = _make_intent(backend={"backend_type": "hf", "max_length": 1024})
        instance_config = {"backend_type": "hf", "max_length": 512}
        assert _detect_backend_drift(intent, instance_config) is True

    def test_detect_backend_drift_skips_identity_fields(self):
        """Identity/server fields (alias, model_source, etc.) are ignored."""
        intent = _make_intent(
            backend={"backend_type": "hf", "alias": "x", "model_source": "y"}
        )
        instance_config = {
            "backend_type": "hf",
            "alias": "different",
            "model_source": "z",
        }
        assert _detect_backend_drift(intent, instance_config) is False


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
        """When observed < desired, create actions are generated from candidates."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=3)
        observed = _make_observed(
            managed=[_make_managed_instance("inst-1", host_id="h1")],
            hosts=[_HostStub(id="h2"), _HostStub(id="h3")],
            candidates=[
                (_HostStub(id="h2", name="h2"), _SnapshotStub("h2")),
                (_HostStub(id="h3", name="h3"), _SnapshotStub("h3")),
            ],
        )
        actions = reconciler._diff(intent, observed)
        creates = [a for a in actions if a.type == ActionType.CREATE]
        assert len(creates) == 2  # shortfall of 2

    def test_no_create_without_candidates(self):
        """When no candidates in observed, no CREATE actions even if shortfall."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=3)
        observed = _make_observed(
            managed=[_make_managed_instance("inst-1", host_id="h1")],
            hosts=[_HostStub(id="h2"), _HostStub(id="h3")],
        )
        actions = reconciler._diff(intent, observed)
        creates = [a for a in actions if a.type == ActionType.CREATE]
        assert len(creates) == 0  # candidates empty → no creates

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

    def test_stop_surplus_least_loaded_first(self):
        """Tiebreak among same-age healthy replicas: least-loaded host first."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        snapshots = {
            "h1": _SnapshotStub("h1", running_instance_count=5),
            "h2": _SnapshotStub("h2", running_instance_count=1),
        }
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", host_id="h1"),
                _make_managed_instance("inst-2", host_id="h2"),
            ],
            snapshots=snapshots,
        )
        actions = reconciler._diff(intent, observed)
        stops = [a for a in actions if a.type == ActionType.STOP]
        assert len(stops) == 1
        assert stops[0].instance_id == "inst-2"  # least-loaded stopped first

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

    def test_replace_on_backend_drift(self):
        """When instance backend config differs, replace action is generated."""
        reconciler = Reconciler()
        intent = _make_intent(
            replicas=1,
            backend={"backend_type": "hf", "max_length": 1024},
        )
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", max_length=512),
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

    def test_recreate_on_stopped_instance(self):
        """Stopped managed instances are drift → RECREATE (spec §8.2).

        D-017-9 exempted 'stopped' to stop /stop spam, but the spam came
        from _act RECREATE being stop-only. With restart-or-recreate
        semantics, a managed stopped instance (e.g. a migration target)
        must be restarted automatically.
        """
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", status="stopped"),
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

    def test_disown_on_delete_orphan(self):
        """Deleting intents with orphan=true get DISOWN actions, not STOP."""
        reconciler = Reconciler()
        intent = _make_intent(
            replicas=2,
            metadata={"orphan": "true"},
            status=IntentStatus(phase=IntentPhase.DELETING),
        )
        observed = _make_observed(
            managed=[
                _make_managed_instance("inst-1", host_id="h1"),
                _make_managed_instance("inst-2", host_id="h2"),
            ]
        )
        actions = reconciler._diff(intent, observed)
        disowns = [a for a in actions if a.type == ActionType.DISOWN]
        stops = [a for a in actions if a.type == ActionType.STOP]
        assert len(disowns) == 2
        assert len(stops) == 0

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
            candidates=[(_HostStub(id="h3", name="h3"), _SnapshotStub("h3"))],
        )
        # Surplus of 1 → stop, failed → recreate, shortfall → create
        actions = reconciler._diff(intent, observed)
        priorities = [a.priority for a in actions]
        assert priorities == sorted(
            priorities
        ), f"Expected sorted priorities, got {priorities}"

    def test_migrate_from_displacement(self):
        """When candidates are fewer than shortfall, MIGRATE actions are generated."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=2)
        observed = _make_observed(
            managed=[],
            hosts=[_HostStub(id="h1")],
            candidates=[
                (_HostStub(id="h1", name="h1"), _SnapshotStub("h1")),
            ],
            displaceable_map={
                "h2": [
                    {
                        "instance_id": "inst-ephemeral",
                        "config": {"alias": "other"},
                        "_priority": "ephemeral",
                    }
                ],
            },
        )
        actions = reconciler._diff(intent, observed)
        migrates = [a for a in actions if a.type == ActionType.MIGRATE]
        creates = [a for a in actions if a.type == ActionType.CREATE]
        assert len(creates) == 1  # 1 candidate used
        assert len(migrates) == 1  # 1 displacement needed for remaining shortfall


# ── Build instance config test ──────────────────────────────────


class TestBuildInstanceConfig:
    """Test _build_instance_config method."""

    def test_maps_fields_correctly(self):
        """Top-level: managed_by, intent_id, priority. Config: alias, source, backend.

        Per deployment-intent.md §6, managed_by/intent_id/priority are top-level
        fields on the Instance model, not nested inside config.
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
        payload = reconciler._build_instance_config(intent, host)

        # Top-level fields
        assert payload["managed_by"] == "intent"
        assert payload["intent_id"] == "intent-001"
        assert payload["priority"] == "production"

        # Config fields
        assert payload["config"]["alias"] == "iris-osl:110m"
        assert payload["config"]["model_source"] == "repo://iris-osl:v3"
        assert payload["config"]["max_length"] == 512
        assert payload["config"]["backend_type"] == "huggingface_classification"

        # NOT inside config
        assert "managed_by" not in payload["config"]
        assert "intent_id" not in payload["config"]
        assert "priority" not in payload["config"]

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


# ── Backoff tests ──────────────────────────────────────────────


class TestBackoff:
    """Test exponential backoff logic."""

    def test_backoff_clear(self):
        reconciler = Reconciler()
        reconciler._backoff["test-id"] = {"failures": 3}
        reconciler._backoff_clear("test-id")
        assert "test-id" not in reconciler._backoff

    def test_backoff_active_after_failure(self):
        reconciler = Reconciler()
        reconciler._backoff_record_failure("test-id")
        assert reconciler._backoff_active("test-id") is True  # 10s backoff

    def test_backoff_not_active_for_unknown(self):
        reconciler = Reconciler()
        assert reconciler._backoff_active("unknown") is False

    def test_skip_when_backoff_active(self):
        """_reconcile_one returns early when backoff is active."""
        reconciler = Reconciler()
        intent = _make_intent()
        reconciler._backoff_record_failure(intent.id)

        with patch.object(reconciler, "_observe") as mock_observe:
            # Should not call _observe because backoff is active
            import asyncio

            asyncio.run(reconciler._reconcile_one(intent))
            mock_observe.assert_not_called()


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

        observed = _make_observed(
            managed=[],
            hosts=[host],
            candidates=[(host, _SnapshotStub("h1"))],
        )

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

        observed = _make_observed(
            managed=[],
            hosts=[host],
            candidates=[(host, _SnapshotStub("h1"))],
        )

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

    @pytest.mark.anyio
    async def test_backoff_recorded_on_failure(self):
        """After a failure, backoff is set."""
        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        host = _HostStub(id="h1")

        observed = _make_observed(
            managed=[],
            hosts=[host],
            candidates=[(host, _SnapshotStub("h1"))],
        )

        with (
            patch.object(reconciler, "_observe", new=AsyncMock(return_value=observed)),
            patch.object(
                reconciler,
                "_act",
                new=AsyncMock(side_effect=RuntimeError("fail")),
            ),
            patch.object(reconciler, "_update_status", new=AsyncMock()),
        ):
            await reconciler._reconcile_one(intent)

        assert reconciler._backoff_active(intent.id) is True
        assert reconciler._backoff[intent.id]["failures"] == 1


# ── _act tests ──────────────────────────────────────────────────


class TestActRecreate:
    """_act RECREATE: restart-or-recreate with backoff (§8.2)."""

    @pytest.mark.anyio
    async def test_recreate_restarts_failed_instance(self):
        """RECREATE restarts the instance in place (spec §8.2)."""
        reconciler = Reconciler()
        host = _HostStub(id="h1")
        action = Action(
            type=ActionType.RECREATE,
            intent_id="intent-001",
            alias="test-model",
            host_id="h1",
            instance_id="inst-1",
            reason="Instance stopped, recreating",
        )
        with (
            patch("app.database.hosts.host_db") as mock_db,
            patch.object(reconciler, "_start_instance", new=AsyncMock()) as mock_start,
        ):
            mock_db.get_host = AsyncMock(return_value=host)
            result = await reconciler._act(_make_intent(), action)
            mock_start.assert_awaited_once_with(host, "inst-1")
            assert result is None  # restart path returns None (next tick is no-op)

    @pytest.mark.anyio
    async def test_recreate_deletes_and_raises_when_restart_fails(self):
        """Restart failure → delete broken replica + raise (backoff recorded)."""
        from fastapi import HTTPException

        reconciler = Reconciler()
        host = _HostStub(id="h1")
        action = Action(
            type=ActionType.RECREATE,
            intent_id="intent-001",
            alias="test-model",
            host_id="h1",
            instance_id="inst-1",
            reason="Instance failed, recreating",
        )
        with (
            patch("app.database.hosts.host_db") as mock_db,
            patch.object(
                reconciler,
                "_start_instance",
                new=AsyncMock(
                    side_effect=HTTPException(status_code=502, detail="boom")
                ),
            ),
            patch.object(
                reconciler, "_delete_instance", new=AsyncMock()
            ) as mock_delete,
        ):
            mock_db.get_host = AsyncMock(return_value=host)
            with pytest.raises(HTTPException):
                await reconciler._act(_make_intent(), action)
            mock_delete.assert_awaited_once_with(host, "inst-1")

    @pytest.mark.anyio
    async def test_recreate_restart_failure_records_backoff(self):
        """Failed recreate (via _reconcile_one) engages exponential backoff."""
        from fastapi import HTTPException

        reconciler = Reconciler()
        intent = _make_intent(replicas=1)
        managed = [_make_managed_instance("inst-1", status="failed")]
        observed = _make_observed(managed=managed)

        async def boom(intent_, action):
            raise HTTPException(status_code=502, detail="start failed")

        with (
            patch.object(reconciler, "_observe", new=AsyncMock(return_value=observed)),
            patch.object(reconciler, "_act", new=AsyncMock(side_effect=boom)),
            patch.object(reconciler, "_update_status", new=AsyncMock()),
        ):
            await reconciler._reconcile_one(intent)
        assert reconciler._backoff_active(intent.id) is True


class TestActMigrate:
    """_act MIGRATE: stop fallback when no target exists (§8.5)."""

    @pytest.mark.anyio
    async def test_migrate_no_target_ephemeral_falls_back_to_stop(self):
        """No migration target + ephemeral instance → stop+delete fallback."""
        reconciler = Reconciler()
        host = _HostStub(id="h1")
        action = Action(
            type=ActionType.MIGRATE,
            intent_id="intent-001",
            alias="other-alias",
            host_id="h1",
            instance_id="inst-1",
            reason="Displacing other-alias (ephemeral) to free capacity",
        )
        with (
            patch("app.database.hosts.host_db") as mock_db,
            patch("app.redis_state.host_store") as mock_store,
            patch(
                "app.services.migration.stop_source_instance", new=AsyncMock()
            ) as mock_stop,
            patch.object(
                reconciler, "_delete_instance", new=AsyncMock()
            ) as mock_delete,
        ):
            mock_db.get_host = AsyncMock(return_value=host)
            mock_db.get_all_hosts = AsyncMock(return_value=[])
            with patch(
                "app.services.placement.find_candidates",
                new=AsyncMock(return_value=[]),
            ):
                mock_store.get_host_instances = AsyncMock(
                    return_value=[{"instance_id": "inst-1", "priority": "ephemeral"}]
                )
                await reconciler._act(_make_intent(), action)
            mock_stop.assert_awaited_once_with(host, "inst-1")
            mock_delete.assert_awaited_once_with(host, "inst-1")

    @pytest.mark.anyio
    async def test_migrate_no_target_staging_not_stopped(self):
        """No migration target + staging instance → no stop fallback."""
        reconciler = Reconciler()
        host = _HostStub(id="h1")
        action = Action(
            type=ActionType.MIGRATE,
            intent_id="intent-001",
            alias="other-alias",
            host_id="h1",
            instance_id="inst-1",
            reason="Displacing other-alias (staging) to free capacity",
        )
        with (
            patch("app.database.hosts.host_db") as mock_db,
            patch("app.redis_state.host_store") as mock_store,
            patch(
                "app.services.migration.stop_source_instance", new=AsyncMock()
            ) as mock_stop,
            patch.object(
                reconciler, "_delete_instance", new=AsyncMock()
            ) as mock_delete,
        ):
            mock_db.get_host = AsyncMock(return_value=host)
            mock_db.get_all_hosts = AsyncMock(return_value=[])
            with patch(
                "app.services.placement.find_candidates",
                new=AsyncMock(return_value=[]),
            ):
                mock_store.get_host_instances = AsyncMock(
                    return_value=[{"instance_id": "inst-1", "priority": "staging"}]
                )
                result = await reconciler._act(_make_intent(), action)
            assert result is None
            mock_stop.assert_not_called()
            mock_delete.assert_not_called()


class TestUpdateStatusConditions:
    """Status conditions emitted by _update_status (§10.3)."""

    @pytest.mark.anyio
    async def test_update_status_emits_degraded_condition(self):
        """DEGRADED phase → Degraded condition, not Progressing."""
        reconciler = Reconciler()
        intent = _make_intent(
            status=IntentStatus(
                phase=IntentPhase.DEGRADED,
                reconcile=ReconcileState.IN_PROGRESS,
                desired_replicas=2,
            )
        )
        observed = _make_observed(
            managed=[_make_managed_instance("inst-1", status="running")],
            gateway_aliases={"test-model"},
        )
        with patch("app.database.intents.intent_db") as mock_db:
            mock_db.update_status = AsyncMock()
            await reconciler._update_status(intent, observed)
        _, kwargs = mock_db.update_status.call_args
        status_json = kwargs["status_json"]
        types = {c["type"] for c in status_json["conditions"]}
        assert "Degraded" in types
        assert "Progressing" not in types

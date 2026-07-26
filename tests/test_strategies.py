"""Comprehensive tests for S-042 deployment strategies module.

Tests cover:
  - check_instance_healthy_sync() health gate
  - RollingStrategy.init() and RollingStrategy.continue_step()
  - ImmediateStrategy.init() and ImmediateStrategy.continue_step()
  - initiate_strategy() and continue_strategy() dispatch
  - Failure modes: timeout, shortfall, in-place replacement
"""

from dataclasses import dataclass


from app.services.strategies import (
    ImmediateStrategy,
    RollingStrategy,
    StrategyPhase,
    _advance_to_next_rolling,
    _count_updated,
    _find_instance_on_host,
    _find_old_instance,
    _strategy_failed,
    _strategy_held,
    check_instance_healthy_sync,
    continue_strategy,
    initiate_strategy,
    should_initiate_strategy,
)

# ── Test stubs ──────────────────────────────────────────────────


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
class _IntentStub:
    """Minimal intent stub for dispatch tests."""

    id: str = "intent-001"
    alias: str = "test-model"
    model_source: str = "repo://test:v2"
    replicas: int = 3
    strategy: str = "rolling"


def _make_managed_instance(
    instance_id,
    host_id="host-1",
    host_name="host1",
    alias="test-model",
    model_source="repo://test:v1",
    status="running",
    **extra_config,
):
    """Build a managed instance dict matching the Redis-backed format."""
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


# ── Helpers ─────────────────────────────────────────────────────


def _make_candidates(*host_ids):
    """Build a list of (HostStub, None) tuples for candidate hosts."""
    return [(_HostStub(id=h), None) for h in host_ids]


# ═════════════════════════════════════════════════════════════════
#  health gate tests
# ═════════════════════════════════════════════════════════════════


class TestCheckInstanceHealthySync:
    """Tests for check_instance_healthy_sync() — §11.1 health gate."""

    def test_none_instance_data_returns_false(self):
        result = check_instance_healthy_sync(
            instance_data=None,
            alias="test-model",
            gateway_aliases={"test-model"},
        )
        assert result is False

    def test_status_not_running_returns_false(self):
        inst = _make_managed_instance("inst-1", status="starting")
        result = check_instance_healthy_sync(
            instance_data=inst,
            alias="test-model",
            gateway_aliases={"test-model"},
        )
        assert result is False

    def test_alias_not_in_gateway_returns_false(self):
        inst = _make_managed_instance("inst-1", status="running")
        result = check_instance_healthy_sync(
            instance_data=inst,
            alias="test-model",
            gateway_aliases=set(),
        )
        assert result is False

    def test_running_and_in_gateway_returns_true(self):
        inst = _make_managed_instance("inst-1", status="running")
        result = check_instance_healthy_sync(
            instance_data=inst,
            alias="test-model",
            gateway_aliases={"test-model", "other-model"},
        )
        assert result is True

    def test_state_field_as_fallback(self):
        """When 'status' is absent, 'state' field is used as fallback."""
        inst = _make_managed_instance("inst-1", status="running")
        del inst["status"]
        inst["state"] = "running"
        result = check_instance_healthy_sync(
            instance_data=inst,
            alias="test-model",
            gateway_aliases={"test-model"},
        )
        assert result is True

    def test_state_field_non_running_returns_false(self):
        inst = _make_managed_instance("inst-1", status="running")
        del inst["status"]
        inst["state"] = "error"
        result = check_instance_healthy_sync(
            instance_data=inst,
            alias="test-model",
            gateway_aliases={"test-model"},
        )
        assert result is False

    def test_both_status_and_state_absent_returns_false(self):
        inst = _make_managed_instance("inst-1", status="running")
        del inst["status"]
        # No 'state' key either — .get("state", "") returns ""
        result = check_instance_healthy_sync(
            instance_data=inst,
            alias="test-model",
            gateway_aliases={"test-model"},
        )
        assert result is False


# ═════════════════════════════════════════════════════════════════
#  internal helpers
# ═════════════════════════════════════════════════════════════════


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_count_updated_all_on_target(self):
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v2"),
        ]
        assert _count_updated(instances, "repo://test:v2") == 2

    def test_count_updated_mixed(self):
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
        ]
        assert _count_updated(instances, "repo://test:v2") == 1

    def test_count_updated_empty(self):
        assert _count_updated([], "repo://test:v2") == 0

    def test_find_instance_on_host_by_instance_id(self):
        instances = [
            _make_managed_instance("i1", host_id="h1"),
            _make_managed_instance("i2", host_id="h2"),
        ]
        found = _find_instance_on_host(instances, host_id=None, instance_id="i2")
        assert found is not None
        assert found["instance_id"] == "i2"

    def test_find_instance_on_host_by_host_id_no_instance_id(self):
        instances = [
            _make_managed_instance("i1", host_id="h1"),
            _make_managed_instance("i2", host_id="h2"),
        ]
        found = _find_instance_on_host(instances, host_id="h2", instance_id=None)
        assert found is not None
        assert found["_host_id"] == "h2"

    def test_find_instance_on_host_not_found(self):
        instances = [_make_managed_instance("i1", host_id="h1")]
        found = _find_instance_on_host(instances, host_id="h99", instance_id=None)
        assert found is None

    def test_find_old_instance_finds_mismatched_source(self):
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]
        found = _find_old_instance(instances, "h1", "repo://test:v2")
        assert found is not None
        assert found["instance_id"] == "i1"

    def test_find_old_instance_skips_matching_source(self):
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]
        found = _find_old_instance(instances, "h1", "repo://test:v2")
        assert found is None

    def test_find_old_instance_wrong_host(self):
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]
        found = _find_old_instance(instances, "h2", "repo://test:v2")
        assert found is None

    def test_strategy_held_sets_failed_phase(self):
        progress = {"phase": "waiting_healthy", "failed": 0, "in_progress": 1}
        action, new_progress = _strategy_held(progress, "timeout exceeded")
        assert action == {"type": "wait", "reason": "strategy held"}
        assert new_progress["phase"] == StrategyPhase.FAILED
        assert new_progress["failed"] == 1
        assert new_progress["in_progress"] == 0
        assert "timeout exceeded" in new_progress["message"]

    def test_strategy_failed_sets_failed_phase_no_action(self):
        progress = {"phase": "creating_replacement"}
        action, new_progress = _strategy_failed(progress, "permanent failure")
        assert action is None
        assert new_progress["phase"] == StrategyPhase.FAILED
        assert "permanent failure" in new_progress["message"]


# ═════════════════════════════════════════════════════════════════
#  scale-up: same source, more replicas (§11.6 scenario 1)
# ═════════════════════════════════════════════════════════════════


class TestScaleUp:
    """Scale-up scenarios (§11.6 scenario 1)."""

    def test_already_at_desired_state_returns_none(self):
        """All instances on target source at desired count → no strategy needed."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v2"),
        ]
        candidates = _make_candidates("h3")

        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=instances,
            candidates=candidates,
        )
        assert result is None, "Should return None when already at desired state"

    def test_scale_up_same_source_returns_none(self):
        """Pure scale-up (same source, more replicas) returns None —
        handled by normal reconciliation diff, not by strategy module."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v2"),
        ]
        candidates = _make_candidates("h3", "h4")

        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=3,  # scale up from 2 to 3
            managed_instances=instances,
            candidates=candidates,
        )
        assert (
            result is None
        ), "Pure scale-up should return None — handled by normal diff"

    def test_initial_deployment_from_zero_returns_none(self):
        """Initial deployment (0→N) with no existing instances returns None —
        handled by normal reconciliation diff, not by strategy module."""
        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=3,
            managed_instances=[],
            candidates=_make_candidates("h1", "h2", "h3"),
        )
        assert (
            result is None
        ), "Initial deployment should return None — handled by normal diff"


# ═════════════════════════════════════════════════════════════════
#  model version change — rolling (§11.6 scenario 3)
# ═════════════════════════════════════════════════════════════════


class TestRollingModelVersionChange:
    """Rolling update: create → wait-healthy → retire-old → advance (§11.6 scenario 3)."""

    # ── init ────────────────────────────────────────────────────

    def test_init_creates_progress_with_creating_replacement_phase(self):
        """init() creates proper progress with creating_replacement phase."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
        ]
        candidates = _make_candidates("h3", "h4")

        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["phase"] == StrategyPhase.CREATING_REPLACEMENT
        assert result["strategy"] == "rolling"
        assert result["target_model_source"] == "repo://test:v2"
        assert result["step"] == "1/2"
        assert result["updated"] == 0
        assert result["in_progress"] == 1
        assert result["failed"] == 0
        assert result["current_host_id"] is not None
        assert result["current_instance_id"] is None
        assert result["pending_hosts"] is not None
        assert len(result["pending_hosts"]) >= 0
        assert result["started_at"] is not None
        assert "Creating replacement" in result["message"]

    # ── Phase: creating_replacement → waiting_healthy ──────────

    def test_creating_replacement_without_instance_id_emits_create(self):
        """When current_instance_id is None, emits a create action."""
        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENT,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": None,
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Creating replacement on h1",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is not None
        assert action["type"] == "create"
        assert action["host_id"] == "h1"
        assert "Rolling replacement step" in action["reason"]
        # Progress unchanged — reconciler sets current_instance_id
        assert new_progress == progress

    def test_creating_replacement_with_instance_id_transitions_to_waiting(self):
        """When current_instance_id is set, transitions to waiting_healthy phase."""
        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENT,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Creating replacement on h1",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action == {"type": "wait", "reason": "awaiting health"}
        assert new_progress["phase"] == StrategyPhase.WAITING_HEALTHY
        assert "Waiting for replacement" in new_progress["message"]

    # ── Phase: waiting_healthy → retiring_old ──────────────────

    def test_waiting_healthy_when_healthy_transitions_to_retiring_old(self):
        """Healthy replacement triggers transition to retiring_old with stop action."""
        # Old instance on h1, replacement on h1
        old_inst = _make_managed_instance(
            "i1", host_id="h1", model_source="repo://test:v1", status="running"
        )
        replacement = _make_managed_instance(
            "inst-new-1", host_id="h1", model_source="repo://test:v2", status="running"
        )
        managed = [old_inst, replacement]

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.WAITING_HEALTHY,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Waiting for replacement",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=managed,
            candidates=[],
            gateway_aliases={"test-model"},
            health_gate_started_at=10.0,
            health_gate_timeout_s=300.0,
        )

        assert action is not None
        assert action["type"] == "stop"
        assert action["host_id"] == "h1"
        assert action["instance_id"] == "i1"
        assert "retiring old replica" in action["reason"]
        assert new_progress["phase"] == StrategyPhase.RETIRING_OLD
        assert "Retiring old replica" in new_progress["message"]

    def test_waiting_healthy_no_old_instance_advances(self):
        """When no old instance exists (fresh host), advances to next slot."""
        replacement = _make_managed_instance(
            "inst-new-1", host_id="h3", model_source="repo://test:v2", status="running"
        )
        managed = [replacement]

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.WAITING_HEALTHY,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h3",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h4"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Waiting for replacement",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=managed,
            candidates=[_HostStub(id="h4"), _HostStub(id="h5")],
            gateway_aliases={"test-model"},
            health_gate_started_at=10.0,
            health_gate_timeout_s=300.0,
        )

        # Should advance — emit create for next host
        assert action is not None
        assert action["type"] == "create"
        assert action["host_id"] == "h4"
        assert new_progress["phase"] == StrategyPhase.CREATING_REPLACEMENT

    # ── Phase: retiring_old ────────────────────────────────────

    def test_retiring_old_waits_while_old_still_exists(self):
        """While the old instance still exists in managed, emit wait."""
        old_inst = _make_managed_instance(
            "i1", host_id="h1", model_source="repo://test:v1", status="stopping"
        )
        managed = [old_inst]

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.RETIRING_OLD,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Retiring old replica",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=managed,
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action == {"type": "wait", "reason": "awaiting stop"}
        assert "Waiting for old replica" in new_progress["message"]

    def test_retiring_old_advances_when_old_gone(self):
        """When old instance is gone, advances to next host or completes."""
        replacement = _make_managed_instance(
            "inst-new-1", host_id="h1", model_source="repo://test:v2", status="running"
        )
        managed = [replacement]  # old instance "i1" removed

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.RETIRING_OLD,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Retiring old replica",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=managed,
            candidates=_make_candidates("h2"),
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        # Should advance to next host
        assert action is not None
        assert action["type"] == "create"
        assert action["host_id"] == "h2"
        assert new_progress["phase"] == StrategyPhase.CREATING_REPLACEMENT

    # ── Full lifecycle: completion ─────────────────────────────

    def test_strategy_completes_when_all_replicas_updated(self):
        """Strategy returns (None, None) when all replicas are on target source."""
        # All instances on target source, count matches desired
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v2"),
        ]

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.RETIRING_OLD,
            "step": "2/2",
            "updated": 1,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h2",
            "current_instance_id": "inst-new-2",
            "pending_hosts": [],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Retiring old replica",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=instances,
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        # Strategy complete — both None
        assert action is None
        assert new_progress is None

    # ── End-to-end rolling lifecycle test ──────────────────────

    def test_full_rolling_lifecycle(self):
        """Exercise the complete rolling state machine across all phases."""
        # Start: 2 instances on v1, target v2
        old_i1 = _make_managed_instance(
            "i1", host_id="h1", model_source="repo://test:v1"
        )
        old_i2 = _make_managed_instance(
            "i2", host_id="h2", model_source="repo://test:v1"
        )

        # Init
        progress = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2],
            candidates=_make_candidates("h3", "h4"),
        )
        assert progress["phase"] == StrategyPhase.CREATING_REPLACEMENT
        assert progress["current_host_id"] is not None

        # Step 1: creating_replacement — emit create
        action, _ = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2],
            candidates=_make_candidates("h3", "h4"),
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action["type"] == "create"

        # Simulate reconciler setting instance_id
        progress = dict(progress)
        progress["current_instance_id"] = "new-1"

        # Step 2: creating_replacement with instance_id → wait
        action, progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2],
            candidates=_make_candidates("h3", "h4"),
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action == {"type": "wait", "reason": "awaiting health"}
        assert progress["phase"] == StrategyPhase.WAITING_HEALTHY

        # Step 3: waiting_healthy — instance is healthy
        new_inst = _make_managed_instance(
            "new-1",
            host_id=progress["current_host_id"],
            model_source="repo://test:v2",
            status="running",
        )
        action, progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2, new_inst],
            candidates=_make_candidates("h3", "h4"),
            gateway_aliases={"test-model"},
            health_gate_started_at=10.0,
            health_gate_timeout_s=300.0,
        )
        assert action["type"] == "stop"  # retiring old
        assert progress["phase"] == StrategyPhase.RETIRING_OLD

        # Step 4: retiring_old — old instance gone
        action, progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_i2, new_inst],  # old_i1 removed
            candidates=_make_candidates("h3", "h4"),
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action["type"] == "create"  # advance to next host
        assert progress["phase"] == StrategyPhase.CREATING_REPLACEMENT


# ═════════════════════════════════════════════════════════════════
#  model version change — immediate (§11.6 scenario 4)
# ═════════════════════════════════════════════════════════════════


class TestImmediateModelVersionChange:
    """Immediate update: stop-all → create-all (§11.6 scenario 4)."""

    # ── init ────────────────────────────────────────────────────

    def test_init_creates_stopping_old_phase(self):
        """init() with source drift creates stopping_old phase."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
        ]
        candidates = _make_candidates("h3", "h4")

        result = ImmediateStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["phase"] == StrategyPhase.STOPPING_OLD
        assert result["strategy"] == "immediate"
        assert result["target_model_source"] == "repo://test:v2"
        assert result["in_progress"] == 2  # 2 drifted instances
        assert result["updated"] == 0
        assert result["failed"] == 0
        assert result["pending_hosts"] == ["h3", "h4"]
        assert "Stopping 2 old replica" in result["message"]

    def test_init_returns_none_when_already_at_target(self):
        """No drift → init returns None (no strategy needed)."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v2"),
        ]

        result = ImmediateStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=instances,
            candidates=[],
        )

        assert result is None

    # ── Phase: stopping_old ────────────────────────────────────

    def test_continue_step_stopping_old_emits_stop(self):
        """stopping_old emits stop for first old instance found."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
        ]

        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.STOPPING_OLD,
            "step": "0/2",
            "updated": 0,
            "in_progress": 2,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": ["h3", "h4"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Stopping 2 old replica(s)",
        }

        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=instances,
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action["type"] == "stop"
        assert action["host_id"] == "h1"
        assert action["instance_id"] == "i1"
        assert "Immediate: stopping old replica" in action["reason"]

    def test_continue_step_stops_all_old_then_transitions(self):
        """After all old instances are stopped, transitions to creating_replacements."""
        # No old instances left — all stopped
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.STOPPING_OLD,
            "step": "0/2",
            "updated": 0,
            "in_progress": 2,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": ["h3", "h4"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Stopping 2 old replica(s)",
        }

        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],  # all old instances removed
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action == {"type": "wait", "reason": "transitioning"}
        assert new_progress["phase"] == StrategyPhase.CREATING_REPLACEMENTS
        assert "Creating 2 replacement" in new_progress["message"]

    def test_stopping_old_with_no_replacements_needed_returns_none(self):
        """When all old are stopped AND updated >= desired, returns (None, None)."""
        # All instances are already on target — should not happen in normal
        # flow but guard is in place
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.STOPPING_OLD,
            "step": "0/2",
            "updated": 2,
            "in_progress": 0,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": [],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Stopping old replicas",
        }

        # Managed instances already on target
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]

        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=1,
            managed_instances=instances,
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is None
        assert new_progress is None

    # ── Phase: creating_replacements ────────────────────────────

    def test_creating_replacements_emits_create(self):
        """creating_replacements emits create for each pending host."""
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENTS,
            "step": "0/2",
            "updated": 0,
            "in_progress": 2,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": ["h3", "h4"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Creating 2 replacement(s)",
        }

        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action["type"] == "create"
        assert action["host_id"] == "h3"
        assert "Immediate: creating replacement" in action["reason"]
        assert new_progress["pending_hosts"] == ["h4"]

    def test_creating_replacements_completes_when_no_pending(self):
        """When no hosts left to create, returns (None, None)."""
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENTS,
            "step": "0/2",
            "updated": 0,
            "in_progress": 0,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": [],  # all dispatched
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Creating replacements",
        }

        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is None
        assert new_progress is None


# ═════════════════════════════════════════════════════════════════
#  failed health check — rolling holds (§11.6 scenario 5)
# ═════════════════════════════════════════════════════════════════


class TestFailedHealthCheckRolling:
    """Timeout on health gate causes strategy to hold (§11.6 scenario 5)."""

    def test_waiting_healthy_timeout_sets_phase_failed(self):
        """When health gate timeout exceeded, phase → failed, keeps old replicas."""
        old_inst = _make_managed_instance(
            "i1", host_id="h1", model_source="repo://test:v1"
        )
        replacement = _make_managed_instance(
            "inst-new-1",
            host_id="h1",
            model_source="repo://test:v2",
            status="starting",  # not running yet
        )

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.WAITING_HEALTHY,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Waiting for replacement",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_inst, replacement],  # old still there
            candidates=[],
            gateway_aliases={"test-model"},
            health_gate_started_at=500.0,  # > 300s timeout
            health_gate_timeout_s=300.0,
        )

        assert new_progress["phase"] == StrategyPhase.FAILED
        assert new_progress["failed"] == 1
        assert new_progress["in_progress"] == 0
        assert "Health gate timeout" in new_progress["message"]

    def test_waiting_healthy_still_waiting_within_timeout(self):
        """Within timeout, emits wait — does not fail."""
        old_inst = _make_managed_instance(
            "i1", host_id="h1", model_source="repo://test:v1"
        )

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.WAITING_HEALTHY,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Waiting for replacement",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_inst],  # replacement not in managed yet
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=50.0,  # within 300s
            health_gate_timeout_s=300.0,
        )

        assert action == {"type": "wait", "reason": "health gate"}
        assert new_progress["phase"] == StrategyPhase.WAITING_HEALTHY
        assert "Waiting for replacement" in new_progress["message"]


# ═════════════════════════════════════════════════════════════════
#  failed health check — immediate degraded (§11.6 scenario 6)
# ═════════════════════════════════════════════════════════════════


class TestFailedHealthCheckImmediate:
    """Immediate strategy — some replacements succeed, some fail (§11.6 scenario 6)."""

    def test_immediate_held_when_no_hosts_available(self):
        """When transition to creating_replacements finds no hosts, strategy held."""
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.STOPPING_OLD,
            "step": "0/3",
            "updated": 0,
            "in_progress": 3,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": [],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Stopping old replicas",
        }

        # No old instances, no candidates → no hosts for replacement
        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=3,
            managed_instances=[],  # all old stopped
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert new_progress["phase"] == StrategyPhase.FAILED
        assert "No hosts available" in new_progress["message"]

    def test_partial_success_some_replacements_dispatch(self):
        """With 3 desired replicas but only 2 hosts, 2 replacements dispatched."""
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENTS,
            "step": "0/3",
            "updated": 0,
            "in_progress": 3,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": ["h3", "h4"],  # only 2 hosts for 3 needed
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Creating replacements",
        }

        # Dispatch first
        action1, progress1 = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=3,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action1["type"] == "create"
        assert action1["host_id"] == "h3"

        # Dispatch second
        action2, progress2 = ImmediateStrategy.continue_step(
            progress_data=progress1,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=3,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action2["type"] == "create"
        assert action2["host_id"] == "h4"

        # Third call — no more pending, strategy complete
        action3, progress3 = ImmediateStrategy.continue_step(
            progress_data=progress2,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=3,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action3 is None
        assert progress3 is None


# ═════════════════════════════════════════════════════════════════
#  shortfall — fewer hosts than replicas (§11.6 scenario 7)
# ═════════════════════════════════════════════════════════════════


class TestShortfall:
    """Shortfall: fewer candidate hosts than needed replicas (§11.6 scenario 7)."""

    def test_rolling_init_with_shortfall_still_creates_progress(self):
        """When candidates < needed, init still creates progress with available hosts."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
            _make_managed_instance("i3", host_id="h3", model_source="repo://test:v1"),
        ]
        # Only 1 new candidate host, but we need 3 replacements
        candidates = _make_candidates("h4")

        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=3,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["phase"] == StrategyPhase.CREATING_REPLACEMENT
        # First replacement uses a drifted host (h1), then candidate (h4)
        # drifted_host_ids = {h1, h2, h3}, candidate_hosts = [h4]
        # all_replacement = [h1, h2, h3, h4] → unique → [h1, h2, h3, h4]
        # replacement_hosts[:3] = [h1, h2, h3]
        assert result["current_host_id"] is not None
        assert len(result["pending_hosts"]) == 2  # h2, h3

    def test_immediate_init_with_shortfall_still_creates_progress(self):
        """Immediate init with few candidate hosts still creates progress."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
            _make_managed_instance("i3", host_id="h3", model_source="repo://test:v1"),
        ]
        candidates = _make_candidates("h4")  # only 1 candidate for 3 needed

        result = ImmediateStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=3,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["phase"] == StrategyPhase.STOPPING_OLD
        assert result["in_progress"] == 3  # 3 drifted
        # Only 1 pending host (h4), but 3 needed — shortfall
        assert len(result["pending_hosts"]) == 1  # [h4]


# ═════════════════════════════════════════════════════════════════
#  in-place replacement (§11.6 scenario 8)
# ═════════════════════════════════════════════════════════════════


class TestInPlaceReplacement:
    """Rolling in-place: replace old instance on same host (§11.6 scenario 8)."""

    def test_in_place_init_uses_drifted_hosts_as_targets(self):
        """When no new candidate hosts, drifted hosts become replacement targets."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
        ]
        candidates = []  # no new hosts

        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["phase"] == StrategyPhase.CREATING_REPLACEMENT
        # Drifted hosts used as replacement targets
        assert result["current_host_id"] in ("h1", "h2")
        assert result["pending_hosts"] is not None

    def test_in_place_full_lifecycle(self):
        """Complete in-place replacement: stop old on h1, create new on h1."""
        old_i1 = _make_managed_instance(
            "i1", host_id="h1", model_source="repo://test:v1"
        )
        old_i2 = _make_managed_instance(
            "i2", host_id="h2", model_source="repo://test:v1"
        )

        progress = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2],
            candidates=[],  # in-place
        )

        first_host = progress["current_host_id"]
        assert first_host in ("h1", "h2")

        # Emit create
        action, _ = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert action["type"] == "create"
        assert action["host_id"] == first_host

        # Simulate instance created → transition to waiting_healthy
        progress = dict(progress)
        progress["current_instance_id"] = "new-on-same-host"
        action, progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[old_i1, old_i2],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )
        assert progress["phase"] == StrategyPhase.WAITING_HEALTHY

        # Healthy → retire old on same host
        new_inst = _make_managed_instance(
            "new-on-same-host",
            host_id=first_host,
            model_source="repo://test:v2",
            status="running",
        )
        all_managed = [old_i1, old_i2, new_inst]
        action, progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=all_managed,
            candidates=[],
            gateway_aliases={"test-model"},
            health_gate_started_at=10.0,
            health_gate_timeout_s=300.0,
        )
        assert action["type"] == "stop"
        assert action["host_id"] == first_host

    def test_in_place_no_drifted_hosts_no_candidates_returns_none(self):
        """When all instances are on target source, init returns None regardless."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]

        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=1,
            managed_instances=instances,
            candidates=[],
        )

        assert result is None


# ═════════════════════════════════════════════════════════════════
#  initiate_strategy / continue_strategy dispatch (§11.6 scenario 9)
# ═════════════════════════════════════════════════════════════════


class TestDispatchFunctions:
    """initiate_strategy() and continue_strategy() dispatch tests (§11.6 scenario 9)."""

    def test_initiate_rolling_delegates_to_rolling_strategy(self):
        """strategy='rolling' returns RollingStrategy.init result."""
        intent = _IntentStub(
            strategy="rolling", model_source="repo://test:v2", replicas=2
        )
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]
        candidates = _make_candidates("h2")

        result = initiate_strategy(
            intent=intent,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["strategy"] == "rolling"
        assert result["phase"] == StrategyPhase.CREATING_REPLACEMENT

    def test_initiate_immediate_delegates_to_immediate_strategy(self):
        """strategy='immediate' returns ImmediateStrategy.init result."""
        intent = _IntentStub(
            strategy="immediate", model_source="repo://test:v2", replicas=2
        )
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]
        candidates = _make_candidates("h2")

        result = initiate_strategy(
            intent=intent,
            managed_instances=instances,
            candidates=candidates,
        )

        assert result is not None
        assert result["strategy"] == "immediate"
        assert result["phase"] == StrategyPhase.STOPPING_OLD

    def test_initiate_unknown_strategy_returns_none(self):
        """Unknown strategy name → returns None."""
        intent = _IntentStub(strategy="blue-green", model_source="repo://test:v2")
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]

        result = initiate_strategy(
            intent=intent,
            managed_instances=instances,
            candidates=[],
        )

        assert result is None

    def test_initiate_no_strategy_returns_none(self):
        """No strategy set → returns None."""
        intent = _IntentStub(strategy=None, model_source="repo://test:v2")

        result = initiate_strategy(
            intent=intent,
            managed_instances=[],
            candidates=[],
        )

        assert result is None

    def test_continue_rolling_dispatches(self):
        """continue_strategy with 'rolling' delegates to RollingStrategy.continue_step."""
        intent = _IntentStub(
            strategy="rolling", model_source="repo://test:v2", replicas=2
        )

        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENT,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": None,
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Creating replacement on h1",
        }

        action, _ = continue_strategy(
            progress_data=progress,
            intent=intent,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is not None
        assert action["type"] == "create"

    def test_continue_immediate_dispatches(self):
        """continue_strategy with 'immediate' delegates to ImmediateStrategy.continue_step."""
        intent = _IntentStub(
            strategy="immediate", model_source="repo://test:v2", replicas=2
        )
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]

        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.STOPPING_OLD,
            "step": "0/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": ["h2"],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Stopping old replicas",
        }

        action, _ = continue_strategy(
            progress_data=progress,
            intent=intent,
            managed_instances=instances,
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is not None
        assert action["type"] == "stop"

    def test_continue_unknown_strategy_returns_none_none(self):
        """Unknown strategy in progress_data → returns (None, None)."""
        intent = _IntentStub(strategy="blue-green")

        progress = {
            "strategy": "blue-green",
            "phase": "unknown",
        }

        action, new_progress = continue_strategy(
            progress_data=progress,
            intent=intent,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is None
        assert new_progress is None


# ═════════════════════════════════════════════════════════════════
#  should_initiate_strategy
# ═════════════════════════════════════════════════════════════════


class TestShouldInitiateStrategy:
    """Tests for should_initiate_strategy() dispatch guard."""

    def test_returns_true_when_drift_detected(self):
        """When managed instances have different model_source, returns True."""
        intent = _IntentStub(strategy="rolling", model_source="repo://test:v2")
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]

        assert (
            should_initiate_strategy(intent=intent, managed_instances=instances) is True
        )

    def test_returns_false_when_no_drift(self):
        """All instances on target source → returns False."""
        intent = _IntentStub(strategy="rolling", model_source="repo://test:v2")
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]

        assert (
            should_initiate_strategy(intent=intent, managed_instances=instances)
            is False
        )

    def test_returns_false_for_unsupported_strategy(self):
        """strategy='none' or other → returns False even with drift."""
        intent = _IntentStub(strategy="none", model_source="repo://test:v2")
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]

        assert (
            should_initiate_strategy(intent=intent, managed_instances=instances)
            is False
        )

    def test_returns_false_when_no_target_source(self):
        """No model_source on intent → returns False."""
        intent = _IntentStub(strategy="rolling", model_source="")
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]

        assert (
            should_initiate_strategy(intent=intent, managed_instances=instances)
            is False
        )

    def test_returns_false_for_empty_managed(self):
        """When managed_instances is empty, returns False (drift check passes
        since loop has nothing to iterate, but no scale-up trigger either)."""
        intent = _IntentStub(strategy="rolling", model_source="repo://test:v2")

        assert should_initiate_strategy(intent=intent, managed_instances=[]) is False


# ═════════════════════════════════════════════════════════════════
#  edge cases & additional coverage
# ═════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for strategy robustness."""

    def test_rolling_advance_pending_empty_with_extra_candidates(self):
        """When pending_hosts empty but candidates available, _advance_to_next_rolling
        finds new candidates from the candidates list."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]
        # Only 1 instance on target, need 1 more
        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.RETIRING_OLD,
            "step": "1/2",
            "updated": 1,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": [],  # depleted
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Retiring old",
        }
        candidates = _make_candidates("h3", "h4")

        action, new_progress = _advance_to_next_rolling(
            progress_data=progress,
            managed_instances=instances,
            candidates=candidates,
            desired_replicas=2,
            alias="test-model",
            target_source="repo://test:v2",
        )

        # Should find h3 as an extra candidate
        assert action is not None
        assert action["type"] == "create"
        assert action["host_id"] == "h3"

    def test_rolling_advance_pending_empty_no_candidates(self):
        """When no pending hosts and no extra candidates, advance returns None,None."""
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]
        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.RETIRING_OLD,
            "step": "1/2",
            "updated": 1,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": "h1",
            "current_instance_id": "inst-new-1",
            "pending_hosts": [],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "Retiring old",
        }

        action, new_progress = _advance_to_next_rolling(
            progress_data=progress,
            managed_instances=instances,
            candidates=[],
            desired_replicas=2,
            alias="test-model",
            target_source="repo://test:v2",
        )

        assert action is None
        assert new_progress is None

    def test_creating_replacement_no_host_id(self):
        """When current_host_id is None/empty in creating_replacement, strategy held."""
        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENT,
            "step": "1/2",
            "updated": 0,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": [],
            "failed_hosts": [],
            "started_at": "2024-01-01T00:00:00",
            "message": "",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert new_progress["phase"] == StrategyPhase.FAILED

    def test_unknown_phase_returns_none_none(self):
        """Unknown phase in continue_step returns (None, None)."""
        progress = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": "bogus_phase",
        }

        action, new_progress = RollingStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is None
        assert new_progress is None

    def test_immediate_unknown_phase_returns_none_none(self):
        """Unknown phase in ImmediateStrategy.continue_step returns (None, None)."""
        progress = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": "bogus_phase",
        }

        action, new_progress = ImmediateStrategy.continue_step(
            progress_data=progress,
            intent_id="intent-001",
            alias="test-model",
            desired_replicas=2,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=300.0,
        )

        assert action is None
        assert new_progress is None

    def test_scale_down_during_strategy_continues_normally(self):
        """When managed > desired (scale-down during strategy), the strategy
        module operates correctly — excess handling is the reconciler's
        responsibility (§11.6 scenario 2)."""
        # 3 managed instances on v1, desired=2 on v2
        instances = [
            _make_managed_instance("i1", host_id="h1", model_source="repo://test:v1"),
            _make_managed_instance("i2", host_id="h2", model_source="repo://test:v1"),
            _make_managed_instance("i3", host_id="h3", model_source="repo://test:v1"),
        ]
        candidates = _make_candidates("h4", "h5")

        # Rolling init: despite scale-down (3→2), strategy initiates for
        # model version change (all 3 need replacing).
        result = RollingStrategy.init(
            intent_id="intent-001",
            alias="test-model",
            target_model_source="repo://test:v2",
            desired_replicas=2,
            managed_instances=instances,
            candidates=candidates,
        )
        assert result is not None
        assert result["phase"] == StrategyPhase.CREATING_REPLACEMENT
        assert result["updated"] == 0  # none on target
        # Strategy processes the first replacement on available host
        assert result["current_host_id"] is not None
        assert result["step"] == "1/2"  # needed=2 despite 3 managed

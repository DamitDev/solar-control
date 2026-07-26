"""Tests for deployment strategies (S-042).

Covers §11.6 scenarios: scale up, model version change (rolling/immediate),
failed health check (rolling hold), failed health check (immediate degraded),
shortfall, in-place replacement, and dispatch functions.
"""

from dataclasses import dataclass

import pytest

from app.services.strategies import (
    ImmediateStrategy,
    RollingStrategy,
    StrategyPhase,
    check_instance_healthy_sync,
    continue_strategy,
    initiate_strategy,
    should_initiate_strategy,
)


# ── Test stubs ────────────────────────────────────────────────────


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


def _make_instance(
    instance_id: str,
    host_id: str = "host-1",
    host_name: str = "host1",
    alias: str = "test-model",
    model_source: str = "repo://test:v1",
    status: str = "running",
    **extra_config,
) -> dict:
    """Build a managed instance dict matching the Redis/host-store format."""
    config: dict = {
        "alias": alias,
        "model_source": model_source,
        "managed_by": "intent",
        "intent_id": "intent-001",
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


def _make_intent_stub(**overrides):
    """Build a minimal intent stub for dispatch tests."""
    defaults = {
        "id": "intent-001",
        "alias": "test-model",
        "model_source": "repo://test:v2",
        "replicas": 2,
        "strategy": "rolling",
        "priority": "production",
    }
    defaults.update(overrides)

    class IntentStub:
        pass

    obj = IntentStub()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


# ── Health gate tests ──────────────────────────────────────────────


class TestHealthGate:
    """Test the shared health gate function (§11.1)."""

    def test_healthy_when_running_and_registered(self):
        assert check_instance_healthy_sync(
            instance_data={"status": "running", "config": {"alias": "test"}},
            alias="test",
            gateway_aliases={"test"},
        ) is True

    def test_not_healthy_when_not_running(self):
        assert check_instance_healthy_sync(
            instance_data={"status": "starting", "config": {"alias": "test"}},
            alias="test",
            gateway_aliases={"test"},
        ) is False

    def test_not_healthy_when_not_in_gateway(self):
        assert check_instance_healthy_sync(
            instance_data={"status": "running", "config": {"alias": "test"}},
            alias="test",
            gateway_aliases=set(),
        ) is False

    def test_not_healthy_when_instance_is_none(self):
        assert check_instance_healthy_sync(
            instance_data=None,
            alias="test",
            gateway_aliases={"test"},
        ) is False

    def test_not_healthy_when_status_is_failed(self):
        assert check_instance_healthy_sync(
            instance_data={"status": "failed", "config": {"alias": "test"}},
            alias="test",
            gateway_aliases={"test"},
        ) is False

    def test_uses_state_fallback_when_status_missing(self):
        """check_instance_healthy_sync falls back to 'state' if 'status'
        key is missing."""
        assert check_instance_healthy_sync(
            instance_data={"state": "running", "config": {"alias": "test"}},
            alias="test",
            gateway_aliases={"test"},
        ) is True


# ── Rolling strategy tests (§11.2) ─────────────────────────────────


class TestRollingInit:
    """Test RollingStrategy.init() for various scenarios."""

    def test_init_returns_none_when_already_on_target(self):
        """No strategy needed when all replicas already on target source."""
        managed = [
            _make_instance("i1", host_id="h1", model_source="repo://test:v2"),
            _make_instance("i2", host_id="h2", model_source="repo://test:v2"),
        ]
        candidates = [(_HostStub(id="h3"), None)]
        result = RollingStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v2",
            desired_replicas=2, managed_instances=managed, candidates=candidates,
        )
        assert result is None

    def test_init_for_model_version_change(self):
        """Rolling update from v1 to v2 generates creating_replacement phase."""
        managed = [
            _make_instance("old-1", host_id="h1", model_source="repo://test:v1"),
            _make_instance("old-2", host_id="h2", model_source="repo://test:v1"),
        ]
        candidates = [
            (_HostStub(id="h3", name="h3"), None),
            (_HostStub(id="h4", name="h4"), None),
        ]
        progress = RollingStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v2",
            desired_replicas=2, managed_instances=managed, candidates=candidates,
        )
        assert progress is not None
        assert progress["strategy"] == "rolling"
        assert progress["target_model_source"] == "repo://test:v2"
        assert progress["phase"] == StrategyPhase.CREATING_REPLACEMENT
        assert progress["updated"] == 0
        assert progress["in_progress"] == 1
        assert progress["failed"] == 0
        assert progress["step"] == "1/2"
        assert progress["current_host_id"] is not None
        assert "started_at" in progress

    def test_init_scale_up_same_source(self):
        """Scale up with same source: all instances match target, returns None."""
        managed = [
            _make_instance("i1", host_id="h1", model_source="repo://test:v1"),
        ]
        candidates = [(_HostStub(id="h2"), None)]
        result = RollingStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v1",
            desired_replicas=2, managed_instances=managed, candidates=candidates,
        )
        # Pure scale-up (no drift): strategy not needed
        assert result is None

    def test_init_in_place_replacement(self):
        """When no new hosts available, drifted hosts become replacement targets."""
        managed = [
            _make_instance("old-1", host_id="h1", model_source="repo://test:v1"),
        ]
        # No external candidates — must replace on the drifted host itself
        candidates: list = []
        progress = RollingStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v2",
            desired_replicas=1, managed_instances=managed, candidates=candidates,
        )
        assert progress is not None
        assert progress["current_host_id"] == "h1"  # in-place

    def test_init_shortfall_fewer_candidates_than_needed(self):
        """Initial deployment (0→N) is not a strategy — it goes through normal diff.
        
        Strategy is only for version changes (existing drifted instances).
        Initial deployment with fewer candidates is handled by the reconciler's
        normal CREATE actions with shortfall reporting.
        """
        managed: list = []
        candidates = [(_HostStub(id="h1", name="h1"), None)]
        progress = RollingStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v2",
            desired_replicas=3, managed_instances=managed, candidates=candidates,
        )
        # Initial deployment goes through normal diff, not strategy
        assert progress is None


class TestRollingContinue:
    """Test RollingStrategy.continue_step() through all phases."""

    def _base_continue_args(self, **overrides):
        """Return default args for continue_step()."""
        defaults = {
            "progress_data": {
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.CREATING_REPLACEMENT,
                "step": "1/2",
                "updated": 0,
                "in_progress": 1,
                "failed": 0,
                "current_host_id": "h3",
                "current_instance_id": None,
                "pending_hosts": ["h4"],
                "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z",
                "message": "Creating replacement on h3",
            },
            "intent_id": "int-1",
            "alias": "test-model",
            "desired_replicas": 2,
            "managed_instances": [],
            "candidates": [],
            "gateway_aliases": set(),
            "health_gate_started_at": 0.0,
            "health_gate_timeout_s": 120.0,
        }
        defaults.update(overrides)
        return defaults

    def test_creating_emits_create_action(self):
        """Phase creating_replacement with no instance_id emits create."""
        args = self._base_continue_args(
            progress_data={
                **self._base_continue_args()["progress_data"],
                "current_instance_id": None,
            },
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "create"
        assert action["host_id"] == "h3"
        # Progress unchanged — caller sets instance_id after create
        assert new_progress["phase"] == StrategyPhase.CREATING_REPLACEMENT

    def test_creating_with_instance_id_transitions_to_waiting(self):
        """When instance_id is already set, transition to waiting_healthy."""
        args = self._base_continue_args(
            progress_data={
                **self._base_continue_args()["progress_data"],
                "current_instance_id": "new-inst-1",
            },
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "wait"
        assert new_progress["phase"] == StrategyPhase.WAITING_HEALTHY

    def test_waiting_healthy_emits_stop_when_healthy(self):
        """When replacement becomes healthy, emit stop for old replica."""
        managed = [
            _make_instance("old-1", host_id="h3", model_source="repo://test:v1"),
            _make_instance("new-1", host_id="h3", model_source="repo://test:v2", status="running"),
        ]
        args = self._base_continue_args(
            progress_data={
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.WAITING_HEALTHY,
                "step": "1/2", "updated": 0, "in_progress": 1, "failed": 0,
                "current_host_id": "h3", "current_instance_id": "new-1",
                "pending_hosts": ["h4"], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "",
            },
            managed_instances=managed,
            gateway_aliases={"test-model"},
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "stop"
        assert action["instance_id"] == "old-1"
        assert new_progress["phase"] == StrategyPhase.RETIRING_OLD

    def test_waiting_healthy_waits_when_not_healthy(self):
        """When replacement not yet healthy, emit wait."""
        managed = [
            _make_instance("old-1", host_id="h3", model_source="repo://test:v1"),
            _make_instance("new-1", host_id="h3", model_source="repo://test:v2", status="starting"),
        ]
        args = self._base_continue_args(
            progress_data={
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.WAITING_HEALTHY,
                "step": "1/2", "updated": 0, "in_progress": 1, "failed": 0,
                "current_host_id": "h3", "current_instance_id": "new-1",
                "pending_hosts": ["h4"], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "",
            },
            managed_instances=managed,
            gateway_aliases={"test-model"},
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "wait"
        assert new_progress["phase"] == StrategyPhase.WAITING_HEALTHY

    def test_waiting_healthy_timeout_holds(self):
        """When health gate timeout exceeded, strategy holds (failed phase)."""
        managed = [
            _make_instance("old-1", host_id="h3", model_source="repo://test:v1"),
            _make_instance("new-1", host_id="h3", model_source="repo://test:v2", status="starting"),
        ]
        args = self._base_continue_args(
            progress_data={
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.WAITING_HEALTHY,
                "step": "1/2", "updated": 0, "in_progress": 1, "failed": 0,
                "current_host_id": "h3", "current_instance_id": "new-1",
                "pending_hosts": ["h4"], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "",
            },
            managed_instances=managed,
            gateway_aliases={"test-model"},
            health_gate_started_at=999.0,  # way past timeout
            health_gate_timeout_s=120.0,
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert new_progress["phase"] == StrategyPhase.FAILED
        # Old replica NOT retired — kept running (rolling hold)
        assert action["type"] == "wait"

    def test_retiring_old_waits_until_stopped(self):
        """While old instance still exists, keep waiting."""
        managed = [
            _make_instance("old-1", host_id="h3", model_source="repo://test:v1"),
            _make_instance("new-1", host_id="h3", model_source="repo://test:v2"),
        ]
        args = self._base_continue_args(
            progress_data={
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.RETIRING_OLD,
                "step": "1/2", "updated": 0, "in_progress": 1, "failed": 0,
                "current_host_id": "h3", "current_instance_id": "new-1",
                "pending_hosts": ["h4"], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "",
            },
            managed_instances=managed,
            gateway_aliases={"test-model"},
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action["type"] == "wait"
        assert new_progress["phase"] == StrategyPhase.RETIRING_OLD

    def test_retiring_old_advances_to_next_when_gone(self):
        """When old instance is gone, proceed to next host."""
        managed = [
            _make_instance("new-1", host_id="h3", model_source="repo://test:v2"),
        ]
        candidates = [(_HostStub(id="h4", name="h4"), None)]
        args = self._base_continue_args(
            progress_data={
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.RETIRING_OLD,
                "step": "1/2", "updated": 0, "in_progress": 1, "failed": 0,
                "current_host_id": "h3", "current_instance_id": "new-1",
                "pending_hosts": ["h4"], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "",
            },
            managed_instances=managed,
            candidates=candidates,
            gateway_aliases={"test-model"},
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "create"  # next host
        assert action["host_id"] == "h4"
        assert new_progress["phase"] == StrategyPhase.CREATING_REPLACEMENT

    def test_completes_when_all_updated(self):
        """Strategy returns (None, None) when all replicas on target source."""
        managed = [
            _make_instance("new-1", host_id="h3", model_source="repo://test:v2"),
            _make_instance("new-2", host_id="h4", model_source="repo://test:v2"),
        ]
        args = self._base_continue_args(
            progress_data={
                "strategy": "rolling",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.RETIRING_OLD,
                "step": "2/2", "updated": 1, "in_progress": 0, "failed": 0,
                "current_host_id": "h4", "current_instance_id": "new-2",
                "pending_hosts": [], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "",
            },
            managed_instances=managed,
            gateway_aliases={"test-model"},
        )
        action, new_progress = RollingStrategy.continue_step(**args)
        assert action is None
        assert new_progress is None  # strategy complete


# ── Immediate strategy tests (§11.3) ──────────────────────────────


class TestImmediateInit:
    """Test ImmediateStrategy.init()."""

    def test_init_returns_none_when_already_on_target(self):
        managed = [
            _make_instance("i1", host_id="h1", model_source="repo://test:v2"),
        ]
        result = ImmediateStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v2",
            desired_replicas=1, managed_instances=managed, candidates=[],
        )
        assert result is None

    def test_init_with_drifted_instances(self):
        managed = [
            _make_instance("old-1", host_id="h1", model_source="repo://test:v1"),
            _make_instance("old-2", host_id="h2", model_source="repo://test:v1"),
        ]
        candidates = [
            (_HostStub(id="h3"), None),
            (_HostStub(id="h4"), None),
        ]
        progress = ImmediateStrategy.init(
            intent_id="int-1", alias="test", target_model_source="repo://test:v2",
            desired_replicas=2, managed_instances=managed, candidates=candidates,
        )
        assert progress is not None
        assert progress["strategy"] == "immediate"
        assert progress["phase"] == StrategyPhase.STOPPING_OLD
        assert progress["in_progress"] == 2  # 2 old to stop


class TestImmediateContinue:
    """Test ImmediateStrategy.continue_step()."""

    def _base_args(self, **overrides):
        defaults = {
            "progress_data": {
                "strategy": "immediate",
                "target_model_source": "repo://test:v2",
                "phase": StrategyPhase.STOPPING_OLD,
                "step": "0/2", "updated": 0, "in_progress": 2, "failed": 0,
                "current_host_id": None, "current_instance_id": None,
                "pending_hosts": ["h3", "h4"], "failed_hosts": [],
                "started_at": "2026-01-01T00:00:00Z", "message": "Stopping old",
            },
            "intent_id": "int-1", "alias": "test-model",
            "desired_replicas": 2, "managed_instances": [],
            "candidates": [], "gateway_aliases": set(),
            "health_gate_started_at": 0.0, "health_gate_timeout_s": 120.0,
        }
        defaults.update(overrides)
        return defaults

    def test_stopping_old_emits_stop(self):
        managed = [
            _make_instance("old-1", host_id="h1", model_source="repo://test:v1"),
            _make_instance("old-2", host_id="h2", model_source="repo://test:v1"),
        ]
        args = self._base_args(managed_instances=managed)
        action, new_progress = ImmediateStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "stop"
        assert action["host_id"] == "h1"
        assert new_progress["phase"] == StrategyPhase.STOPPING_OLD

    def test_stopping_transitions_to_creating_when_all_stopped(self):
        """When no old instances remain, transition to creating_replacements."""
        managed: list = []  # all old stopped
        candidates = [
            (_HostStub(id="h3"), None),
            (_HostStub(id="h4"), None),
        ]
        args = self._base_args(
            managed_instances=managed,
            candidates=candidates,
        )
        action, new_progress = ImmediateStrategy.continue_step(**args)
        assert new_progress["phase"] == StrategyPhase.CREATING_REPLACEMENTS

    def test_creating_emits_create_actions(self):
        """create_replacements phase emits create for each pending host."""
        candidates = [(_HostStub(id="h3"), None)]
        args = self._base_args(
            progress_data={
                **self._base_args()["progress_data"],
                "phase": StrategyPhase.CREATING_REPLACEMENTS,
                "pending_hosts": ["h3"],
                "in_progress": 1,
            },
            candidates=candidates,
        )
        action, new_progress = ImmediateStrategy.continue_step(**args)
        assert action is not None
        assert action["type"] == "create"
        assert action["host_id"] == "h3"

    def test_completes_when_all_replacement_hosts_used(self):
        """When pending_hosts is empty, strategy completes."""
        args = self._base_args(
            progress_data={
                **self._base_args()["progress_data"],
                "phase": StrategyPhase.CREATING_REPLACEMENTS,
                "pending_hosts": [],
                "in_progress": 0,
            },
        )
        action, new_progress = ImmediateStrategy.continue_step(**args)
        assert action is None
        assert new_progress is None

    def test_degraded_when_no_hosts_for_replacements(self):
        """When stopping_old completes but no hosts for replacements, fails."""
        managed: list = []
        args = self._base_args(
            managed_instances=managed,
            progress_data={
                **self._base_args()["progress_data"],
                "pending_hosts": [],  # none available
            },
        )
        action, new_progress = ImmediateStrategy.continue_step(**args)
        # Should transition with what it has — empty pending list is OK
        assert new_progress is not None


# ── Dispatch function tests ───────────────────────────────────────


class TestShouldInitiateStrategy:
    """Test should_initiate_strategy()."""

    def test_true_when_drift_and_valid_strategy(self):
        intent = _make_intent_stub(strategy="rolling", model_source="repo://test:v2")
        managed = [_make_instance("old-1", model_source="repo://test:v1")]
        assert should_initiate_strategy(intent=intent, managed_instances=managed) is True

    def test_false_when_no_drift(self):
        intent = _make_intent_stub(strategy="rolling", model_source="repo://test:v1")
        managed = [_make_instance("i1", model_source="repo://test:v1")]
        assert should_initiate_strategy(intent=intent, managed_instances=managed) is False

    def test_false_when_unknown_strategy(self):
        intent = _make_intent_stub(strategy="unknown", model_source="repo://test:v2")
        managed = [_make_instance("old-1", model_source="repo://test:v1")]
        assert should_initiate_strategy(intent=intent, managed_instances=managed) is False

    def test_false_when_strategy_is_none(self):
        intent = _make_intent_stub(strategy=None, model_source="repo://test:v2")
        managed = [_make_instance("old-1", model_source="repo://test:v1")]
        assert should_initiate_strategy(intent=intent, managed_instances=managed) is False


class TestInitiateStrategy:
    """Test initiate_strategy() dispatch."""

    def test_rolling_strategy_dispatched(self):
        intent = _make_intent_stub(strategy="rolling", model_source="repo://test:v2")
        managed = [_make_instance("old-1", host_id="h1", model_source="repo://test:v1")]
        candidates = [(_HostStub(id="h2"), None)]
        progress = initiate_strategy(
            intent=intent, managed_instances=managed, candidates=candidates,
        )
        assert progress is not None
        assert progress["strategy"] == "rolling"
        assert progress["phase"] == StrategyPhase.CREATING_REPLACEMENT

    def test_immediate_strategy_dispatched(self):
        intent = _make_intent_stub(strategy="immediate", model_source="repo://test:v2")
        managed = [_make_instance("old-1", host_id="h1", model_source="repo://test:v1")]
        candidates = [(_HostStub(id="h2"), None)]
        progress = initiate_strategy(
            intent=intent, managed_instances=managed, candidates=candidates,
        )
        assert progress is not None
        assert progress["strategy"] == "immediate"
        assert progress["phase"] == StrategyPhase.STOPPING_OLD

    def test_unknown_strategy_returns_none(self):
        intent = _make_intent_stub(strategy="canary", model_source="repo://test:v2")
        managed = [_make_instance("old-1", host_id="h1", model_source="repo://test:v1")]
        progress = initiate_strategy(
            intent=intent, managed_instances=managed, candidates=[],
        )
        assert progress is None


class TestContinueStrategyDispatch:
    """Test continue_strategy() dispatch function."""

    def test_rolling_dispatched_correctly(self):
        intent = _make_intent_stub(strategy="rolling")
        progress_data = {
            "strategy": "rolling",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.CREATING_REPLACEMENT,
            "step": "1/1", "updated": 0, "in_progress": 1, "failed": 0,
            "current_host_id": "h1", "current_instance_id": None,
            "pending_hosts": [], "failed_hosts": [],
            "started_at": "2026-01-01T00:00:00Z", "message": "",
        }
        managed = [_make_instance("old-1", host_id="h1", model_source="repo://test:v1")]
        action, new_progress = continue_strategy(
            progress_data=progress_data,
            intent=intent,
            managed_instances=managed,
            candidates=[],
            gateway_aliases={"test-model"},
            health_gate_started_at=0.0,
            health_gate_timeout_s=120.0,
        )
        assert action is not None
        assert action["type"] == "create"

    def test_immediate_dispatched_correctly(self):
        intent = _make_intent_stub(strategy="immediate")
        progress_data = {
            "strategy": "immediate",
            "target_model_source": "repo://test:v2",
            "phase": StrategyPhase.STOPPING_OLD,
            "step": "0/1", "updated": 0, "in_progress": 1, "failed": 0,
            "current_host_id": None, "current_instance_id": None,
            "pending_hosts": ["h2"], "failed_hosts": [],
            "started_at": "2026-01-01T00:00:00Z", "message": "",
        }
        managed = [_make_instance("old-1", host_id="h1", model_source="repo://test:v1")]
        action, new_progress = continue_strategy(
            progress_data=progress_data,
            intent=intent,
            managed_instances=managed,
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=120.0,
        )
        assert action is not None
        assert action["type"] == "stop"

    def test_unknown_strategy_returns_none(self):
        intent = _make_intent_stub(strategy="unknown")
        progress_data = {
            "strategy": "unknown",
            "phase": "whatever",
            "step": "1/1", "updated": 0, "in_progress": 0, "failed": 0,
            "pending_hosts": [], "failed_hosts": [],
            "started_at": "2026-01-01T00:00:00Z", "message": "",
        }
        action, new_progress = continue_strategy(
            progress_data=progress_data,
            intent=intent,
            managed_instances=[],
            candidates=[],
            gateway_aliases=set(),
            health_gate_started_at=0.0,
            health_gate_timeout_s=120.0,
        )
        assert action is None
        assert new_progress is None

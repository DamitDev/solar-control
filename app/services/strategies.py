"""Deployment strategies for intent reconciliation (S-042).

Implements rolling and immediate strategies as state machines driven by
the reconciliation loop.  Each strategy persists its progress in intent
status so state survives reconciler restarts.

Strategy phases (rolling):
  creating_replacement → waiting_healthy → retiring_old → (next host or done)

Strategy phases (immediate):
  stopping_old → creating_replacements → done

Health gate (shared, per §11.1):
  replacement is healthy when instance status == "running" AND the alias
  is registered in the gateway registry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Strategy phases ──────────────────────────────────────────────


class StrategyPhase:
    CREATING_REPLACEMENT = "creating_replacement"
    WAITING_HEALTHY = "waiting_healthy"
    RETIRING_OLD = "retiring_old"
    DONE = "done"
    FAILED = "failed"
    # Immediate phases
    STOPPING_OLD = "stopping_old"
    CREATING_REPLACEMENTS = "creating_replacements"


# ── Health gate (shared) ─────────────────────────────────────────


async def check_instance_healthy(
    *,
    instance_data: dict[str, Any] | None,
    host_id: str,
    alias: str,
    gateway_aliases: set[str],
) -> bool:
    """Check if a replacement instance is healthy per §11.1.

    Returns True when ALL hold:
    1. Instance status == "running"
    2. Alias is registered in the gateway registry
    """
    if instance_data is None:
        return False

    status = instance_data.get("status") or instance_data.get("state", "")
    if status != "running":
        return False

    if alias not in gateway_aliases:
        return False

    return True


def check_instance_healthy_sync(
    *,
    instance_data: dict[str, Any] | None,
    alias: str,
    gateway_aliases: set[str],
) -> bool:
    """Synchronous variant of check_instance_healthy.

    Used inside strategy continue_step() which runs in a sync context
    after the reconciler has already collected all observed state.
    """
    if instance_data is None:
        return False

    status = instance_data.get("status") or instance_data.get("state", "")
    if status != "running":
        return False

    if alias not in gateway_aliases:
        return False

    return True


# ── Shared helpers ───────────────────────────────────────────────


def _find_instance_on_host(
    managed: list[dict[str, Any]],
    host_id: str | None,
    instance_id: str | None,
) -> dict[str, Any] | None:
    """Find an instance on a host by instance_id or host_id."""
    for inst in managed:
        hid = inst.get("_host_id")
        iid = inst.get("instance_id") or inst.get("id")
        if instance_id and iid == instance_id:
            return inst
        if host_id and hid == host_id and not instance_id:
            # Return the first instance on this host that's on the
            # *target* source (the replacement, not an old one still
            # being retired).  Callers should prefer instance_id.
            return inst
    return None


def _find_old_instance(
    managed: list[dict[str, Any]],
    host_id: str | None,
    target_source: str,
) -> dict[str, Any] | None:
    """Find an instance on *host_id* whose model_source != target_source."""
    for inst in managed:
        hid = inst.get("_host_id")
        if hid != host_id:
            continue
        cfg = inst.get("config", inst)
        if cfg.get("model_source") != target_source:
            return inst
    return None


def _count_updated(
    managed: list[dict[str, Any]],
    target_source: str,
) -> int:
    """Count managed instances already on the target model_source."""
    count = 0
    for inst in managed:
        cfg = inst.get("config", inst)
        if cfg.get("model_source") == target_source:
            count += 1
    return count


# ── Rolling Strategy (§11.2) ────────────────────────────────────


class RollingStrategy:
    """Rolling update: one host at a time, create→wait-healthy→retire-old.

    Implements deployment-intent.md §11.2.

    State machine:
      creating_replacement: emit create action → transition to waiting_healthy
      waiting_healthy: check health gate → if healthy, transition to retiring_old
      retiring_old: emit stop for old replica → advance to next host or done
    """

    @staticmethod
    def init(
        *,
        intent_id: str,
        alias: str,
        target_model_source: str,
        desired_replicas: int,
        managed_instances: list[dict[str, Any]],
        candidates: list[tuple[Any, Any]],
    ) -> dict[str, Any] | None:
        """Initialize a rolling strategy from current observed state.

        Returns strategy_progress dict, or None if nothing to do (already
        on target source at desired replicas).
        """
        now = datetime.now(timezone.utc).isoformat()

        updated = _count_updated(managed_instances, target_model_source)

        # How many replicas still need to be on the target source?
        needed = desired_replicas - updated
        if needed <= 0:
            return None  # Already at desired state

        # Only initiate strategy when there are drifted instances that
        # need replacing.  Pure scale-up (same source, more replicas)
        # and initial deployment (0→N) are handled by normal diff.
        if updated == len(managed_instances) and needed > 0:
            # All existing instances are already on target source —
            # this is a pure scale-up, not a version change.
            return None

        # Hosts of managed instances that are NOT yet on the target source
        drifted_host_ids = {
            inst.get("_host_id")
            for inst in managed_instances
            if (inst.get("config", inst).get("model_source") != target_model_source)
            and inst.get("_host_id") is not None
        }

        # Replacement hosts: from candidates, exclude hosts already hosting
        # a managed instance (to enforce one-replica-per-host during
        # replacement — the old instance will be retired first).
        candidate_hosts = [h for h, _ in candidates if h.id not in drifted_host_ids]

        # If we have drifted instances, we can also replace in-place:
        # stop old, then create new on same host (candidate hosts list
        # excludes drifted_host_ids, so we add drifted hosts back as
        # potential targets for replacement after their old instance is
        # retired).
        # For in-place: first replacement uses a drifted host directly.
        all_replacement_hosts: list[str] = []
        if drifted_host_ids:
            all_replacement_hosts.extend(sorted(drifted_host_ids))
        all_replacement_hosts.extend(h.id for h in candidate_hosts)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_hosts: list[str] = []
        for h in all_replacement_hosts:
            if h not in seen:
                seen.add(h)
                unique_hosts.append(h)

        # Need at most `needed` replacement hosts
        replacement_hosts = unique_hosts[:needed]

        if not replacement_hosts:
            return None  # No hosts available to place replacements

        first_host = replacement_hosts[0]
        pending = replacement_hosts[1:] if len(replacement_hosts) > 1 else []

        total_steps = needed

        return {
            "strategy": "rolling",
            "target_model_source": target_model_source,
            "phase": StrategyPhase.CREATING_REPLACEMENT,
            "step": f"1/{total_steps}",
            "updated": updated,
            "in_progress": 1,
            "failed": 0,
            "current_host_id": first_host,
            "current_instance_id": None,
            "pending_hosts": pending,
            "failed_hosts": [],
            "started_at": now,
            "message": f"Creating replacement on {first_host}",
        }

    @staticmethod
    def continue_step(
        *,
        progress_data: dict[str, Any],
        intent_id: str,
        alias: str,
        desired_replicas: int,
        managed_instances: list[dict[str, Any]],
        candidates: list[tuple[Any, Any]],
        gateway_aliases: set[str],
        health_gate_started_at: float,
        health_gate_timeout_s: float,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Continue a rolling strategy from its current phase.

        Returns (action, new_progress) where:
        - action: dict with type/host_id/instance_id/reason, or None if done
        - new_progress: updated progress dict, or None if strategy complete
        """
        phase = progress_data.get("phase", "")
        current_host_id = progress_data.get("current_host_id")
        current_instance_id = progress_data.get("current_instance_id")
        target_source = progress_data.get("target_model_source", "")

        # ── PHASE: creating_replacement ──────────────────────────
        if phase == StrategyPhase.CREATING_REPLACEMENT:
            if not current_host_id:
                return _strategy_held(
                    progress_data,
                    "No replacement host available",
                )

            # If we already have a current_instance_id (from a prior
            # tick that executed the create), transition to waiting.
            if current_instance_id:
                new_progress = dict(progress_data)
                new_progress["phase"] = StrategyPhase.WAITING_HEALTHY
                new_progress["message"] = (
                    f"Waiting for replacement on {current_host_id} "
                    f"to become healthy"
                )
                return {"type": "wait", "reason": "awaiting health"}, new_progress

            # Emit create action for the replacement
            return (
                {
                    "type": "create",
                    "host_id": current_host_id,
                    "reason": (
                        f"Rolling replacement step " f"{progress_data.get('step', '?')}"
                    ),
                },
                progress_data,  # unchanged; _continue_strategy sets instance_id
            )

        # ── PHASE: waiting_healthy ───────────────────────────────
        if phase == StrategyPhase.WAITING_HEALTHY:
            # Find the replacement instance on the current host
            replacement = _find_instance_on_host(
                managed_instances, current_host_id, current_instance_id
            )

            is_healthy = check_instance_healthy_sync(
                instance_data=replacement,
                alias=alias,
                gateway_aliases=gateway_aliases,
            )

            if is_healthy:
                # Find old instance to retire on this host
                old_instance = _find_old_instance(
                    managed_instances, current_host_id, target_source
                )
                if old_instance:
                    old_id = old_instance.get("instance_id") or old_instance.get("id")
                    new_progress = dict(progress_data)
                    new_progress["phase"] = StrategyPhase.RETIRING_OLD
                    new_progress["message"] = (
                        f"Retiring old replica on {current_host_id}"
                    )
                    return (
                        {
                            "type": "stop",
                            "host_id": current_host_id,
                            "instance_id": old_id,
                            "reason": "Rolling: retiring old replica",
                        },
                        new_progress,
                    )
                else:
                    # No old instance — this was a pure scale-up add on a
                    # new host.  Advance to next slot.
                    return _advance_to_next_rolling(
                        progress_data,
                        managed_instances,
                        candidates,
                        desired_replicas,
                        alias,
                        target_source,
                    )

            # Check timeout
            elapsed = health_gate_started_at
            if elapsed > health_gate_timeout_s:
                return _strategy_held(
                    progress_data,
                    (
                        f"Health gate timeout for replacement on "
                        f"{current_host_id} after {elapsed:.0f}s"
                    ),
                )

            # Still waiting — no action
            new_progress = dict(progress_data)
            new_progress["message"] = (
                f"Waiting for replacement on {current_host_id} "
                f"({elapsed:.0f}s / {health_gate_timeout_s:.0f}s)"
            )
            return {"type": "wait", "reason": "health gate"}, new_progress

        # ── PHASE: retiring_old ──────────────────────────────────
        if phase == StrategyPhase.RETIRING_OLD:
            # Check if old instance has been stopped (gone from managed)
            old_still_exists = _find_old_instance(
                managed_instances, current_host_id, target_source
            )
            if old_still_exists:
                # Keep waiting for the stop to take effect
                new_progress = dict(progress_data)
                new_progress["message"] = (
                    f"Waiting for old replica on {current_host_id} to stop"
                )
                return {"type": "wait", "reason": "awaiting stop"}, new_progress

            # Old replica gone → advance to next slot
            return _advance_to_next_rolling(
                progress_data,
                managed_instances,
                candidates,
                desired_replicas,
                alias,
                target_source,
            )

        # ── Unknown phase ────────────────────────────────────────
        logger.warning("Unknown rolling strategy phase: %s", phase)
        return None, None


def _advance_to_next_rolling(
    progress_data: dict[str, Any],
    managed_instances: list[dict[str, Any]],
    candidates: list[tuple[Any, Any]],
    desired_replicas: int,
    alias: str,
    target_source: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Advance to the next replacement host, or complete the strategy."""
    updated = _count_updated(managed_instances, target_source)

    # Determine total steps from the step string
    step_str = progress_data.get("step", "0/1")
    try:
        total_steps = int(step_str.split("/")[-1])
    except (ValueError, IndexError):
        total_steps = 1

    if updated >= desired_replicas:
        # All replicas on target source
        return None, None  # strategy done

    pending = list(progress_data.get("pending_hosts", []))
    if not pending:
        # No more hosts to use — check if we still need more
        remaining = desired_replicas - updated
        if remaining <= 0:
            return None, None
        # Try to find additional candidates
        current_host_ids = {inst.get("_host_id") for inst in managed_instances}
        extra_candidates = [h.id for h, _ in candidates if h.id not in current_host_ids]
        pending = extra_candidates[:remaining]

    if not pending:
        # No hosts available but still need replicas → strategy can't complete
        return None, None

    next_host = pending[0]
    step_num = updated + 1  # next step after the one that just completed
    new_progress = dict(progress_data)
    new_progress["phase"] = StrategyPhase.CREATING_REPLACEMENT
    new_progress["step"] = f"{step_num}/{total_steps}"
    new_progress["updated"] = updated
    new_progress["in_progress"] = 1
    new_progress["current_host_id"] = next_host
    new_progress["current_instance_id"] = None
    new_progress["pending_hosts"] = pending[1:]
    new_progress["message"] = f"Creating replacement on {next_host}"

    return (
        {
            "type": "create",
            "host_id": next_host,
            "reason": f"Rolling replacement step {step_num}/{total_steps}",
        },
        new_progress,
    )


# ── Immediate Strategy (§11.3) ──────────────────────────────────


class ImmediateStrategy:
    """Immediate update: stop all old replicas, then create all replacements.

    Implements deployment-intent.md §11.3.

    State machine:
      stopping_old: emit stop for each old replica → transition to creating
      creating_replacements: emit create for each replacement → done
    """

    @staticmethod
    def init(
        *,
        intent_id: str,
        alias: str,
        target_model_source: str,
        desired_replicas: int,
        managed_instances: list[dict[str, Any]],
        candidates: list[tuple[Any, Any]],
    ) -> dict[str, Any] | None:
        """Initialize an immediate strategy from current observed state."""
        now = datetime.now(timezone.utc).isoformat()

        updated = _count_updated(managed_instances, target_model_source)
        needed = desired_replicas - updated
        if needed <= 0:
            return None  # Already at desired state

        # Only initiate strategy when there are drifted instances that
        # need replacing.  Pure scale-up is handled by normal diff.
        if updated == len(managed_instances) and needed > 0:
            return None

        # All managed instances with old source need stopping
        drifted = [
            inst
            for inst in managed_instances
            if (inst.get("config", inst).get("model_source") != target_model_source)
        ]

        # Replacement host candidates
        current_host_ids = {inst.get("_host_id") for inst in managed_instances}
        replacement_hosts = [
            h.id for h, _ in candidates if h.id not in current_host_ids
        ]

        total_steps = max(len(drifted), needed)

        return {
            "strategy": "immediate",
            "target_model_source": target_model_source,
            "phase": StrategyPhase.STOPPING_OLD,
            "step": f"0/{total_steps}",
            "updated": updated,
            "in_progress": len(drifted),
            "failed": 0,
            "current_host_id": None,
            "current_instance_id": None,
            "pending_hosts": replacement_hosts[:needed],
            "failed_hosts": [],
            "started_at": now,
            "message": f"Stopping {len(drifted)} old replica(s)",
        }

    @staticmethod
    def continue_step(
        *,
        progress_data: dict[str, Any],
        intent_id: str,
        alias: str,
        desired_replicas: int,
        managed_instances: list[dict[str, Any]],
        candidates: list[tuple[Any, Any]],
        gateway_aliases: set[str],
        health_gate_started_at: float,
        health_gate_timeout_s: float,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Continue an immediate strategy from its current phase."""
        phase = progress_data.get("phase", "")
        target_source = progress_data.get("target_model_source", "")

        # ── PHASE: stopping_old ──────────────────────────────────
        if phase == StrategyPhase.STOPPING_OLD:
            # Find first old instance still running
            for inst in managed_instances:
                cfg = inst.get("config", inst)
                if cfg.get("model_source") != target_source:
                    inst_id = inst.get("instance_id") or inst.get("id")
                    host_id = inst.get("_host_id")
                    remaining = sum(
                        1
                        for m in managed_instances
                        if (m.get("config", m).get("model_source") != target_source)
                    )
                    new_progress = dict(progress_data)
                    new_progress["message"] = (
                        f"Stopping old replica on {host_id} " f"({remaining} remaining)"
                    )
                    return (
                        {
                            "type": "stop",
                            "host_id": host_id,
                            "instance_id": inst_id,
                            "reason": "Immediate: stopping old replica",
                        },
                        new_progress,
                    )

            # All old instances stopped → transition to creating
            updated = _count_updated(managed_instances, target_source)
            needed = desired_replicas - updated
            if needed <= 0:
                return None, None  # Nothing left to do

            # Refresh replacement hosts
            current_host_ids = {inst.get("_host_id") for inst in managed_instances}
            # Also include candidate hosts from the pending_hosts list
            # that were saved during init
            pending_from_init = list(progress_data.get("pending_hosts", []))
            # Plus new candidates not in current set
            new_candidates = [
                h.id
                for h, _ in candidates
                if h.id not in current_host_ids and h.id not in pending_from_init
            ]
            all_pending = pending_from_init + new_candidates

            if not all_pending and needed > 0:
                return _strategy_held(
                    progress_data,
                    f"No hosts available for {needed} replacement replicas",
                )

            new_progress = dict(progress_data)
            new_progress["phase"] = StrategyPhase.CREATING_REPLACEMENTS
            new_progress["pending_hosts"] = all_pending[:needed]
            new_progress["in_progress"] = needed
            new_progress["message"] = f"Creating {needed} replacement replica(s)"
            return {"type": "wait", "reason": "transitioning"}, new_progress

        # ── PHASE: creating_replacements ─────────────────────────
        if phase == StrategyPhase.CREATING_REPLACEMENTS:
            pending = list(progress_data.get("pending_hosts", []))
            if not pending:
                # All replacements dispatched — strategy done
                return None, None

            host_id = pending[0]
            remaining = len(pending)
            new_progress = dict(progress_data)
            new_progress["pending_hosts"] = pending[1:]
            new_progress["message"] = (
                f"Creating replacement on {host_id} ({remaining} remaining)"
            )
            new_progress["in_progress"] = remaining
            return (
                {
                    "type": "create",
                    "host_id": host_id,
                    "reason": "Immediate: creating replacement",
                },
                new_progress,
            )

        # ── Unknown phase ────────────────────────────────────────
        logger.warning("Unknown immediate strategy phase: %s", phase)
        return None, None


# ── Failure helpers ──────────────────────────────────────────────


def _strategy_held(
    progress_data: dict[str, Any],
    message: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Strategy is held — do not proceed, keep existing state.

    For rolling: old replicas keep running, intent becomes degraded
    but service remains available with a mix of old/new replicas.
    For immediate: replacements that came up stay running; shortfall
    is reported.
    """
    new_progress = dict(progress_data)
    new_progress["phase"] = StrategyPhase.FAILED
    new_progress["failed"] = new_progress.get("failed", 0) + 1
    new_progress["message"] = message
    new_progress["in_progress"] = 0
    return {"type": "wait", "reason": "strategy held"}, new_progress


def _strategy_failed(
    progress_data: dict[str, Any],
    message: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Strategy has permanently failed — clear and report."""
    new_progress = dict(progress_data)
    new_progress["phase"] = StrategyPhase.FAILED
    new_progress["message"] = message
    return None, new_progress


# ── Initiation helper ────────────────────────────────────────────


def should_initiate_strategy(
    *,
    intent: Any,
    managed_instances: list[dict[str, Any]],
) -> bool:
    """Check if a strategy should be initiated for this intent.

    Returns True when:
    - Any managed instance has a different model_source from the intent
    - AND the intent specifies a supported strategy ('rolling'/'immediate')
    """
    strategy = getattr(intent, "strategy", None)
    if strategy not in ("rolling", "immediate"):
        return False

    target_source = getattr(intent, "model_source", None)
    if not target_source:
        return False

    for inst in managed_instances:
        cfg = inst.get("config", inst)
        if cfg.get("model_source") != target_source:
            return True

    # Also trigger if observed < desired (scale-up) — handled by normal
    # diff, but if there's also source drift we want the strategy.
    return False


def initiate_strategy(
    *,
    intent: Any,
    managed_instances: list[dict[str, Any]],
    candidates: list[tuple[Any, Any]],
) -> dict[str, Any] | None:
    """Create initial strategy_progress for the intent's strategy.

    Returns strategy_progress dict or None if no strategy needed.
    """
    strategy_name = getattr(intent, "strategy", None)
    alias = getattr(intent, "alias", "")
    target_source = getattr(intent, "model_source", "")
    desired = getattr(intent, "replicas", 0)
    intent_id = getattr(intent, "id", "")

    if strategy_name == "rolling":
        return RollingStrategy.init(
            intent_id=intent_id,
            alias=alias,
            target_model_source=target_source,
            desired_replicas=desired,
            managed_instances=managed_instances,
            candidates=candidates,
        )
    elif strategy_name == "immediate":
        return ImmediateStrategy.init(
            intent_id=intent_id,
            alias=alias,
            target_model_source=target_source,
            desired_replicas=desired,
            managed_instances=managed_instances,
            candidates=candidates,
        )

    return None


def continue_strategy(
    *,
    progress_data: dict[str, Any],
    intent: Any,
    managed_instances: list[dict[str, Any]],
    candidates: list[tuple[Any, Any]],
    gateway_aliases: set[str],
    health_gate_started_at: float,
    health_gate_timeout_s: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Continue an in-flight strategy from its current phase.

    Returns (action_dict, new_progress) — same convention as
    RollingStrategy.continue_step / ImmediateStrategy.continue_step.
    """
    strategy_name = progress_data.get("strategy", "")
    alias = getattr(intent, "alias", "")
    desired = getattr(intent, "replicas", 0)
    intent_id = getattr(intent, "id", "")

    if strategy_name == "rolling":
        return RollingStrategy.continue_step(
            progress_data=progress_data,
            intent_id=intent_id,
            alias=alias,
            desired_replicas=desired,
            managed_instances=managed_instances,
            candidates=candidates,
            gateway_aliases=gateway_aliases,
            health_gate_started_at=health_gate_started_at,
            health_gate_timeout_s=health_gate_timeout_s,
        )
    elif strategy_name == "immediate":
        return ImmediateStrategy.continue_step(
            progress_data=progress_data,
            intent_id=intent_id,
            alias=alias,
            desired_replicas=desired,
            managed_instances=managed_instances,
            candidates=candidates,
            gateway_aliases=gateway_aliases,
            health_gate_started_at=health_gate_started_at,
            health_gate_timeout_s=health_gate_timeout_s,
        )

    logger.warning("Unknown strategy: %s", strategy_name)
    return None, None

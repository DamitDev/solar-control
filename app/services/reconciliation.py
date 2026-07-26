"""Intent reconciliation engine (S-041).

Level-triggered reconciliation loop that compares desired state
(intents) with observed state (instances + gateway registry) and
converges the cluster by creating, stopping, and migrating instances.

Design:
- Periodic tick (configurable interval) + event-driven wake-ups.
- Per-intent Redis lock prevents concurrent reconciliation from
  multiple Solar Control replicas.
- Reuses shared placement policy (app.services.placement) and
  migration orchestrator (app.services.migration).
- Stateless/restart-safe: recomputes managed(I) from observed state
  on every pass; never trusts memory alone.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.redis_state.connection import redis_client

logger = logging.getLogger(__name__)

# Per-intent lock key prefix and TTL.
# _LOCK_TTL auto-expires so a crashed reconciler doesn't block forever.
_LOCK_PREFIX = "solar:reconcile:lock:"
_LOCK_TTL = 30


# ── Action model ───────────────────────────────────────────────


class ActionType:
    CREATE = "create"
    STOP = "stop"
    REPLACE = "replace"
    RECREATE = "recreate"
    MIGRATE = "migrate"
    DISOWN = "disown"
    NOOP = "noop"


@dataclass
class Action:
    """A single reconciliation action to execute on a host."""

    type: str
    intent_id: str
    alias: str
    host_id: str | None = None
    host_name: str | None = None
    instance_id: str | None = None  # for stop / replace / recreate / migrate / disown
    target_host_id: str | None = None  # for migrate: where to move the instance
    target_host_name: str | None = None  # for migrate
    reason: str = ""
    priority: int = 0  # lower executes first (stops before creates)


# Exponential backoff bounds (seconds)
_BACKOFF_MIN_S = 10
_BACKOFF_MAX_S = 300


# ── Reconciler ─────────────────────────────────────────────────


class Reconciler:
    """Periodic + event-driven intent reconciliation engine."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()
        self._running = False
        # Per-intent exponential backoff: {intent_id: {"failures": N, "next_retry_at": iso}}
        self._backoff: dict[str, dict[str, Any]] = {}

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background reconciliation loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Reconciler started (interval=%.1fs)", settings.reconcile_interval_s
        )

    async def stop(self) -> None:
        """Stop the background reconciliation loop."""
        self._running = False
        self._wake_event.set()  # unblock any sleep
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Reconciler stopped")

    def wake(self) -> None:
        """Trigger an immediate reconciliation pass (event-driven)."""
        self._wake_event.set()

    # ── Backoff ────────────────────────────────────────────────

    def _backoff_clear(self, intent_id: str) -> None:
        """Clear backoff state for *intent_id* after a successful tick."""
        self._backoff.pop(intent_id, None)

    def _backoff_record_failure(self, intent_id: str) -> None:
        """Record a failure and set the next retry time with exponential backoff."""
        now = datetime.now(timezone.utc)
        entry = self._backoff.get(intent_id, {"failures": 0})
        entry["failures"] = entry.get("failures", 0) + 1
        delay = min(_BACKOFF_MIN_S * (2 ** (entry["failures"] - 1)), _BACKOFF_MAX_S)
        entry["next_retry_at"] = (
            datetime.fromtimestamp(now.timestamp() + delay, tz=timezone.utc)
        ).isoformat()
        self._backoff[intent_id] = entry
        logger.debug(
            "Backoff for intent %s: failures=%d delay=%.0fs",
            intent_id,
            entry["failures"],
            delay,
        )

    def _backoff_active(self, intent_id: str) -> bool:
        """Return True if backoff is active and retry should be skipped."""
        entry = self._backoff.get(intent_id)
        if not entry:
            return False
        next_retry = entry.get("next_retry_at")
        if not next_retry:
            return False
        try:
            retry_at = datetime.fromisoformat(next_retry)
            return datetime.now(timezone.utc) < retry_at
        except (ValueError, TypeError):
            return False

    # ── Main loop ──────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main loop: sleep → reconcile → repeat."""
        while self._running:
            try:
                await self._reconcile_all()
            except Exception:
                logger.exception("Reconciliation pass failed")

            # Wait for the next interval or an event-driven wake-up
            try:
                self._wake_event.clear()
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=settings.reconcile_interval_s,
                )
            except asyncio.TimeoutError:
                pass  # interval elapsed — normal

    async def _reconcile_all(self) -> None:
        """Fetch all active intents and reconcile each one."""
        from app.database.intents import intent_db

        intents = await intent_db.list_active_for_reconciliation()
        if not intents:
            return

        logger.debug("Reconciling %d active intent(s)", len(intents))
        for intent in intents:
            if not self._running:
                break

            # Acquire per-intent lock to avoid concurrent reconciliation
            lock_key = f"{_LOCK_PREFIX}{intent.id}"
            r = redis_client()
            acquired = await r.set(lock_key, "1", nx=True, ex=_LOCK_TTL)
            if not acquired:
                logger.debug("Intent %s locked by another replica, skipping", intent.id)
                continue
            try:
                await self._reconcile_one(intent)
            except Exception:
                logger.exception("Reconciliation failed for intent %s", intent.id)
            finally:
                await r.delete(lock_key)

    # ── Per-intent reconciliation ──────────────────────────────

    async def _reconcile_one(self, intent: Any) -> None:
        """Reconcile a single intent: observe → diff → act → update status.

        Implements deployment-intent.md §8.1 reconciliation loop.
        """
        # Check backoff before acting
        if self._backoff_active(intent.id):
            logger.debug("Intent %s in backoff, skipping", intent.id)
            return

        # 1. Observe
        observed = await self._observe(intent)

        # 2. Diff
        actions = self._diff(intent, observed)

        if not actions:
            # No actions needed — still update status to reflect current state
            await self._update_status(intent, observed)
            return

        # 3. Act — execute at most one action per tick for gradual convergence
        # Actions are sorted by priority (stops first, creates last)
        actions.sort(key=lambda a: a.priority)
        action = actions[0]
        logger.info(
            "Reconciling intent %s (%s): action=%s reason=%s",
            intent.id,
            intent.alias,
            action.type,
            action.reason,
        )

        last_error = None
        action_succeeded = False
        try:
            result = await self._act(intent, action)
            action_succeeded = True

            # If we created/migrated, re-observe for fresh state
            if action.type in (ActionType.CREATE, ActionType.MIGRATE) and result:
                await asyncio.sleep(0.5)
                observed = await self._observe(intent)
        except Exception as e:
            logger.error(
                "Action %s failed for intent %s: %s",
                action.type,
                intent.id,
                e,
            )
            last_error = {
                "code": type(e).__name__,
                "message": str(e),
                "host_id": action.host_id,
                "source_uri": intent.model_source,
                "at": datetime.now(timezone.utc).isoformat(),
            }

        # Update backoff state
        if action_succeeded and last_error is None:
            self._backoff_clear(intent.id)
        elif last_error is not None:
            self._backoff_record_failure(intent.id)

        # 4. Update status
        await self._update_status(intent, observed, last_error=last_error)

    # ── Observe ────────────────────────────────────────────────

    async def _observe(self, intent: Any) -> dict[str, Any]:
        """Collect observed state for *intent*.

        Returns a dict with:
            managed_instances: instances with managed_by='intent' and intent_id==intent.id
            alias_instances: ALL instances serving this alias (managed + manual)
            manual_conflicts: manual instances serving this alias
            hosts: list of Host models
            snapshots: dict[host_id, HostResourceSnapshot]
            gateway_aliases: set of aliases registered in gateway
            candidates: list of (Host, snapshot) pairs from shared placement policy
            displaceable_map: dict[host_id, list[dict]] of displaceable instances
        """
        from app.database.hosts import host_db
        from app.redis_state import host_store, registry_store
        from app.routes.management.resources import _fetch_host_resource_snapshot
        from app.services.placement import find_candidates, find_displaceable_instances

        alias = intent.alias

        # 1. Fetch all hosts and resource snapshots
        hosts = await host_db.get_all_hosts()
        snapshots_list = await asyncio.gather(
            *[_fetch_host_resource_snapshot(h) for h in hosts]
        )
        snapshots: dict[str, Any] = {s.host_id: s for s in snapshots_list}

        # 2. Collect all instances for this alias across all hosts
        managed_instances: list[dict[str, Any]] = []
        alias_instances: list[dict[str, Any]] = []
        manual_conflicts: list[dict[str, Any]] = []
        for host in hosts:
            instances = await host_store.get_host_instances(host.id)
            for inst in instances:
                cfg = inst.get("config", inst)
                inst_alias = cfg.get("alias") or inst.get("alias")
                if inst_alias != alias:
                    continue
                # Annotate with host context
                inst["_host_id"] = host.id
                inst["_host_name"] = host.name
                alias_instances.append(inst)

                # Owned by this intent?
                managed_by = cfg.get("managed_by") or inst.get("managed_by")
                intent_id = cfg.get("intent_id") or inst.get("intent_id")
                if managed_by == "intent" and intent_id == intent.id:
                    managed_instances.append(inst)
                elif managed_by != "intent" or intent_id != intent.id:
                    # Manual instance or owned by a different intent
                    manual_conflicts.append(inst)

        # 3. Gateway registry — which aliases are registered?
        gateway_aliases: set[str] = set()
        try:
            registry = await registry_store.get_registry()
            if isinstance(registry, dict):
                gateway_aliases = set(registry.keys())
        except Exception:
            logger.warning("Failed to read gateway registry", exc_info=True)

        # 4. Compute placement candidates using shared policy (§8.4)
        placement = intent.placement
        resources = intent.resources
        req_vram = (
            float(resources.vram_gb or 0) if hasattr(resources, "vram_gb") else 0.0
        )
        req_ram = (
            float(resources.ram_gb)
            if hasattr(resources, "ram_gb") and resources.ram_gb
            else None
        )
        req_roles = list(placement.roles) if placement.roles else ["inference"]
        req_gpu = (
            placement.gpu_type
            if hasattr(placement, "gpu_type") and placement.gpu_type
            else None
        )
        req_allow = (
            list(placement.host_allow)
            if hasattr(placement, "host_allow") and placement.host_allow
            else None
        )
        req_deny = (
            list(placement.host_deny)
            if hasattr(placement, "host_deny") and placement.host_deny
            else None
        )

        candidates = await find_candidates(
            hosts,
            snapshots,
            roles=req_roles,
            gpu_type=req_gpu,
            host_allow=req_allow,
            host_deny=req_deny,
            vram_gb=req_vram,
            ram_gb=req_ram,
            exclude_alias=alias,
        )

        # 5. Compute displaceable instances per host (for displacement evaluation)
        displaceable_map: dict[str, list[dict[str, Any]]] = {}
        intent_priority = intent.priority
        candidate_host_ids = {h.id for h, _ in candidates}
        # Pre-collect hosts with active training jobs (non-displaceable per §8.5)
        hosts_with_active_jobs: set[str] = set()
        try:
            from app.database.jobs import job_db
            from app.models.job import JobStatus

            for host in hosts:
                jobs = await job_db.get_jobs_by_host(host.id)
                if any(
                    j.status in (JobStatus.PENDING, JobStatus.RUNNING) for j in jobs
                ):
                    hosts_with_active_jobs.add(host.id)
        except Exception:
            logger.warning(
                "Failed to query active training jobs for displacement pre-filter",
                exc_info=True,
            )
        for host in hosts:
            if host.id in candidate_host_ids or host.status != "online":
                continue
            # Skip hosts with active training jobs (§8.5)
            if host.id in hosts_with_active_jobs:
                continue
            # Check if host has right roles/GPU (basic pre-filter)
            host_roles = host.roles or []
            if not all(r in host_roles for r in req_roles):
                continue
            if req_gpu and host.gpu_type != req_gpu:
                continue
            displaced = await find_displaceable_instances(
                host.id, intent_priority, preserve_alias=alias
            )
            if displaced:
                displaceable_map[host.id] = displaced

        return {
            "managed_instances": managed_instances,
            "alias_instances": alias_instances,
            "manual_conflicts": manual_conflicts,
            "hosts": hosts,
            "snapshots": snapshots,
            "gateway_aliases": gateway_aliases,
            "candidates": candidates,
            "displaceable_map": displaceable_map,
        }

    # ── Diff ───────────────────────────────────────────────────

    def _diff(
        self,
        intent: Any,
        observed: dict[str, Any],
    ) -> list[Action]:
        """Compare desired vs observed state and produce actions.

        Implements deployment-intent.md §8.2 diff and actions table.
        Uses shared placement policy candidates and evaluates
        priority-aware displacement when capacity is insufficient.
        """
        desired = intent.replicas
        managed = observed["managed_instances"]
        is_orphan = _intent_orphan(intent)

        observed_count = len(managed)
        actions: list[Action] = []

        # ── Deleting intent ──────────────────────────────────────
        if _intent_phase(intent) == "deleting":
            for inst in managed:
                inst_id = inst.get("instance_id") or inst.get("id")
                if not inst_id:
                    continue
                if is_orphan:
                    actions.append(
                        Action(
                            type=ActionType.DISOWN,
                            intent_id=intent.id,
                            alias=intent.alias,
                            host_id=inst.get("_host_id"),
                            instance_id=inst_id,
                            reason="Intent deleted (orphan)",
                            priority=1,
                        )
                    )
                else:
                    actions.append(
                        Action(
                            type=ActionType.STOP,
                            intent_id=intent.id,
                            alias=intent.alias,
                            host_id=inst.get("_host_id"),
                            instance_id=inst_id,
                            reason="Intent deleted",
                            priority=0,
                        )
                    )
            return actions

        # ── replicas == 0 → stop all managed instances ───────────
        if desired == 0:
            for inst in managed:
                inst_id = inst.get("instance_id") or inst.get("id")
                if inst_id:
                    actions.append(
                        Action(
                            type=ActionType.STOP,
                            intent_id=intent.id,
                            alias=intent.alias,
                            host_id=inst.get("_host_id"),
                            instance_id=inst_id,
                            reason="replicas=0",
                            priority=0,
                        )
                    )
            return actions

        # ── Check for drift (model_source + backend config) ──────
        for inst in managed:
            cfg = inst.get("config", inst)
            inst_source = cfg.get("model_source") or inst.get("model_source")
            inst_id = inst.get("instance_id") or inst.get("id")
            has_source_drift = inst_source and inst_source != intent.model_source
            has_backend_drift = _detect_backend_drift(intent, cfg)

            if has_source_drift:
                actions.append(
                    Action(
                        type=ActionType.REPLACE,
                        intent_id=intent.id,
                        alias=intent.alias,
                        host_id=inst.get("_host_id"),
                        instance_id=inst_id,
                        reason=(
                            f"model_source drift: {inst_source} → "
                            f"{intent.model_source}"
                        ),
                        priority=20,
                    )
                )
            elif has_backend_drift and not has_source_drift:
                actions.append(
                    Action(
                        type=ActionType.REPLACE,
                        intent_id=intent.id,
                        alias=intent.alias,
                        host_id=inst.get("_host_id"),
                        instance_id=inst_id,
                        reason="backend config drift",
                        priority=20,
                    )
                )

            # Check if instance failed/stopped unexpectedly
            status = inst.get("status") or inst.get("state", "")
            if status in ("failed", "stopped", "error"):
                if not any(
                    a.instance_id == inst_id and a.type == ActionType.REPLACE
                    for a in actions
                ):
                    actions.append(
                        Action(
                            type=ActionType.RECREATE,
                            intent_id=intent.id,
                            alias=intent.alias,
                            host_id=inst.get("_host_id"),
                            instance_id=inst_id,
                            reason=f"Instance {status}, recreating",
                            priority=15,
                        )
                    )

        # ── Observed < Desired → CREATE (placement policy) ───────
        shortfall = desired - observed_count
        if shortfall > 0:
            candidates = observed.get("candidates", [])
            for i in range(min(shortfall, len(candidates))):
                host, _snap = candidates[i]
                actions.append(
                    Action(
                        type=ActionType.CREATE,
                        intent_id=intent.id,
                        alias=intent.alias,
                        host_id=host.id,
                        host_name=host.name,
                        reason=f"shortfall {i + 1}/{shortfall}",
                        priority=50,
                    )
                )

            # If still short, evaluate priority-aware displacement (§8.5)
            remaining = shortfall - min(shortfall, len(candidates))
            if remaining > 0:
                displaceable_map = observed.get("displaceable_map", {})
                for host_id, displaceable_list in displaceable_map.items():
                    if remaining <= 0:
                        break
                    if not displaceable_list:
                        continue
                    inst = displaceable_list[0]
                    inst_id = inst.get("instance_id") or inst.get("id", "")
                    inst_alias = inst.get("config", inst).get("alias") or inst.get(
                        "alias", ""
                    )
                    if not inst_id:
                        continue
                    actions.append(
                        Action(
                            type=ActionType.MIGRATE,
                            intent_id=intent.id,
                            alias=inst_alias,
                            host_id=host_id,
                            instance_id=inst_id,
                            reason=(
                                f"Displacing {inst_alias} "
                                f"({inst.get('_priority', '?')}) "
                                f"to free capacity for {intent.alias} "
                                f"({intent.priority})"
                            ),
                            priority=25,
                        )
                    )
                    remaining -= 1

        # ── Observed > Desired → STOP surplus ────────────────────
        surplus = observed_count - desired
        if surplus > 0:
            # Sort per §8.2: unhealthy/failed first (oldest→newest within
            # that group), then healthy instances most-recently-created
            # first so long-lived replicas survive.
            unhealthy_insts: list[dict[str, Any]] = []
            healthy_insts: list[dict[str, Any]] = []
            for inst in managed:
                status = inst.get("status") or inst.get("state", "")
                if status in ("failed", "stopped", "error"):
                    unhealthy_insts.append(inst)
                else:
                    healthy_insts.append(inst)
            unhealthy_insts.sort(key=lambda i: i.get("created_at") or "0")
            healthy_insts.sort(key=lambda i: i.get("created_at") or "0", reverse=True)
            to_stop = (unhealthy_insts + healthy_insts)[:surplus]
            for inst in to_stop:
                inst_id = inst.get("instance_id") or inst.get("id")
                if inst_id:
                    actions.append(
                        Action(
                            type=ActionType.STOP,
                            intent_id=intent.id,
                            alias=intent.alias,
                            host_id=inst.get("_host_id"),
                            instance_id=inst_id,
                            reason="surplus replica",
                            priority=0,
                        )
                    )

        # Sort by priority so stops/disowns execute before migrates/creates
        actions.sort(key=lambda a: a.priority)
        return actions

    # ── Act ────────────────────────────────────────────────────

    async def _act(
        self,
        intent: Any,
        action: Action,
    ) -> dict[str, Any] | None:
        """Execute one reconciliation action via Solar Host primitives.

        Returns the created instance dict on create, None otherwise.
        """
        from app.database.hosts import host_db
        from app.services.migration import (
            create_instance_on_host,
            stop_source_instance,
        )

        if action.type == ActionType.NOOP:
            return None

        if action.type == ActionType.STOP:
            if not action.host_id or not action.instance_id:
                return None
            host = await host_db.get_host(action.host_id)
            if host is None:
                logger.warning("Host %s not found for stop action", action.host_id)
                return None
            logger.info(
                "Stopping instance %s on %s (reason: %s)",
                action.instance_id,
                host.name,
                action.reason,
            )
            await stop_source_instance(host, action.instance_id)
            # Delete the instance so the reconciler stops observing it.
            # Without this, stopped instances persist and observed_replicas
            # can never reach 0 for DELETE / scale-to-zero flows.
            try:
                await self._delete_instance(host, action.instance_id)
            except Exception:
                logger.warning(
                    "Failed to delete instance %s on %s after stop",
                    action.instance_id,
                    host.name,
                    exc_info=True,
                )
            return None

        if action.type == ActionType.DISOWN:
            # Clear ownership markers from the instance in Redis so the
            # reconciler stops tracking it.  The underlying Solar Host
            # instance config retains the markers (there is no host-side
            # PATCH for running instances), but the stale reference is
            # harmless: the intent is being deleted and will not be
            # reconciled again.
            if not action.host_id or not action.instance_id:
                return None
            from app.redis_state import host_store as _hs

            instances = await _hs.get_host_instances(action.host_id)
            found = False
            for inst in instances:
                iid = inst.get("instance_id") or inst.get("id")
                if iid == action.instance_id:
                    cfg = inst.get("config", inst)
                    if isinstance(cfg, dict):
                        cfg.pop("managed_by", None)
                        cfg.pop("intent_id", None)
                        inst["config"] = cfg
                    inst.pop("managed_by", None)
                    inst.pop("intent_id", None)
                    found = True
                    break
            if found:
                await _hs.set_host_instances(action.host_id, instances)
            logger.info(
                "Disowned instance %s on host %s (reason: %s)",
                action.instance_id,
                action.host_id,
                action.reason,
            )
            return None

        if action.type == ActionType.CREATE:
            if not action.host_id:
                return None
            host = await host_db.get_host(action.host_id)
            if host is None:
                logger.warning("Host %s not found for create action", action.host_id)
                return None

            instance_config = self._build_instance_config(intent, host)

            logger.info(
                "Creating instance for alias=%s on %s (reason: %s)",
                intent.alias,
                host.name,
                action.reason,
            )
            result = await create_instance_on_host(host, instance_config)
            # The host wraps the response in {"instance": {...}};
            # extract the instance and start it (host creates in stopped state).
            created = result.get("instance", result)
            instance_id = created.get("id") or created.get("instance_id")
            if instance_id:
                logger.info("Starting instance %s on %s", instance_id, host.name)
                await self._start_instance(host, instance_id)
            return result

        if action.type == ActionType.REPLACE:
            # Replace = stop old + create new on next tick
            if action.instance_id and action.host_id:
                host = await host_db.get_host(action.host_id)
                if host:
                    logger.info(
                        "Stopping drifted instance %s on %s for replacement",
                        action.instance_id,
                        host.name,
                    )
                    await stop_source_instance(host, action.instance_id)
            return None

        if action.type == ActionType.RECREATE:
            # Recreate = stop failed instance, next tick creates replacement
            if action.instance_id and action.host_id:
                host = await host_db.get_host(action.host_id)
                if host:
                    logger.info(
                        "Stopping failed instance %s on %s for recreation",
                        action.instance_id,
                        host.name,
                    )
                    await stop_source_instance(host, action.instance_id)
            return None

        if action.type == ActionType.MIGRATE:
            # Migrate = use S-037 to move instance off this host, freeing capacity
            if not action.instance_id or not action.host_id:
                return None
            from app.services.placement import find_candidates
            from app.services.migration import execute_migration
            from app.routes.management.resources import _fetch_host_resource_snapshot

            # Look up the source host to inherit its GPU type and roles as
            # placement constraints for the migration target (§8.5: move to
            # "another eligible host" implies matching capabilities).
            source_host = await host_db.get_host(action.host_id)
            host_roles = (
                source_host.roles
                if source_host and source_host.roles
                else ["inference"]
            )
            host_gpu = source_host.gpu_type if source_host else None

            # Select a target host using placement policy
            all_hosts = await host_db.get_all_hosts()
            snapshots_list = await asyncio.gather(
                *[_fetch_host_resource_snapshot(h) for h in all_hosts]
            )
            snapshots_map = {s.host_id: s for s in snapshots_list}

            target_candidates = await find_candidates(
                all_hosts,
                snapshots_map,
                roles=host_roles,
                gpu_type=host_gpu,
                vram_gb=0.0,
                exclude_alias=action.alias,
            )
            # Exclude the source host from candidates
            target_candidates = [
                (h, s) for h, s in target_candidates if h.id != action.host_id
            ]

            if not target_candidates:
                logger.warning(
                    "No migration target found for instance %s (alias=%s)",
                    action.instance_id,
                    action.alias,
                )
                return None

            target_host, _tsnap = target_candidates[0]
            logger.info(
                "Migrating instance %s (%s) from %s to %s (reason: %s)",
                action.instance_id,
                action.alias,
                action.host_id,
                target_host.name,
                action.reason,
            )
            try:
                result = await execute_migration(
                    instance_id=action.instance_id,
                    source_host_id=action.host_id,
                    target_host_id=target_host.id,
                    allow_production=False,
                )
                return {"migration_id": result.migration_id, "status": result.status}
            except Exception:
                logger.exception(
                    "Migration failed for instance %s: %s → %s",
                    action.instance_id,
                    action.host_id,
                    target_host.id,
                )
                raise

        return None

    # ── Start / Delete instance helpers ────────────────────────

    async def _start_instance(self, host: Any, instance_id: str) -> None:
        """Start a stopped instance on *host* via POST /instances/{id}/start."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{host.url.rstrip('/')}/instances/{instance_id}/start"
                headers = {"X-API-Key": host.api_key}
                async with session.post(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            "Failed to start instance %s on %s: HTTP %s %s",
                            instance_id,
                            host.name,
                            resp.status,
                            text,
                        )
        except Exception as e:
            logger.warning(
                "Failed to start instance %s on %s: %s",
                instance_id,
                host.name,
                e,
            )

    async def _delete_instance(self, host: Any, instance_id: str) -> None:
        """Delete an instance from *host* via DELETE /instances/{id}."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{host.url.rstrip('/')}/instances/{instance_id}"
                headers = {"X-API-Key": host.api_key}
                async with session.delete(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status not in (200, 204, 404):
                        text = await resp.text()
                        logger.warning(
                            "Failed to delete instance %s on %s: HTTP %s %s",
                            instance_id,
                            host.name,
                            resp.status,
                            text,
                        )
        except Exception as e:
            logger.warning(
                "Failed to delete instance %s on %s: %s",
                instance_id,
                host.name,
                e,
            )

    # ── Build instance config ──────────────────────────────────

    def _build_instance_config(self, intent: Any, host: Any) -> dict[str, Any]:
        """Compose a Solar Host InstanceConfig from the intent.

        Implements deployment-intent.md §6 mapping: alias, model_source,
        priority, managed_by, intent_id, plus backend runtime params.

        The host expects ``managed_by``, ``intent_id``, and ``priority``
        at the TOP LEVEL (outside ``config``), matching the Instance model.
        """
        config: dict[str, Any] = {
            "backend_type": intent.backend.get("backend_type", "llamacpp"),
            "alias": intent.alias,
            "model_source": intent.model_source,
        }

        # Copy backend runtime params, excluding backend_type (already set)
        for key, value in intent.backend.items():
            if key == "backend_type":
                continue
            config[key] = value

        return {
            "config": config,
            "managed_by": "intent",
            "intent_id": intent.id,
            "priority": intent.priority,
        }

    # ── Update status ──────────────────────────────────────────

    async def _update_status(
        self,
        intent: Any,
        observed: dict[str, Any],
        last_error: dict[str, Any] | None = None,
    ) -> None:
        """Compute and persist the intent status after reconciliation.

        Implements deployment-intent.md §10.2 status fields.
        """
        from app.database.intents import intent_db
        from app.models.intent import (
            Condition,
            IntentPhase,
            LastError,
            ReconcileState,
            ReplicaEntry,
        )

        managed = observed["managed_instances"]
        gateway_aliases = observed["gateway_aliases"]

        observed_count = len(managed)
        desired = intent.replicas
        alias = intent.alias

        # Count ready replicas (running AND in gateway registry)
        ready_count = 0
        updated_count = 0
        replica_set: list[dict[str, Any]] = []

        for inst in managed:
            cfg = inst.get("config", inst)
            status = inst.get("status") or inst.get("state", "")
            inst_id = inst.get("instance_id") or inst.get("id")
            inst_source = cfg.get("model_source") or inst.get("model_source")
            healthy = status == "running" and alias in gateway_aliases
            on_target_source = inst_source == intent.model_source

            if healthy:
                ready_count += 1
                if on_target_source:
                    updated_count += 1

            replica_set.append(
                ReplicaEntry(
                    host_id=inst.get("_host_id"),
                    host_name=inst.get("_host_name"),
                    instance_id=inst_id,
                    state=status,
                    model_source=inst_source,
                    healthy=healthy,
                    message=None,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ).model_dump()
            )

        # Determine phase
        current_phase = intent.status.phase.value
        if current_phase == "deleting":
            if observed_count == 0:
                phase = IntentPhase.DELETED
            else:
                phase = IntentPhase.DELETING
        elif desired == 0 and observed_count == 0:
            phase = IntentPhase.READY
        elif ready_count == desired and desired > 0:
            phase = IntentPhase.READY
        elif ready_count == 0 and desired > 0 and observed_count == 0:
            phase = IntentPhase.RECONCILING
        elif ready_count > 0:
            phase = IntentPhase.DEGRADED
        elif ready_count == 0 and (observed_count > 0 or desired > 0):
            phase = IntentPhase.FAILED
        else:
            phase = IntentPhase.RECONCILING

        # Build conditions
        conditions: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        if ready_count >= 1:
            conditions.append(
                Condition(
                    type="Available",
                    status=True,
                    reason="MinimumReplicasAvailable",
                    message=f"{ready_count}/{desired} ready",
                    last_transition=now_iso,
                ).model_dump()
            )
        if phase in (IntentPhase.RECONCILING, IntentPhase.DEGRADED):
            conditions.append(
                Condition(
                    type="Progressing",
                    status=True,
                    reason="Reconciling",
                    message="Reconciliation in progress",
                    last_transition=now_iso,
                ).model_dump()
            )

        # Conflict condition for manual instances (§5.3)
        manual_conflicts = observed.get("manual_conflicts", [])
        if manual_conflicts:
            conflict_hosts = sorted(
                {
                    m.get("_host_name") or m.get("_host_id", "?")
                    for m in manual_conflicts
                }
            )
            conditions.append(
                Condition(
                    type="Conflict",
                    status=True,
                    reason="ManualInstanceConflict",
                    message=(
                        f"Manual instance(s) serving '{alias}' on host(s): "
                        f"{', '.join(conflict_hosts[:5])}"
                        f"{'...' if len(conflict_hosts) > 5 else ''}"
                    ),
                    last_transition=now_iso,
                ).model_dump()
            )

        # Build last_error
        last_error_model = None
        if last_error:
            last_error_model = LastError(
                code=last_error.get("code", "unknown"),
                message=last_error.get("message", ""),
                host_id=last_error.get("host_id"),
                source_uri=last_error.get("source_uri"),
                at=last_error.get("at", now_iso),
            )

        # Build status_json
        # Shortfall = desired - placeable (accounting for displacement candidates)
        candidates = observed.get("candidates", [])
        displaceable_map = observed.get("displaceable_map", {})
        placeable = len(candidates) + len(displaceable_map)
        structural_shortfall = max(0, desired - observed_count - placeable)

        status_json = {
            "observed_replicas": observed_count,
            "ready_replicas": ready_count,
            "updated_replicas": updated_count,
            "available": ready_count >= 1,
            "shortfall": structural_shortfall,
            "replica_set": replica_set,
            "conditions": conditions,
            "strategy_progress": None,
            "last_error": last_error_model.model_dump() if last_error_model else None,
        }

        # Determine reconcile state
        if last_error:
            reconcile = ReconcileState.FAILED
        elif phase in (IntentPhase.READY, IntentPhase.DEGRADED):
            reconcile = ReconcileState.SUCCEEDED
        else:
            reconcile = ReconcileState.IN_PROGRESS

        now = datetime.now(timezone.utc)

        # Set ready_at when first reaching ready
        ready_at = None
        if phase == IntentPhase.READY:
            ready_at = (
                intent.status.ready_at if intent.status.ready_at else now.isoformat()
            )

        await intent_db.update_status(
            intent.id,
            phase=phase.value,
            reconcile=reconcile.value,
            status_json=status_json,
            last_reconciled_at=now,
            ready_at=ready_at,
        )

        # ── Emit Socket.IO events for live WebUI updates (§10.4) ──
        try:
            from app.socketio_app import sio

            # Fetch the updated intent so the event carries the full record
            updated = await intent_db.get_intent(intent.id)
            if updated:
                await sio.emit(
                    "intent_update",
                    updated.model_dump(),
                    namespace="/webui",
                )
                if phase == IntentPhase.DELETED:
                    await sio.emit(
                        "intent_removed",
                        {"id": intent.id, "alias": intent.alias},
                        namespace="/webui",
                    )
        except Exception:
            logger.warning(
                "Failed to emit intent_update event for %s", intent.id, exc_info=True
            )


# ── Helpers ────────────────────────────────────────────────────


def _intent_phase(intent: Any) -> str:
    """Safely extract the intent phase as a string."""
    try:
        return intent.status.phase.value
    except (AttributeError, TypeError):
        return str(getattr(intent, "phase", "pending"))


def _intent_orphan(intent: Any) -> bool:
    """Check if the intent was deleted with orphan=true."""
    try:
        return intent.metadata.get("orphan") == "true"
    except (AttributeError, TypeError):
        return False


def _detect_backend_drift(intent: Any, instance_config: dict[str, Any]) -> bool:
    """Check if the instance's backend config has drifted from the intent.

    Compares the intent's backend fields (excluding identity/server-derived
    fields) against the instance config for the same keys.
    """
    intent_backend = intent.backend if isinstance(intent.backend, dict) else {}
    _skip_keys = {
        "backend_type",
        "alias",
        "model_source",
        "host",
        "port",
        "api_key",
        "managed_by",
        "intent_id",
        "model",
        "model_id",
    }
    for key, value in intent_backend.items():
        if key in _skip_keys:
            continue
        inst_value = instance_config.get(key)
        if inst_value != value:
            return True
    return False


# Singleton
reconciler = Reconciler()

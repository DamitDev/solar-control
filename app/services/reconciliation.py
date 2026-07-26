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
    NOOP = "noop"


@dataclass
class Action:
    """A single reconciliation action to execute on a host."""

    type: str
    intent_id: str
    alias: str
    host_id: str | None = None
    host_name: str | None = None
    instance_id: str | None = None  # for stop / replace / recreate
    reason: str = ""
    priority: int = 0  # lower executes first (stops before creates)


# ── Reconciler ─────────────────────────────────────────────────


class Reconciler:
    """Periodic + event-driven intent reconciliation engine."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()
        self._running = False

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
                logger.debug(
                    "Intent %s locked by another replica, skipping", intent.id
                )
                continue
            try:
                await self._reconcile_one(intent)
            except Exception:
                logger.exception(
                    "Reconciliation failed for intent %s", intent.id
                )
            finally:
                await r.delete(lock_key)

    # ── Per-intent reconciliation ──────────────────────────────

    async def _reconcile_one(self, intent: Any) -> None:
        """Reconcile a single intent: observe → diff → act → update status.

        Implements deployment-intent.md §8.1 reconciliation loop.
        """
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
        try:
            result = await self._act(intent, action)

            # If we created an instance, re-observe for fresh state
            if action.type == ActionType.CREATE and result:
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

        # 4. Update status
        await self._update_status(intent, observed, last_error=last_error)

    # ── Observe ────────────────────────────────────────────────

    async def _observe(self, intent: Any) -> dict[str, Any]:
        """Collect observed state for *intent*.

        Returns a dict with:
            managed_instances: instances with managed_by='intent' and intent_id==intent.id
            alias_instances: ALL instances serving this alias (managed + manual)
            hosts: list of Host models
            snapshots: dict[host_id, HostResourceSnapshot]
            gateway_aliases: set of aliases registered in gateway
        """
        from app.database.hosts import host_db
        from app.redis_state import host_store, registry_store
        from app.routes.management.resources import _fetch_host_resource_snapshot

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

        # 3. Gateway registry — which aliases are registered?
        gateway_aliases: set[str] = set()
        try:
            registry = await registry_store.get_registry()
            if isinstance(registry, dict):
                gateway_aliases = set(registry.keys())
        except Exception:
            logger.warning("Failed to read gateway registry", exc_info=True)

        return {
            "managed_instances": managed_instances,
            "alias_instances": alias_instances,
            "hosts": hosts,
            "snapshots": snapshots,
            "gateway_aliases": gateway_aliases,
        }

    # ── Diff ───────────────────────────────────────────────────

    def _diff(
        self,
        intent: Any,
        observed: dict[str, Any],
    ) -> list[Action]:
        """Compare desired vs observed state and produce actions.

        Implements deployment-intent.md §8.2 diff and actions table.
        """
        desired = intent.replicas
        managed = observed["managed_instances"]
        alias_instances = observed["alias_instances"]
        hosts = observed["hosts"]

        observed_count = len(managed)
        actions: list[Action] = []

        # ── Deleting intent: stop all managed instances ──────────
        if intent.status.phase.value == "deleting":
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

        # ── Check for drift (model_source change) ────────────────
        for inst in managed:
            cfg = inst.get("config", inst)
            inst_source = cfg.get("model_source") or inst.get("model_source")
            inst_id = inst.get("instance_id") or inst.get("id")

            if inst_source and inst_source != intent.model_source:
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

        # ── Observed < Desired → CREATE ──────────────────────────
        shortfall = desired - observed_count
        if shortfall > 0:
            # Build set of hosts already serving this alias (one-replica-per-host)
            occupied_host_ids: set[str] = set()
            for inst in alias_instances:
                hid = inst.get("_host_id")
                if hid:
                    occupied_host_ids.add(hid)

            eligible = [
                h
                for h in hosts
                if h.id not in occupied_host_ids and h.status == "online"
            ]

            for i in range(min(shortfall, len(eligible))):
                host = eligible[i]
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

        # ── Observed > Desired → STOP surplus ────────────────────
        surplus = observed_count - desired
        if surplus > 0:
            # Sort: unhealthy first, then newest (by created_at if available)
            def _stop_sort_key(inst: dict[str, Any]) -> tuple[int, str]:
                status = inst.get("status") or inst.get("state", "")
                unhealthy = 0 if status in ("failed", "stopped", "error") else 1
                created = inst.get("created_at") or "0"
                return (unhealthy, created)

            to_stop = sorted(managed, key=_stop_sort_key)[:surplus]
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
                logger.warning(
                    "Host %s not found for stop action", action.host_id
                )
                return None
            logger.info(
                "Stopping instance %s on %s (reason: %s)",
                action.instance_id,
                host.name,
                action.reason,
            )
            await stop_source_instance(host, action.instance_id)
            return None

        if action.type == ActionType.CREATE:
            if not action.host_id:
                return None
            host = await host_db.get_host(action.host_id)
            if host is None:
                logger.warning(
                    "Host %s not found for create action", action.host_id
                )
                return None

            instance_config = self._build_instance_config(intent, host)

            logger.info(
                "Creating instance for alias=%s on %s (reason: %s)",
                intent.alias,
                host.name,
                action.reason,
            )
            result = await create_instance_on_host(host, instance_config)
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

        return None

    # ── Build instance config ──────────────────────────────────

    def _build_instance_config(self, intent: Any, host: Any) -> dict[str, Any]:
        """Compose a Solar Host InstanceConfig from the intent.

        Implements deployment-intent.md §6 mapping: alias, model_source,
        priority, managed_by, intent_id, plus backend runtime params.
        """
        config: dict[str, Any] = {
            "backend_type": intent.backend.get("backend_type", "llamacpp"),
            "alias": intent.alias,
            "model_source": intent.model_source,
            "priority": intent.priority,
            "managed_by": "intent",
            "intent_id": intent.id,
        }

        # Copy backend runtime params, excluding backend_type (already set)
        for key, value in intent.backend.items():
            if key == "backend_type":
                continue
            config[key] = value

        return {"config": config}

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
        status_json = {
            "observed_replicas": observed_count,
            "ready_replicas": ready_count,
            "updated_replicas": updated_count,
            "available": ready_count >= 1,
            "shortfall": max(0, desired - observed_count),
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


# Singleton
reconciler = Reconciler()

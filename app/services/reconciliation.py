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
        """Reconcile a single intent.  To be implemented in later tasks."""
        pass

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
        raise NotImplementedError("_diff will be implemented in Task 6")

    # ── Act ────────────────────────────────────────────────────

    async def _act(
        self,
        intent: Any,
        action: Action,
    ) -> dict[str, Any] | None:
        """Execute one reconciliation action via Solar Host primitives."""
        raise NotImplementedError("_act will be implemented in Task 7")

    # ── Build instance config ──────────────────────────────────

    def _build_instance_config(self, intent: Any, host: Any) -> dict[str, Any]:
        """Compose a Solar Host InstanceConfig from the intent.

        Implements deployment-intent.md §6 mapping.
        """
        raise NotImplementedError(
            "_build_instance_config will be implemented in Task 7"
        )

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
        raise NotImplementedError("_update_status will be implemented in Task 8")


# Singleton
reconciler = Reconciler()

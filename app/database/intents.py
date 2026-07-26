"""PostgreSQL-backed intent CRUD operations (S-040)."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.models.intent import (
    IntentPhase,
    IntentResponse,
    IntentStatus,
    PlacementConstraints,
    ReconcileState,
    ResourceRequirements,
)
from .connection import get_session_factory
from .tables import IntentRow


class IntentDB:
    """Database-backed intent management."""

    def _session(self):
        return get_session_factory()()

    def _row_to_response(self, row: IntentRow) -> IntentResponse:
        status = row.status_json or {}
        return IntentResponse(
            id=str(row.id),
            alias=row.alias,
            model_source=row.model_source,
            replicas=row.replicas,
            priority=row.priority,
            strategy=row.strategy,
            backend=row.backend or {},
            placement=PlacementConstraints(**(row.placement or {})),
            resources=ResourceRequirements(**(row.resources or {})),
            metadata=row.metadata_ or {},
            status=IntentStatus(
                phase=IntentPhase(row.phase),
                reconcile=ReconcileState(row.reconcile),
                desired_replicas=row.replicas,
                observed_replicas=status.get("observed_replicas", 0),
                ready_replicas=status.get("ready_replicas", 0),
                updated_replicas=status.get("updated_replicas", 0),
                available=status.get("available", False),
                shortfall=status.get("shortfall", 0),
                replica_set=status.get("replica_set", []),
                conditions=status.get("conditions", []),
                strategy_progress=status.get("strategy_progress"),
                last_error=status.get("last_error"),
                created_at=row.created_at.isoformat() if row.created_at else None,
                updated_at=row.updated_at.isoformat() if row.updated_at else None,
                last_reconciled_at=(
                    row.last_reconciled_at.isoformat()
                    if row.last_reconciled_at
                    else None
                ),
                ready_at=row.ready_at.isoformat() if row.ready_at else None,
            ),
        )

    def _build_status_json(self) -> dict[str, Any]:
        """Build the initial status JSON for a new intent."""
        return {
            "observed_replicas": 0,
            "ready_replicas": 0,
            "updated_replicas": 0,
            "available": False,
            "shortfall": 0,
            "replica_set": [],
            "conditions": [],
            "strategy_progress": None,
            "last_error": None,
        }

    async def create_intent(
        self,
        *,
        alias: str,
        model_source: str,
        replicas: int,
        priority: str,
        strategy: str,
        backend: dict[str, Any],
        placement: dict[str, Any],
        resources: dict[str, Any],
        metadata: dict[str, str],
    ) -> IntentResponse:
        """Insert a new intent with phase='pending'."""
        now = datetime.now(timezone.utc)
        async with self._session() as session:
            row = IntentRow(
                alias=alias,
                model_source=model_source,
                replicas=replicas,
                priority=priority,
                strategy=strategy,
                backend=backend,
                placement=placement,
                resources=resources,
                metadata_=metadata,
                phase="pending",
                reconcile="idle",
                status_json=self._build_status_json(),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._row_to_response(row)

    async def get_intent(self, intent_id: str) -> IntentResponse | None:
        """Get a single intent by ID (excluding soft-deleted)."""
        async with self._session() as session:
            row = await session.get(IntentRow, intent_id)
            if row is None or row.deleted_at is not None:
                return None
            return self._row_to_response(row)

    async def get_intent_by_alias(self, alias: str) -> IntentResponse | None:
        """Get an active (non-deleted) intent by alias."""
        async with self._session() as session:
            result = await session.execute(
                select(IntentRow).where(
                    IntentRow.alias == alias,
                    IntentRow.deleted_at.is_(None),
                )
            )
            row = result.scalar_one_or_none()
            return self._row_to_response(row) if row else None

    async def list_intents(
        self,
        *,
        alias: str | None = None,
        priority: str | None = None,
        phase: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[IntentResponse]:
        """List active (non-deleted) intents with optional filters."""
        async with self._session() as session:
            stmt = select(IntentRow).where(IntentRow.deleted_at.is_(None))

            if alias:
                stmt = stmt.where(IntentRow.alias == alias)
            if priority:
                stmt = stmt.where(IntentRow.priority == priority)
            if phase:
                stmt = stmt.where(IntentRow.phase == phase)

            stmt = (
                stmt.order_by(IntentRow.created_at.desc()).limit(limit).offset(offset)
            )
            result = await session.execute(stmt)
            return [self._row_to_response(row) for row in result.scalars()]

    async def soft_delete_intent(self, intent_id: str) -> IntentResponse | None:
        """Mark an intent as deleted (soft-delete with deleted_at timestamp)."""
        now = datetime.now(timezone.utc)
        async with self._session() as session:
            row = await session.get(IntentRow, intent_id)
            if row is None or row.deleted_at is not None:
                return None
            row.phase = "deleting"
            row.deleted_at = now
            row.updated_at = now
            await session.commit()
            await session.refresh(row)
            return self._row_to_response(row)

    async def check_alias_conflict(
        self, alias: str, *, exclude_id: str | None = None
    ) -> bool:
        """Return True if an active (non-deleted) intent already uses this alias."""
        async with self._session() as session:
            stmt = select(IntentRow).where(
                IntentRow.alias == alias,
                IntentRow.deleted_at.is_(None),
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return False
            if exclude_id and str(row.id) == exclude_id:
                return False
            return True


intent_db = IntentDB()

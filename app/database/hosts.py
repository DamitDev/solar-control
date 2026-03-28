"""PostgreSQL-backed host CRUD operations using SQLAlchemy ORM."""

from typing import Any
from datetime import datetime, timezone

from sqlalchemy import select, delete, update

from app.models import Host, HostStatus, MemoryInfo
from .connection import get_session_factory
from .tables import HostRow


class HostDB:
    """Database-backed host management."""

    def _session(self):
        return get_session_factory()()

    def _row_to_host(self, row: HostRow) -> Host:
        memory = None
        if row.memory and isinstance(row.memory, dict):
            memory = MemoryInfo(**row.memory)

        roles: list[str] = []
        if isinstance(row.roles, list):
            roles = row.roles

        return Host(
            id=row.id,
            name=row.name,
            url=row.url,
            api_key=row.api_key,
            status=HostStatus(row.status),
            last_seen=row.last_seen,
            memory=memory,
            gpu_type=row.gpu_type,
            roles=roles,
            disk_total_gb=row.disk_total_gb,
            disk_used_gb=row.disk_used_gb,
            disk_available_gb=row.disk_available_gb,
            memory_available_gb=row.memory_available_gb,
            created_at=row.created_at,
        )

    def _host_to_dict(self, host: Host) -> dict[str, Any]:
        return {
            "id": host.id,
            "name": host.name,
            "url": host.url,
            "api_key": host.api_key,
            "status": host.status.value,
            "last_seen": host.last_seen,
            "memory": host.memory.model_dump() if host.memory else None,
            "gpu_type": host.gpu_type,
            "roles": host.roles or [],
            "disk_total_gb": host.disk_total_gb,
            "disk_used_gb": host.disk_used_gb,
            "disk_available_gb": host.disk_available_gb,
            "memory_available_gb": host.memory_available_gb,
            "created_at": host.created_at,
        }

    async def add_host(self, host: Host) -> Host:
        async with self._session() as session:
            existing = await session.get(HostRow, host.id)
            if existing:
                values = self._host_to_dict(host)
                values.pop("id")
                values.pop("created_at")
                await session.execute(
                    update(HostRow).where(HostRow.id == host.id).values(**values)
                )
            else:
                session.add(HostRow(**self._host_to_dict(host)))
            await session.commit()
        return host

    async def remove_host(self, host_id: str) -> bool:
        async with self._session() as session:
            result = await session.execute(delete(HostRow).where(HostRow.id == host_id))
            await session.commit()
            return result.rowcount == 1

    async def get_host(self, host_id: str) -> Host | None:
        async with self._session() as session:
            row = await session.get(HostRow, host_id)
            return self._row_to_host(row) if row else None

    async def get_host_by_api_key(self, api_key: str) -> Host | None:
        async with self._session() as session:
            result = await session.execute(
                select(HostRow).where(HostRow.api_key == api_key)
            )
            row = result.scalar_one_or_none()
            return self._row_to_host(row) if row else None

    async def get_all_hosts(self, *, role: str | None = None) -> list[Host]:
        async with self._session() as session:
            stmt = select(HostRow).order_by(HostRow.created_at)
            if role:
                stmt = stmt.where(HostRow.roles.op("@>")(f'["{role}"]'))
            result = await session.execute(stmt)
            return [self._row_to_host(row) for row in result.scalars()]

    async def update_host_status(
        self,
        host_id: str,
        status: HostStatus,
        *,
        memory: dict[str, Any] | None = None,
    ) -> bool:
        values: dict[str, Any] = {"status": status.value}
        if status == HostStatus.ONLINE:
            values["last_seen"] = datetime.now(timezone.utc)
        if memory is not None:
            values["memory"] = memory

        async with self._session() as session:
            result = await session.execute(
                update(HostRow).where(HostRow.id == host_id).values(**values)
            )
            await session.commit()
            return result.rowcount == 1

    async def update_host_memory(
        self,
        host_id: str,
        memory: dict[str, Any],
        *,
        gpu_type: str | None = None,
        disk_total_gb: float | None = None,
        disk_used_gb: float | None = None,
        disk_available_gb: float | None = None,
    ) -> bool:
        values: dict[str, Any] = {
            "memory": memory,
            "last_seen": datetime.now(timezone.utc),
            "memory_available_gb": memory.get("available_gb"),
        }
        if gpu_type is not None:
            values["gpu_type"] = gpu_type
        if disk_total_gb is not None:
            values["disk_total_gb"] = disk_total_gb
        if disk_used_gb is not None:
            values["disk_used_gb"] = disk_used_gb
        if disk_available_gb is not None:
            values["disk_available_gb"] = disk_available_gb

        async with self._session() as session:
            result = await session.execute(
                update(HostRow).where(HostRow.id == host_id).values(**values)
            )
            await session.commit()
            return result.rowcount == 1

    async def update_host_gpu_type(self, host_id: str, gpu_type: str) -> bool:
        async with self._session() as session:
            result = await session.execute(
                update(HostRow).where(HostRow.id == host_id).values(gpu_type=gpu_type)
            )
            await session.commit()
            return result.rowcount == 1

    async def update_host_roles(self, host_id: str, roles: list[str]) -> bool:
        async with self._session() as session:
            result = await session.execute(
                update(HostRow).where(HostRow.id == host_id).values(roles=roles)
            )
            await session.commit()
            return result.rowcount == 1

    async def update_host_registration(
        self,
        host_id: str,
        *,
        gpu_type: str | None = None,
        roles: list[str] | None = None,
    ) -> bool:
        """Persist gpu_type and roles from a registration event in a single UPDATE."""
        values: dict[str, Any] = {}
        if gpu_type is not None:
            values["gpu_type"] = gpu_type
        if roles is not None:
            values["roles"] = roles
        if not values:
            return True

        async with self._session() as session:
            result = await session.execute(
                update(HostRow).where(HostRow.id == host_id).values(**values)
            )
            await session.commit()
            return result.rowcount == 1


host_db = HostDB()

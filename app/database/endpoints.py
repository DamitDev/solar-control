"""PostgreSQL-backed API endpoint CRUD operations using SQLAlchemy ORM.

Each API endpoint represents a tenant (dev, uat, prod) with its own API key.
All endpoints serve the same models but have separate request logging.
"""

import uuid
import secrets
from typing import Any
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlalchemy import select, delete, update, func

from .connection import get_session_factory
from .tables import ApiEndpointRow, GatewayRequestRow


class ApiEndpoint(BaseModel):
    """An OpenAI-compatible API endpoint (tenant)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    api_key: str = Field(default_factory=lambda: f"sk-{secrets.token_urlsafe(32)}")
    description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_UNSET = object()


class EndpointDB:
    """Database-backed API endpoint management."""

    def _session(self):
        return get_session_factory()()

    def _row_to_endpoint(self, row: ApiEndpointRow) -> ApiEndpoint:
        return ApiEndpoint(
            id=str(row.id),
            name=row.name,
            api_key=row.api_key,
            description=row.description,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def create_endpoint(
        self,
        name: str,
        *,
        description: str | None = None,
        api_key: str | None = None,
    ) -> ApiEndpoint:
        ep = ApiEndpoint(name=name, description=description)
        if api_key:
            ep.api_key = api_key
        async with self._session() as session:
            session.add(
                ApiEndpointRow(
                    id=ep.id,
                    name=ep.name,
                    api_key=ep.api_key,
                    description=ep.description,
                    created_at=ep.created_at,
                    updated_at=ep.updated_at,
                )
            )
            await session.commit()
        return ep

    async def get_endpoint(self, endpoint_id: str) -> ApiEndpoint | None:
        async with self._session() as session:
            row = await session.get(ApiEndpointRow, endpoint_id)
            return self._row_to_endpoint(row) if row else None

    async def get_endpoint_by_api_key(self, api_key: str) -> ApiEndpoint | None:
        async with self._session() as session:
            result = await session.execute(
                select(ApiEndpointRow).where(ApiEndpointRow.api_key == api_key)
            )
            row = result.scalar_one_or_none()
            return self._row_to_endpoint(row) if row else None

    async def get_all_endpoints(self) -> list[ApiEndpoint]:
        async with self._session() as session:
            result = await session.execute(
                select(ApiEndpointRow).order_by(ApiEndpointRow.created_at)
            )
            return [self._row_to_endpoint(row) for row in result.scalars()]

    async def update_endpoint(
        self,
        endpoint_id: str,
        *,
        name: str | None = None,
        description: str | None = _UNSET,  # type: ignore[assignment]
        api_key: str | None = None,
    ) -> ApiEndpoint | None:
        values: dict[str, Any] = {}
        if name is not None:
            values["name"] = name
        if description is not _UNSET:
            values["description"] = description
        if api_key is not None:
            values["api_key"] = api_key

        if not values:
            return await self.get_endpoint(endpoint_id)

        values["updated_at"] = datetime.now(timezone.utc)

        async with self._session() as session:
            result = await session.execute(
                update(ApiEndpointRow)
                .where(ApiEndpointRow.id == endpoint_id)
                .values(**values)
                .returning(ApiEndpointRow)
            )
            row = result.scalar_one_or_none()
            await session.commit()
            return self._row_to_endpoint(row) if row else None

    async def delete_endpoint(self, endpoint_id: str) -> bool:
        async with self._session() as session:
            result = await session.execute(
                delete(ApiEndpointRow).where(ApiEndpointRow.id == endpoint_id)
            )
            await session.commit()
            return result.rowcount == 1

    async def get_usage_stats(
        self, endpoint_id: str, *, hours: int = 24
    ) -> dict[str, Any]:
        """Get usage statistics for a specific endpoint."""
        async with self._session() as session:
            result = await session.execute(
                select(
                    func.count().label("total_requests"),
                    func.count()
                    .filter(GatewayRequestRow.status == "success")
                    .label("successful_requests"),
                    func.count()
                    .filter(GatewayRequestRow.status == "error")
                    .label("error_requests"),
                    func.count()
                    .filter(GatewayRequestRow.status == "missed")
                    .label("missed_requests"),
                    func.coalesce(func.sum(GatewayRequestRow.prompt_tokens), 0).label(
                        "total_prompt_tokens"
                    ),
                    func.coalesce(
                        func.sum(GatewayRequestRow.completion_tokens), 0
                    ).label("total_completion_tokens"),
                    func.coalesce(func.sum(GatewayRequestRow.total_tokens), 0).label(
                        "total_tokens"
                    ),
                    func.avg(GatewayRequestRow.duration_s)
                    .filter(GatewayRequestRow.status == "success")
                    .label("avg_duration_s"),
                    func.avg(GatewayRequestRow.decode_tps)
                    .filter(GatewayRequestRow.decode_tps.isnot(None))
                    .label("avg_decode_tps"),
                ).where(
                    GatewayRequestRow.endpoint_id == endpoint_id,
                    GatewayRequestRow.end_timestamp
                    >= func.now() - func.make_interval(0, 0, 0, 0, hours),
                )
            )
            row = result.one_or_none()

        if not row:
            return {}
        return {
            "total_requests": row.total_requests,
            "successful_requests": row.successful_requests,
            "error_requests": row.error_requests,
            "missed_requests": row.missed_requests,
            "total_prompt_tokens": row.total_prompt_tokens,
            "total_completion_tokens": row.total_completion_tokens,
            "total_tokens": row.total_tokens,
            "avg_duration_s": float(row.avg_duration_s) if row.avg_duration_s else None,
            "avg_decode_tps": float(row.avg_decode_tps) if row.avg_decode_tps else None,
        }


endpoint_db = EndpointDB()

"""SQLAlchemy declarative table models for all PostgreSQL tables."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ApiEndpointRow(Base):
    __tablename__ = "api_endpoints"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    api_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class HostRow(Base):
    __tablename__ = "hosts"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    api_key: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="offline")
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    memory: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    gpu_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    roles: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=list)
    disk_total_gb: Mapped[float | None] = mapped_column(Double, nullable=True)
    disk_used_gb: Mapped[float | None] = mapped_column(Double, nullable=True)
    disk_available_gb: Mapped[float | None] = mapped_column(Double, nullable=True)
    memory_available_gb: Mapped[float | None] = mapped_column(Double, nullable=True)
    version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class GatewayEventRow(Base):
    __tablename__ = "gateway_events"
    __table_args__ = (
        Index("idx_events_timestamp", "timestamp"),
        Index("idx_events_type", "event_type"),
        Index("idx_events_request_id", "request_id"),
        Index("idx_events_endpoint", "endpoint_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoint_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("api_endpoints.id", ondelete="SET NULL"),
        nullable=True,
    )
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class GatewayRequestRow(Base):
    __tablename__ = "gateway_requests"
    __table_args__ = (
        Index("idx_requests_end_ts", "end_timestamp"),
        Index("idx_requests_status", "status"),
        Index("idx_requests_model", "model"),
        Index("idx_requests_host", "host_id"),
        Index("idx_requests_type", "request_type"),
        Index("idx_requests_endpoint", "endpoint_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    request_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoint_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("api_endpoints.id", ondelete="SET NULL"),
        nullable=True,
    )
    client_ip: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    start_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    end_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    duration_s: Mapped[float | None] = mapped_column(Double, nullable=True)
    host_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    host_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    instance_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decode_tps: Mapped[float | None] = mapped_column(Double, nullable=True)
    decode_ms_per_token: Mapped[float | None] = mapped_column(Double, nullable=True)

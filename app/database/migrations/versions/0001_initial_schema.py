"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2025-06-01 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # --- api_endpoints ---
    if not _table_exists(conn, "api_endpoints"):
        op.create_table(
            "api_endpoints",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=False),
                server_default=sa.func.gen_random_uuid(),
                primary_key=True,
            ),
            sa.Column("name", sa.Text(), unique=True, nullable=False),
            sa.Column("api_key", sa.Text(), unique=True, nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )

    # --- hosts ---
    if not _table_exists(conn, "hosts"):
        op.create_table(
            "hosts",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("url", sa.Text(), nullable=False),
            sa.Column("api_key", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False, server_default="offline"),
            sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
            sa.Column("memory", postgresql.JSONB(), nullable=True),
            sa.Column("gpu_type", sa.Text(), nullable=True),
            sa.Column("roles", postgresql.JSONB(), nullable=True),
            sa.Column("disk_total_gb", sa.Double(), nullable=True),
            sa.Column("disk_used_gb", sa.Double(), nullable=True),
            sa.Column("disk_available_gb", sa.Double(), nullable=True),
            sa.Column("memory_available_gb", sa.Double(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )

    # --- gateway_events ---
    if not _table_exists(conn, "gateway_events"):
        op.create_table(
            "gateway_events",
            sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
            sa.Column("event_type", sa.Text(), nullable=False),
            sa.Column("request_id", sa.Text(), nullable=True),
            sa.Column(
                "endpoint_id",
                postgresql.UUID(as_uuid=False),
                sa.ForeignKey("api_endpoints.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("data", postgresql.JSONB(), server_default="{}", nullable=False),
            sa.Column(
                "timestamp",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index("idx_events_timestamp", "gateway_events", ["timestamp"])
        op.create_index("idx_events_type", "gateway_events", ["event_type"])
        op.create_index("idx_events_request_id", "gateway_events", ["request_id"])
        op.create_index("idx_events_endpoint", "gateway_events", ["endpoint_id"])

    # --- gateway_requests ---
    if not _table_exists(conn, "gateway_requests"):
        op.create_table(
            "gateway_requests",
            sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
            sa.Column("request_id", sa.Text(), unique=True, nullable=False),
            sa.Column("request_type", sa.Text(), nullable=True),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("model", sa.Text(), nullable=True),
            sa.Column("resolved_model", sa.Text(), nullable=True),
            sa.Column("endpoint", sa.Text(), nullable=True),
            sa.Column(
                "endpoint_id",
                postgresql.UUID(as_uuid=False),
                sa.ForeignKey("api_endpoints.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("client_ip", sa.Text(), nullable=True),
            sa.Column("stream", sa.Boolean(), nullable=True),
            sa.Column("attempts", sa.Integer(), server_default="1"),
            sa.Column("start_timestamp", sa.DateTime(timezone=True), nullable=True),
            sa.Column("end_timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("duration_s", sa.Double(), nullable=True),
            sa.Column("host_id", sa.Text(), nullable=True),
            sa.Column("host_name", sa.Text(), nullable=True),
            sa.Column("instance_id", sa.Text(), nullable=True),
            sa.Column("instance_url", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=True),
            sa.Column("decode_tps", sa.Double(), nullable=True),
            sa.Column("decode_ms_per_token", sa.Double(), nullable=True),
        )
        op.create_index("idx_requests_end_ts", "gateway_requests", ["end_timestamp"])
        op.create_index("idx_requests_status", "gateway_requests", ["status"])
        op.create_index("idx_requests_model", "gateway_requests", ["model"])
        op.create_index("idx_requests_host", "gateway_requests", ["host_id"])
        op.create_index("idx_requests_type", "gateway_requests", ["request_type"])
        op.create_index("idx_requests_endpoint", "gateway_requests", ["endpoint_id"])


def downgrade() -> None:
    op.drop_table("gateway_requests")
    op.drop_table("gateway_events")
    op.drop_table("hosts")
    op.drop_table("api_endpoints")


def _table_exists(conn, table_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)"
        ),
        {"t": table_name},
    )
    return result.scalar()

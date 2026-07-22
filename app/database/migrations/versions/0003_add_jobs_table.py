"""Add jobs table for job step execution tracking

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-13 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, table_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables"
            "  WHERE table_name = :name"
            ")"
        ),
        {"name": table_name},
    )
    return result.scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "jobs"):
        op.create_table(
            "jobs",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column(
                "host_id",
                sa.Text(),
                sa.ForeignKey("hosts.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column(
                "status",
                sa.Text(),
                nullable=False,
                server_default="pending",
            ),
            sa.Column(
                "payload",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("result", postgresql.JSONB(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("correlation_id", sa.Text(), nullable=True),
            sa.Column("submission_id", sa.Text(), nullable=True),
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
            sa.Column(
                "completed_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )

        op.create_index("idx_jobs_status", "jobs", ["status"])
        op.create_index("idx_jobs_host_id", "jobs", ["host_id"])
        op.create_index("idx_jobs_created_at", "jobs", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_jobs_status", table_name="jobs")
    op.drop_index("idx_jobs_host_id", table_name="jobs")
    op.drop_index("idx_jobs_created_at", table_name="jobs")
    op.drop_table("jobs")

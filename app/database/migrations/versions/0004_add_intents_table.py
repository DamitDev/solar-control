"""Add intents table for declarative deployment intents (S-040)

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-24 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
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

    if not _table_exists(conn, "intents"):
        op.create_table(
            "intents",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=False),
                server_default=sa.func.gen_random_uuid(),
                primary_key=True,
            ),
            sa.Column("alias", sa.Text(), nullable=False),
            sa.Column("model_source", sa.Text(), nullable=False),
            sa.Column("replicas", sa.Integer(), nullable=False, server_default="1"),
            sa.Column(
                "priority", sa.Text(), nullable=False, server_default="production"
            ),
            sa.Column("strategy", sa.Text(), nullable=False, server_default="rolling"),
            sa.Column(
                "backend",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "placement",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "resources",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "metadata",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("phase", sa.Text(), nullable=False, server_default="pending"),
            sa.Column("reconcile", sa.Text(), nullable=False, server_default="idle"),
            sa.Column(
                "status",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
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
                "last_reconciled_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.Column(
                "ready_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.Column(
                "deleted_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )

        op.create_index("idx_intents_alias", "intents", ["alias"])
        op.create_index("idx_intents_phase", "intents", ["phase"])
        op.create_index("idx_intents_priority", "intents", ["priority"])
        op.create_index("idx_intents_created_at", "intents", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_intents_alias", table_name="intents")
    op.drop_index("idx_intents_phase", table_name="intents")
    op.drop_index("idx_intents_priority", table_name="intents")
    op.drop_index("idx_intents_created_at", table_name="intents")
    op.drop_table("intents")

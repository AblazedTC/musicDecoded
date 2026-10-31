"""billing_events: processed RevenueCat event ids (T-601 idempotency)

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-29
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "billing_events",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("type", sa.String(40), nullable=False),
        sa.Column("app_user_id", sa.String(200), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("billing_events")

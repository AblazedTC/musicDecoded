"""library_items (sync, T-402)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-29
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "library_items",
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("track_id", sa.Uuid(), sa.ForeignKey("tracks.id"), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("artist", sa.String(500), nullable=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("source_ref", sa.String(600), nullable=True),
        sa.Column("duration_sec", sa.Double(), nullable=True),
        sa.Column("thumbnail_url", sa.String(1000), nullable=True),
        sa.Column("audio_upload_id", sa.Uuid(), sa.ForeignKey("uploads.id"), nullable=True),
        sa.Column("favorite", sa.Boolean(), nullable=False),
        sa.Column("last_played_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("server_updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_library_items_user_sync", "library_items", ["user_id", "server_updated_at", "track_id"])


def downgrade() -> None:
    op.drop_table("library_items")

"""tracks and jobs

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-29
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tracks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("dedupe_key", sa.String(600), nullable=False, unique=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("duration_sec", sa.Double(), nullable=True),
        sa.Column("thumbnail_url", sa.String(1000), nullable=True),
        sa.Column("analysis_key", sa.String(200), nullable=True),
        sa.Column("audio_key", sa.String(200), nullable=True),
        sa.Column("analyzer_version", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("track_id", sa.Uuid(), sa.ForeignKey("tracks.id"), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("stage", sa.String(40), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_track_id", "jobs", ["track_id"])
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("tracks")

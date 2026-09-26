"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-15 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'feature_definitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('data_type', sa.String(50), nullable=False),
        sa.Column('version', sa.String(50), nullable=False, server_default='v2.0.0'),
        sa.Column('status', sa.String(50), nullable=False, server_default='ACTIVE'),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_feature_def_name_version', 'feature_definitions', ['name', 'version'], unique=True)
    op.create_index('ix_feature_def_category', 'feature_definitions', ['category'])
    op.create_index('ix_feature_def_status', 'feature_definitions', ['status'])

    op.create_table(
        'materialization_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=True),
        sa.Column('lookback_days', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('total_entities', sa.Integer(), nullable=True),
        sa.Column('processed_entities', sa.Integer(), nullable=True),
        sa.Column('failed_entities', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('triggered_by', sa.String(100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_mat_job_status', 'materialization_jobs', ['status'])
    op.create_index('ix_mat_job_tenant', 'materialization_jobs', ['tenant_id'])
    op.create_index('ix_mat_job_created', 'materialization_jobs', ['created_at'])

    op.create_table(
        'drift_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('drift_type', sa.String(50), nullable=False),
        sa.Column('feature_name', sa.String(255), nullable=False),
        sa.Column('kl_divergence', sa.Float(), nullable=False),
        sa.Column('js_divergence', sa.Float(), nullable=False),
        sa.Column('is_drifted', sa.Boolean(), nullable=False),
        sa.Column('baseline_mean', sa.Float(), nullable=False),
        sa.Column('current_mean', sa.Float(), nullable=False),
        sa.Column('baseline_std', sa.Float(), nullable=False),
        sa.Column('current_std', sa.Float(), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('drift_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('computed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_drift_feature', 'drift_reports', ['feature_name'])
    op.create_index('ix_drift_type', 'drift_reports', ['drift_type'])
    op.create_index('ix_drift_computed', 'drift_reports', ['computed_at'])


def downgrade() -> None:
    op.drop_table('drift_reports')
    op.drop_table('materialization_jobs')
    op.drop_table('feature_definitions')

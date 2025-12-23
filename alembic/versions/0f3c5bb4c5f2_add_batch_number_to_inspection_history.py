"""Add batch_number to inspection_history

Revision ID: 0f3c5bb4c5f2
Revises: 5cd42af34616
Create Date: 2025-12-19 15:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0f3c5bb4c5f2"
down_revision: Union[str, Sequence[str], None] = "5cd42af34616"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add batch_number column and index."""
    op.add_column(
        "inspection_history",
        sa.Column("batch_number", sa.String(length=100), nullable=True),
    )
    op.create_index(
        "idx_inspection_batch_number",
        "inspection_history",
        ["batch_number"],
        unique=False,
    )


def downgrade() -> None:
    """Revert batch_number column."""
    op.drop_index("idx_inspection_batch_number", table_name="inspection_history")
    op.drop_column("inspection_history", "batch_number")

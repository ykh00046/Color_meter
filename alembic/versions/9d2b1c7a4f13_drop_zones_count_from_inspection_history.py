"""Drop zones_count from inspection_history

Revision ID: 9d2b1c7a4f13
Revises: 0f3c5bb4c5f2
Create Date: 2026-01-13 13:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9d2b1c7a4f13"
down_revision: Union[str, Sequence[str], None] = "0f3c5bb4c5f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop zones_count column."""
    with op.batch_alter_table("inspection_history") as batch_op:
        batch_op.drop_column("zones_count")


def downgrade() -> None:
    """Re-add zones_count column."""
    with op.batch_alter_table("inspection_history") as batch_op:
        batch_op.add_column(sa.Column("zones_count", sa.Integer(), nullable=False, server_default="0"))

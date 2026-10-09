"""read only api role

A NOLOGIN group role that can only SELECT. Operators create the API's login
user separately (with its own password) as a member of this role, so the API
process cannot modify data even if compromised.

Revision ID: d940ed4bf3e7
Revises: d81dfc442a33
Create Date: 2026-10-09 18:49:34.272237

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d940ed4bf3e7"
down_revision: str | Sequence[str] | None = "d81dfc442a33"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLE = "api_readonly"


def upgrade() -> None:
    op.execute(
        f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{ROLE}') THEN
                CREATE ROLE {ROLE} NOLOGIN;
            END IF;
        END $$;
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {ROLE}")
    op.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {ROLE}")
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {ROLE}")


def downgrade() -> None:
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM {ROLE}")
    op.execute(f"REVOKE SELECT ON ALL TABLES IN SCHEMA public FROM {ROLE}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {ROLE}")
    op.execute(f"DROP ROLE IF EXISTS {ROLE}")

"""Alembic environment: the connection comes from app settings, never from alembic.ini."""

from logging.config import fileConfig

from alembic import context

from app.config import database_settings
from app.storage.database import create_db_engine
from app.storage.models import Base

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = create_db_engine(database_settings())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("offline migrations are not supported; run against a database")
run_migrations_online()

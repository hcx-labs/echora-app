"""Every ORM model matches the schema the migrations produce.

Runs every migration into a new, empty database, then asks Alembic's comparison what the models
disagree on. Indexes and check constraints belong to the migrations and are not compared; tables,
columns, column types and nullability are.
"""

import os
import uuid
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

import core.db.models  # noqa: F401 - registers the models on Base.metadata
from core.config import reset_config_cache
from core.db.session import Base

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"
COMPARED = {"add_table", "remove_table", "add_column", "remove_column", "modify_type"}
COMPARED_NULLABLE = "modify_nullable"


@pytest.fixture
def migrated(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required")
    name = "echora_models_" + uuid.uuid4().hex[:12]
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    target = make_url(url).set(database=name).render_as_string(hide_password=False)
    try:
        monkeypatch.setenv("DATABASE_URL", target)
        reset_config_cache()
        command.upgrade(Config(str(ALEMBIC_INI)), "head")
        yield target.replace("postgresql://", "postgresql+psycopg://", 1)
    finally:
        reset_config_cache()
        with psycopg.connect(url, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _differences(diff):
    # Column changes come grouped in lists; table and index changes on their own.
    for item in diff:
        yield from (item if isinstance(item, list) else [item])


def test_models_match_the_migrated_schema(migrated):
    modelled = set(Base.metadata.tables)
    engine = create_engine(migrated)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "compare_type": True,
                    # Tables without a model are only reached through SQL in the stores.
                    "include_name": lambda name, kind, _: kind != "table" or name in modelled,
                },
            )
            found = [
                (change[0], change[2], change[3]) if change[0].startswith("modify") else change[:2]
                for change in _differences(compare_metadata(context, Base.metadata))
                if change[0] in COMPARED or change[0] == COMPARED_NULLABLE
            ]
    finally:
        engine.dispose()
    assert [str(change) for change in found] == []

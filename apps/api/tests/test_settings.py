import pytest
from pydantic import ValidationError

from app.settings import DEV_JWT_SECRET, Settings

GOOD = "a-production-secret-that-is-long-enough"
DB = "postgresql://u:p@db:5432/x"


def test_prod_without_secret_fails_naming_the_variable():
    with pytest.raises(ValidationError, match="JWT_SECRET is not set"):
        Settings(app_env="prod", jwt_secret="", database_url=DB)


def test_prod_blank_secret_fails():
    with pytest.raises(ValidationError, match="JWT_SECRET is not set"):
        Settings(app_env="prod", jwt_secret="   ", database_url=DB)


def test_short_secret_fails():
    with pytest.raises(ValidationError, match="too short"):
        Settings(app_env="prod", jwt_secret="short", database_url=DB)


def test_prod_without_database_url_fails():
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(app_env="prod", jwt_secret=GOOD, database_url="")


def test_dev_supplies_local_defaults():
    s = Settings(app_env="dev", jwt_secret="", database_url="")
    assert s.jwt_secret == DEV_JWT_SECRET
    assert s.database_url.startswith("postgresql+psycopg://postgres:postgres@localhost")


def test_plain_postgres_url_gets_the_driver():
    assert Settings(app_env="prod", jwt_secret=GOOD, database_url=DB).database_url == "postgresql+psycopg://u:p@db:5432/x"

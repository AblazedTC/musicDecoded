from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

MIN_SECRET_BYTES = 32  # HS256 needs a key of at least 256 bits

# Local development only (APP_ENV=dev). Anywhere else these must come from the environment.
DEV_JWT_SECRET = "local-dev-secret-change-in-production-must-be-long"
DEV_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/musicdecoded"


class Settings(BaseSettings):
    app_env: Literal["dev", "prod"] = "prod"
    jwt_secret: str = ""
    jwt_expiration_seconds: int = 86400
    database_url: str = ""

    @model_validator(mode="after")
    def _require_secrets(self):
        if self.app_env == "dev":
            self.jwt_secret = self.jwt_secret or DEV_JWT_SECRET
            self.database_url = self.database_url or DEV_DATABASE_URL
        if not self.jwt_secret.strip():
            raise ValueError("JWT_SECRET is not set. Set the JWT_SECRET environment variable, "
                             "or run with APP_ENV=dev for local development.")
        if len(self.jwt_secret.encode()) < MIN_SECRET_BYTES:
            raise ValueError(f"JWT_SECRET is too short: {len(self.jwt_secret.encode())} bytes, "
                             f"HS256 needs at least {MIN_SECRET_BYTES} bytes.")
        if not self.database_url:
            raise ValueError("DATABASE_URL is not set, or run with APP_ENV=dev for local development.")
        # The worker uses plain postgresql:// URLs; SQLAlchemy needs the driver named.
        if self.database_url.startswith("postgresql://"):
            self.database_url = "postgresql+psycopg://" + self.database_url.removeprefix("postgresql://")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

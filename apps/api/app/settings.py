from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

MIN_SECRET_BYTES = 32  # HS256 needs a key of at least 256 bits

# Local development only (APP_ENV=dev). Anywhere else these must come from the environment.
DEV_JWT_SECRET = "local-dev-secret-change-in-production-must-be-long"
# 127.0.0.1, not localhost: compose binds IPv4 only, and resolving localhost to ::1 first
# made each new Postgres connection take ~2 minutes on Windows.
DEV_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/musicdecoded"
DEV_REDIS_URL = "redis://127.0.0.1:6379"
DEV_S3_ENDPOINT = "http://127.0.0.1:9000"
DEV_S3_KEY = "minioadmin"  # MinIO root user and password in docker-compose.yml


class Settings(BaseSettings):
    app_env: Literal["dev", "prod"] = "prod"
    jwt_secret: str = ""
    jwt_expiration_seconds: int = 86400
    database_url: str = ""
    redis_url: str = ""

    # Object storage (MinIO locally, R2/S3 in prod). Empty endpoint = AWS S3.
    s3_endpoint: str = ""
    s3_bucket: str = "musicdecoded"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"

    # Feature flags (ClientConfig). OFFLINE_STREAM_AUDIO is PLAN.md R-1.
    offline_stream_audio: bool = False
    soundcloud_enabled: bool = False

    # YouTube Data API v3 key (PLAN.md D-4). No dev default: unset = search returns 403 feature_disabled.
    youtube_api_key: str = ""

    # Daily job limits per plan (PLAN.md §9). Empty/unset = unlimited.
    limit_guest: int | None = 10
    limit_free: int | None = 50
    limit_pro: int | None = None

    @field_validator("limit_guest", "limit_free", "limit_pro", mode="before")
    @classmethod
    def _empty_is_unlimited(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _require_secrets(self):
        if self.app_env == "dev":
            self.jwt_secret = self.jwt_secret or DEV_JWT_SECRET
            self.database_url = self.database_url or DEV_DATABASE_URL
            self.redis_url = self.redis_url or DEV_REDIS_URL
            self.s3_endpoint = self.s3_endpoint or DEV_S3_ENDPOINT
            self.s3_access_key = self.s3_access_key or DEV_S3_KEY
            self.s3_secret_key = self.s3_secret_key or DEV_S3_KEY
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

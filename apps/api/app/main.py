from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import auth, config, errors, jobs, tracks, uploads, youtube
from app.db import migrate
from app.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    migrate()  # alembic upgrade head
    yield


def create_app() -> FastAPI:
    get_settings()  # fail fast on a missing/short JWT_SECRET or DATABASE_URL
    app = FastAPI(title="MusicDecoded API", lifespan=lifespan)
    errors.install(app)
    app.include_router(auth.router)
    app.include_router(config.router)
    app.include_router(uploads.router)
    app.include_router(jobs.router)
    app.include_router(tracks.router)
    app.include_router(youtube.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

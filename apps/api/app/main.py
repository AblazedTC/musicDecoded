from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import auth, errors
from app.db import Base, engine
from app.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    # ponytail: create_all only adds missing tables; switch to Alembic migrations before the first deploy.
    Base.metadata.create_all(engine)
    yield


def create_app() -> FastAPI:
    get_settings()  # fail fast on a missing/short JWT_SECRET or DATABASE_URL
    app = FastAPI(title="MusicDecoded API", lifespan=lifespan)
    errors.install(app)
    app.include_router(auth.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

"""Every error response uses the openapi `Error` schema: {code, message}."""
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

log = logging.getLogger(__name__)

_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
          409: "conflict", 413: "too_large", 429: "rate_limited"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def not_found(what: str) -> ApiError:
    return ApiError(404, "not_found", f"{what} not found")


def parse_id(value: str, what: str) -> uuid.UUID:
    """Path ids are UUIDs; anything else can't exist, so it's a 404 like any unknown id."""
    try:
        return uuid.UUID(value)
    except ValueError:
        raise not_found(what)


def _body(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message})


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return _body(e.status, e.code, e.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _body(400, "bad_request", f"{where}: {first.get('msg', 'invalid request')}".lstrip(": "))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException):
        return _body(e.status_code, _CODES.get(e.status_code, "internal"), str(e.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, e: Exception):
        log.exception("Unhandled error", exc_info=e)
        return _body(500, "internal", "Internal server error")  # never leak details to clients

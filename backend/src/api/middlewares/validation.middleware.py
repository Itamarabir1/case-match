"""Validation middleware: optional request validation. No business logic."""
from fastapi import Request
from fastapi.responses import JSONResponse


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch validation errors (e.g. Pydantic) and return 422."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

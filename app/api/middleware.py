"""API key authentication middleware."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings

# Paths that never require authentication
_PUBLIC_PATHS = frozenset({
    "/api/health",
    "/api/auth/register",
    "/api/auth/login",
    "/api/auth/me",
    "/api/knowledge/topics",
    "/api/knowledge/documents",
    "/docs",
    "/redoc",
    "/openapi.json",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates X-API-Key header on /api/* routes.

    - Skips non-/api/ paths (static files, SPA).
    - Skips public paths (health, docs).
    - Skips all auth when API_KEY is empty (dev mode).
    - /api/observability/* requires ADMIN_API_KEY if set.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip non-API paths
        if not path.startswith("/api/"):
            return await call_next(request)

        # Skip public paths
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Dev mode: no auth when API_KEY is empty
        if not settings.API_KEY:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")

        # Observability endpoints require admin key
        if path.startswith("/api/observability"):
            if settings.ADMIN_API_KEY:
                if api_key != settings.ADMIN_API_KEY:
                    # Check if it's a valid regular key but not admin
                    if api_key == settings.API_KEY:
                        return JSONResponse(
                            status_code=403,
                            content={"detail": "Admin access required"},
                        )
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                    )
                return await call_next(request)
            # No admin key configured, fall through to regular key check

        # Regular API key check (admin key also grants access)
        if api_key == settings.API_KEY or (settings.ADMIN_API_KEY and api_key == settings.ADMIN_API_KEY):
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

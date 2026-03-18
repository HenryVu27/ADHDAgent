"""JWT token creation and validation."""

import secrets
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import settings


class InvalidTokenError(Exception):
    """Raised when a JWT cannot be decoded or has expired."""


def _secret() -> str:
    """Return JWT secret, generating a random one if not configured."""
    if settings.JWT_SECRET:
        return settings.JWT_SECRET
    # Dev mode: generate a per-process secret (tokens won't survive restarts)
    if not hasattr(_secret, "_cache"):
        _secret._cache = secrets.token_hex(32)
    return _secret._cache


def create_token(user_id: int) -> str:
    """Create a signed JWT for user_id. Expires in JWT_EXPIRY_HOURS."""
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> int:
    """Decode a JWT and return user_id. Raises InvalidTokenError on failure."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError) as exc:
        raise InvalidTokenError("Invalid or expired token") from exc

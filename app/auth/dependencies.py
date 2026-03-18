"""FastAPI dependency: get_current_user from Bearer JWT."""

import aiosqlite
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import get_db
from app.auth.jwt import InvalidTokenError, decode_token
from app.models.schemas import UserRow

bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    conn: aiosqlite.Connection = Depends(get_db),
) -> UserRow:
    """Decode Bearer JWT and return the authenticated user.

    Raises 401 for invalid/expired tokens, 401 if user no longer exists.
    Missing Authorization header returns 403 (HTTPBearer default).
    """
    try:
        user_id = decode_token(credentials.credentials)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    async with conn.execute(
        "SELECT id, email, created_at FROM users WHERE id = ?", (user_id,)
    ) as cur:
        row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="User not found")

    return UserRow(id=row["id"], email=row["email"], created_at=row["created_at"])

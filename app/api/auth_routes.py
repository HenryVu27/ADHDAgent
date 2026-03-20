"""Auth endpoints: register, login, me."""

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from passlib.context import CryptContext

from app.api.deps import get_db
from app.auth.dependencies import get_current_user
from app.auth.jwt import create_token
from app.models.schemas import AuthRequest, TokenResponse, UserRow

auth_router = APIRouter(prefix="/auth", tags=["auth"])

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def fetch_user_by_email(conn: aiosqlite.Connection, email: str) -> UserRow | None:
    async with conn.execute(
        "SELECT id, email, name, created_at FROM users WHERE email = ?", (email,)
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return UserRow(id=row["id"], email=row["email"], name=row["name"], created_at=row["created_at"])


async def fetch_user_by_id(conn: aiosqlite.Connection, user_id: int) -> UserRow | None:
    async with conn.execute(
        "SELECT id, email, name, created_at FROM users WHERE id = ?", (user_id,)
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return UserRow(id=row["id"], email=row["email"], name=row["name"], created_at=row["created_at"])


async def create_user(conn: aiosqlite.Connection, email: str, password: str, name: str | None = None) -> UserRow:
    password_hash = _pwd.hash(password)
    async with conn.execute(
        "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?) RETURNING id, email, name, created_at",
        (email, password_hash, name),
    ) as cur:
        row = await cur.fetchone()
    await conn.commit()
    return UserRow(id=row["id"], email=row["email"], name=row["name"], created_at=row["created_at"])


async def verify_password(conn: aiosqlite.Connection, email: str, password: str) -> UserRow | None:
    async with conn.execute(
        "SELECT id, email, name, password_hash, created_at FROM users WHERE email = ?", (email,)
    ) as cur:
        row = await cur.fetchone()
    if not row or not _pwd.verify(password, row["password_hash"]):
        return None
    return UserRow(id=row["id"], email=row["email"], name=row["name"], created_at=row["created_at"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@auth_router.post("/register", response_model=TokenResponse)
async def register(body: AuthRequest, conn: aiosqlite.Connection = Depends(get_db)):
    """Create a new user account and return a JWT."""
    try:
        user = await create_user(conn, body.email, body.password, body.name)
    except aiosqlite.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    return TokenResponse(access_token=create_token(user.id))


@auth_router.post("/login", response_model=TokenResponse)
async def login(body: AuthRequest, conn: aiosqlite.Connection = Depends(get_db)):
    """Authenticate and return a JWT."""
    user = await verify_password(conn, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_token(user.id))


@auth_router.get("/me", response_model=UserRow)
async def me(current_user: UserRow = Depends(get_current_user)):
    """Return the authenticated user's info."""
    return current_user

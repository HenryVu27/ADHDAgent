"""Auth layer tests — JWT utilities and HTTP endpoints."""

import aiosqlite
import pytest

from app.auth.jwt import InvalidTokenError, create_token, decode_token
from app.db import init_db_async


# ---------------------------------------------------------------------------
# JWT utility tests
# ---------------------------------------------------------------------------

class TestJWT:
    def test_create_and_decode_roundtrip(self):
        token = create_token(42)
        assert decode_token(token) == 42

    def test_decode_invalid_token_raises(self):
        with pytest.raises(InvalidTokenError):
            decode_token("not.a.token")

    def test_decode_tampered_token_raises(self):
        token = create_token(1)
        tampered = token[:-4] + "XXXX"
        with pytest.raises(InvalidTokenError):
            decode_token(tampered)


# ---------------------------------------------------------------------------
# HTTP endpoint helpers
# ---------------------------------------------------------------------------

import httpx
from httpx import ASGITransport

from app.api.deps import get_db
from app.main import app


async def _make_auth_client():
    """Returns an AsyncClient wired to an isolated in-memory DB."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await init_db_async(conn)
    app.dependency_overrides[get_db] = lambda: conn
    transport = ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return client, conn


async def _cleanup(conn, app):
    await conn.close()
    app.dependency_overrides.clear()


class TestAuthEndpoints:
    async def test_register_creates_user_and_returns_token(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                resp = await client.post("/api/auth/register", json={
                    "email": "alice@test.com", "password": "secret123"
                })
            assert resp.status_code == 200
            data = resp.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
        finally:
            await _cleanup(conn, app)

    async def test_register_duplicate_email_returns_400(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                await client.post("/api/auth/register", json={
                    "email": "bob@test.com", "password": "password1"
                })
                resp = await client.post("/api/auth/register", json={
                    "email": "bob@test.com", "password": "password1"
                })
            assert resp.status_code == 400
        finally:
            await _cleanup(conn, app)

    async def test_login_with_correct_credentials_returns_token(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                await client.post("/api/auth/register", json={
                    "email": "carol@test.com", "password": "mypassword"
                })
                resp = await client.post("/api/auth/login", json={
                    "email": "carol@test.com", "password": "mypassword"
                })
            assert resp.status_code == 200
            assert "access_token" in resp.json()
        finally:
            await _cleanup(conn, app)

    async def test_login_wrong_password_returns_401(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                await client.post("/api/auth/register", json={
                    "email": "dave@test.com", "password": "correctpassword"
                })
                resp = await client.post("/api/auth/login", json={
                    "email": "dave@test.com", "password": "wrongpassword"
                })
            assert resp.status_code == 401
        finally:
            await _cleanup(conn, app)

    async def test_me_returns_user_info(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                reg = await client.post("/api/auth/register", json={
                    "email": "eve@test.com", "password": "password1"
                })
                token = reg.json()["access_token"]
                resp = await client.get("/api/auth/me", headers={
                    "Authorization": f"Bearer {token}"
                })
            assert resp.status_code == 200
            data = resp.json()
            assert data["email"] == "eve@test.com"
            assert "id" in data
        finally:
            await _cleanup(conn, app)

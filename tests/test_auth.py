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
                    "email": "alice@test.com", "password": "secret123", "name": "Alice Smith"
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
                    "email": "eve@test.com", "password": "password1", "name": "Eve Torres"
                })
                token = reg.json()["access_token"]
                resp = await client.get("/api/auth/me", headers={
                    "Authorization": f"Bearer {token}"
                })
            assert resp.status_code == 200
            data = resp.json()
            assert data["email"] == "eve@test.com"
            assert data["name"] == "Eve Torres"
            assert "id" in data
        finally:
            await _cleanup(conn, app)

    async def test_register_without_name_defaults_to_null(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                reg = await client.post("/api/auth/register", json={
                    "email": "noname@test.com", "password": "password1"
                })
                token = reg.json()["access_token"]
                resp = await client.get("/api/auth/me", headers={
                    "Authorization": f"Bearer {token}"
                })
            assert resp.status_code == 200
            assert resp.json()["name"] is None
        finally:
            await _cleanup(conn, app)


class TestSessionIsolation:
    async def test_list_sessions_only_returns_own_sessions(self):
        """User A cannot see User B's sessions."""
        from app.agent.sqlite_store import SQLiteSessionStore

        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await init_db_async(conn)

        # Create two users
        await conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?), (?, ?)",
            ("a@t.com", "hash_a", "b@t.com", "hash_b"),
        )
        await conn.commit()
        async with conn.execute("SELECT id FROM users WHERE email = 'a@t.com'") as cur:
            user_a_id = (await cur.fetchone())["id"]
        async with conn.execute("SELECT id FROM users WHERE email = 'b@t.com'") as cur:
            user_b_id = (await cur.fetchone())["id"]

        store = SQLiteSessionStore(conn)
        # Create sessions for each user
        await store._ensure_session("session-a", user_id=user_a_id)
        await store._ensure_session("session-b", user_id=user_b_id)
        await conn.commit()

        # User A sees only their session
        sessions_a, total_a = await store.get_all_sessions_paginated(user_id=user_a_id)
        assert total_a == 1
        assert sessions_a[0].session_id == "session-a"
        assert all(s.session_id != "session-b" for s in sessions_a)

        # User B sees only their session
        sessions_b, total_b = await store.get_all_sessions_paginated(user_id=user_b_id)
        assert total_b == 1
        assert sessions_b[0].session_id == "session-b"
        assert all(s.session_id != "session-a" for s in sessions_b)

        await conn.close()

    async def test_get_all_sessions_filters_by_user(self):
        """get_all_sessions returns only the requesting user's sessions."""
        from app.agent.sqlite_store import SQLiteSessionStore

        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await init_db_async(conn)

        await conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?), (?, ?)",
            ("x@t.com", "hash_x", "y@t.com", "hash_y"),
        )
        await conn.commit()
        async with conn.execute("SELECT id FROM users WHERE email = 'x@t.com'") as cur:
            user_x_id = (await cur.fetchone())["id"]
        async with conn.execute("SELECT id FROM users WHERE email = 'y@t.com'") as cur:
            user_y_id = (await cur.fetchone())["id"]

        store = SQLiteSessionStore(conn)
        await store._ensure_session("sess-x", user_id=user_x_id)
        await store._ensure_session("sess-y", user_id=user_y_id)
        await conn.commit()

        sessions_x = await store.get_all_sessions(user_id=user_x_id)
        assert len(sessions_x) == 1
        assert sessions_x[0].session_id == "sess-x"
        assert all(s.session_id != "sess-y" for s in sessions_x)

        await conn.close()


class TestOwnershipEnforcement:
    async def test_user_cannot_access_another_users_session(self):
        """User A gets 403 when accessing User B's session."""
        from app.agent.sqlite_store import SQLiteSessionStore
        from app.api.deps import get_orchestrator, get_session_store
        from app.agent.orchestrator import AgentOrchestrator

        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await init_db_async(conn)
        app.dependency_overrides[get_db] = lambda: conn

        session_store = SQLiteSessionStore(conn)
        orchestrator = AgentOrchestrator(agent=None, session_store=session_store)
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator
        app.dependency_overrides[get_session_store] = lambda: session_store

        transport = ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Register two users
                resp_a = await client.post("/api/auth/register", json={"email": "owner@t.com", "password": "ownerpass1"})
                token_a = resp_a.json()["access_token"]
                resp_b = await client.post("/api/auth/register", json={"email": "intruder@t.com", "password": "intruderpass1"})
                token_b = resp_b.json()["access_token"]

                # Seed a session for user A
                await client.post("/api/session/seed", json={
                    "session_id": "private-session",
                    "child_name": "Alex",
                    "goals": [],
                }, headers={"Authorization": f"Bearer {token_a}"})

                # User B tries to access User A's session
                resp = await client.get(
                    "/api/session/private-session",
                    headers={"Authorization": f"Bearer {token_b}"},
                )

            assert resp.status_code == 403
        finally:
            await conn.close()
            app.dependency_overrides.clear()

    async def test_user_can_access_own_session(self):
        """User gets 200 when accessing their own session."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await init_db_async(conn)
        app.dependency_overrides[get_db] = lambda: conn

        from app.agent.sqlite_store import SQLiteSessionStore
        from app.api.deps import get_orchestrator, get_session_store
        from app.agent.orchestrator import AgentOrchestrator

        # Use SQLiteSessionStore wired to the same conn so _check_session_owner
        # can find the session in the sessions table.
        session_store = SQLiteSessionStore(conn)
        orchestrator = AgentOrchestrator(agent=None, session_store=session_store)
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator
        app.dependency_overrides[get_session_store] = lambda: session_store

        transport = ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp_reg = await client.post("/api/auth/register", json={"email": "self@t.com", "password": "selfpass123"})
                token = resp_reg.json()["access_token"]

                await client.post("/api/session/seed", json={
                    "session_id": "my-session",
                    "child_name": "Sam",
                    "goals": [],
                }, headers={"Authorization": f"Bearer {token}"})

                resp = await client.get(
                    "/api/session/my-session",
                    headers={"Authorization": f"Bearer {token}"},
                )

            assert resp.status_code == 200
        finally:
            await conn.close()
            app.dependency_overrides.clear()

    async def test_protected_route_without_token_returns_403(self):
        client, conn = await _make_auth_client()
        try:
            async with client:
                resp = await client.get("/api/sessions")
            assert resp.status_code == 403
        finally:
            await _cleanup(conn, app)

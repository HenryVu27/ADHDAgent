# Auth Layer Design

**Date:** 2026-03-18

## Goal

Add a self-hosted, multi-user authentication layer. Each user sees only their own sessions and data. Frontend connects to a real backend auth API instead of the current localStorage mock.

## Approach

JWT + bcrypt. Stateless 24-hour signed tokens, standard for React SPAs. All complexity lives in a single FastAPI dependency (`get_current_user`) that is injected into every protected route.

## Security considerations

JWT stored in localStorage is vulnerable to XSS. This is acceptable for the current threat model (personal/coaching app, no financial data). Mitigation: CSP headers, input sanitization already in place. A future iteration could move to HttpOnly cookies.

## Tech Stack

- **Backend:** FastAPI, python-jose[cryptography] (JWT), passlib[bcrypt] (password hashing), aiosqlite (existing)
- **Frontend:** React 19, existing SignIn/SignUp pages wired to real API calls

---

## Data Model (SCHEMA_V5)

Add to `app/db.py` as `SCHEMA_V5`, registered in the `MIGRATIONS` dict as key `5`.

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
```

**Note on existing data:** After the migration, existing sessions will have `user_id = NULL`. On startup, after seeding the default test user, a cleanup step assigns all NULL sessions to the default user's ID (see Startup Sequence below).

All other tables (goals, messages, family_profiles, episodes, outcomes, etc.) already link through `session_id → sessions → user_id` — no additional schema changes needed.

---

## Backend

### New files

| File | Purpose |
|---|---|
| `app/auth/__init__.py` | Package marker |
| `app/auth/jwt.py` | `create_token`, `decode_token` |
| `app/auth/dependencies.py` | `get_current_user` FastAPI dependency |
| `app/api/auth_routes.py` | Register and login endpoints |

### Models (`app/models/schemas.py`)

Add:
```python
class UserRow(BaseModel):
    id: int
    email: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class AuthRequest(BaseModel):
    email: str
    password: str
```

### Auth endpoints (`app/api/auth_routes.py`)

DB query helpers live in `app/api/auth_routes.py` (same file, small module):
- `fetch_user_by_email(conn, email) -> UserRow | None`
- `fetch_user_by_id(conn, user_id) -> UserRow | None`
- `create_user(conn, email, password_hash) -> UserRow`

**`POST /api/auth/register`**
- Body: `AuthRequest`
- Hashes password with bcrypt, inserts user, returns `TokenResponse`
- Error: 400 if email already registered

**`POST /api/auth/login`**
- Body: `AuthRequest`
- Verifies bcrypt hash, returns `TokenResponse`
- Error: 401 on invalid credentials (no distinction between wrong email vs wrong password)

**`GET /api/auth/me`**
- Protected by `get_current_user`
- Returns `UserRow` (id + email) so the frontend can restore user state on page reload without decoding the JWT client-side

### JWT (`app/auth/jwt.py`)

```python
class InvalidTokenError(Exception):
    pass

def create_token(user_id: int) -> str: ...
def decode_token(token: str) -> int:
    """Returns user_id. Raises InvalidTokenError on invalid or expired token."""
```

- Library: `python-jose[cryptography]`
- Payload: `{"sub": str(user_id), "exp": now + JWT_EXPIRY_HOURS}`
- `decode_token` raises `InvalidTokenError` (not `HTTPException`) — HTTP concerns stay in the dependency layer

### Dependency (`app/auth/dependencies.py`)

```python
bearer = HTTPBearer()  # from fastapi.security — returns 403 if header missing, integrates with OpenAPI

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    conn: aiosqlite.Connection = Depends(get_db),
) -> UserRow:
    try:
        user_id = decode_token(credentials.credentials)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = await fetch_user_by_id(conn, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
```

### Session store changes

**`_ensure_session` propagation strategy:**

`_ensure_session` is called from many internal store methods (save_message, manage_goal, add_episode, etc.) deep inside the agent tools where no `user_id` is in scope. Only the first call for a new session needs `user_id` — after that the INSERT OR IGNORE is a no-op and the column is already set.

The approach: `_ensure_session` gains an optional `user_id: int | None = None` parameter. The orchestrator calls it with a `user_id` once at session start; all internal store methods pass `user_id=None`. This means `INSERT OR IGNORE INTO sessions (session_id, user_id) VALUES (?, ?)` with `None` for `user_id` could create an ownerless session — but this never happens in practice because the orchestrator always initializes the session first before any internal methods are called.

```python
async def _ensure_session(self, session_id: str, user_id: int | None = None) -> None:
    await self._conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
```

All session store methods that return session lists (`get_all_sessions`, `get_all_sessions_paginated`) gain a `user_id: int` parameter and add `WHERE user_id = ?` filtering.

`SessionStoreBase` (store protocol) is updated accordingly.

### Ownership enforcement

A reusable helper in `app/api/routes.py`:
```python
async def get_session_for_user(
    session_id: str,
    user_id: int,
    conn: aiosqlite.Connection,
) -> None:
    """Raises 404 if session not found, 403 if session belongs to a different user."""
    async with conn.execute(
        "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
```

Called at the start of every route that takes a `session_id` path parameter, before any data access.

### `AgentOrchestrator.process_stream`

Gains a `user_id: int` parameter. Calls `_ensure_session(session_id, user_id)` at the start so new sessions are created with the correct owner. No other orchestrator changes needed.

### `ChatRequest.session_id`

Remove the `= "default"` default — make `session_id` required. The frontend already generates session IDs via `createSessionId()`, so no frontend change is needed. A missing `session_id` in the request body will return a 422 validation error.

### Protected routes

All existing endpoints add `current_user: UserRow = Depends(get_current_user)`:

| Endpoint | Change |
|---|---|
| `POST /api/chat/stream` | Pass `user_id` to `process_stream`; session creation sets owner |
| `POST /api/session/seed` | Pass `user_id` to session store |
| `DELETE /api/session/{session_id}` | Call `get_session_for_user` first |
| `GET /api/session/{session_id}` | Call `get_session_for_user` first |
| `GET /api/session/{session_id}/outcomes` | Call `get_session_for_user` first |
| `GET /api/session/{session_id}/messages` | Call `get_session_for_user` first |
| `GET /api/sessions` | Pass `user_id` to store; returns only user's sessions |
| `GET /api/observability/*` | Require auth (any logged-in user); retain `ADMIN_API_KEY` gate for extra sensitivity |

**Public routes (no auth):**
- `GET /api/health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/knowledge/topics`
- `GET /api/knowledge/documents`

### Config (`app/config.py`)

```python
JWT_SECRET: str = ""           # Required in production; see startup validation
JWT_EXPIRY_HOURS: int = 24
DEFAULT_USER_EMAIL: str = "test@test.com"
DEFAULT_USER_PASSWORD: str = "test1234"
```

### Startup sequence (`app/main.py` lifespan)

Order matters:

1. `await init_db_async(conn)` — runs SCHEMA_V5, creates `users` table and adds `user_id` column
2. Seed default test user: if `DEFAULT_USER_EMAIL` is set and no user exists with that email, insert user with bcrypt-hashed `DEFAULT_USER_PASSWORD`
3. Assign orphan sessions: `UPDATE sessions SET user_id = <default_user_id> WHERE user_id IS NULL`
4. JWT secret validation: if `JWT_SECRET` is empty, generate a random 32-byte hex secret and log a warning (tokens won't survive restarts). If `API_KEY` is set (production mode), raise a startup error instead.

### `APIKeyMiddleware` update

Exempt `/api/auth/register`, `/api/auth/login`, and `/api/auth/me` from the API key check.

---

## Frontend

### Changed files

| File | Change |
|---|---|
| `src/lib/auth.ts` | Replace localStorage mock with real API calls |
| `src/lib/api.ts` | Attach `Authorization: Bearer <token>` header; handle 401 auto-signout |
| `src/hooks/use-auth.tsx` | Make signIn/signUp async, add loading/error state, use `/api/auth/me` on init |
| `src/pages/SignUpPage.tsx` | Remove `name` field (backend users table has no name column) |
| `src/types/index.ts` | Remove `name` from `User` interface |
| `App.tsx` | Move `/observability` behind `ProtectedRoute` |

### `src/lib/auth.ts`

- `signUp(email, password)` → `POST /api/auth/register` → store JWT in localStorage under key `adhd_token`
- `signIn(email, password)` → `POST /api/auth/login` → store JWT in localStorage
- `signOut()` → remove JWT from localStorage
- `getToken()` → read JWT from localStorage
- `getMe()` → `GET /api/auth/me` → returns `{id, email}` for restoring session on reload

**Note:** The `name` field is removed from the `User` interface and `signUp`. The `SignUpPage` is updated to remove the name input.

### `src/lib/api.ts`

All fetch calls include:
```typescript
headers: { "Authorization": `Bearer ${getToken()}` }
```

On `401` response → call `signOut()` and `window.location.href = '/signin'`.

### `src/hooks/use-auth.tsx`

- On `AuthProvider` mount: call `getMe()` to restore user from a valid token, set `isLoading` during the check
- `signIn` and `signUp` become async, return `Promise<void>`
- Add `isLoading: boolean` and `error: string | null` to context

### `App.tsx`

Add `/observability` to the `ProtectedRoute` wrapper.

---

## Error handling

| Scenario | Response |
|---|---|
| Invalid/expired JWT | 401 `"Invalid or expired token"` |
| Missing Authorization header | 403 (FastAPI HTTPBearer default) |
| Session belongs to different user | 403 `"Access denied"` |
| Email already registered | 400 `"Email already registered"` |
| Wrong credentials | 401 `"Invalid credentials"` |
| Frontend receives 401 | Auto sign-out + redirect to /signin |

---

## Testing (`tests/test_auth.py`)

- Register new user → success
- Register duplicate email → 400
- Login with correct credentials → JWT returned
- Login with wrong password → 401
- Access protected route without token → 403
- Access protected route with valid token → 200
- Access another user's session → 403
- Expired token → 401
- `GET /api/auth/me` returns correct user
- Default test user exists on startup

# Auth Layer Design

**Date:** 2026-03-18

## Goal

Add a self-hosted, multi-user authentication layer. Each user sees only their own sessions and data. Frontend connects to a real backend auth API instead of the current localStorage mock.

## Approach

JWT + bcrypt. Stateless 24-hour signed tokens, standard for React SPAs. All complexity lives in a single FastAPI dependency (`get_current_user`) that is injected into every protected route.

## Tech Stack

- **Backend:** FastAPI, python-jose (JWT), passlib[bcrypt] (password hashing), aiosqlite (existing)
- **Frontend:** React 19, existing SignIn/SignUp pages wired to real API calls

---

## Data Model (SCHEMA_V5)

### New table: `users`

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### Modified table: `sessions`

Add `user_id` FK column:

```sql
ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
```

All other tables (goals, messages, family_profiles, episodes, outcomes, etc.) already link through `session_id → sessions → user_id` — no additional changes needed.

---

## Backend

### New files

| File | Purpose |
|---|---|
| `app/api/auth_routes.py` | `POST /api/auth/register`, `POST /api/auth/login` |
| `app/auth/jwt.py` | `create_token(user_id)`, `decode_token(token)` |
| `app/auth/dependencies.py` | `get_current_user` FastAPI dependency |

### Auth endpoints (`app/api/auth_routes.py`)

**`POST /api/auth/register`**
- Body: `{"email": str, "password": str}`
- Validates email uniqueness, hashes password with bcrypt, inserts user row, returns JWT
- Response: `{"access_token": "...", "token_type": "bearer"}`
- Errors: 400 if email already registered

**`POST /api/auth/login`**
- Body: `{"email": str, "password": str}`
- Verifies bcrypt hash, returns JWT on success
- Response: `{"access_token": "...", "token_type": "bearer"}`
- Errors: 401 if credentials invalid (single message, no email/password distinction)

### JWT (`app/auth/jwt.py`)

- Library: `python-jose[cryptography]`
- Token payload: `{"sub": str(user_id), "exp": now + JWT_EXPIRY_HOURS}`
- Secret: `settings.JWT_SECRET` (required in production)
- `create_token(user_id: int) -> str`
- `decode_token(token: str) -> int` — returns `user_id`, raises `401` on invalid/expired

### Dependency (`app/auth/dependencies.py`)

```python
async def get_current_user(
    authorization: str = Header(...),
    conn: aiosqlite.Connection = Depends(get_db),
) -> UserRow:
    token = authorization.removeprefix("Bearer ").strip()
    user_id = decode_token(token)
    user = await fetch_user_by_id(conn, user_id)
    if not user:
        raise HTTPException(401)
    return user
```

### Protected routes

All existing endpoints add `current_user: UserRow = Depends(get_current_user)`:

- `POST /api/chat/stream` — session must belong to `current_user.id`
- `POST /api/session/seed` — seeds session tied to `current_user.id`
- `DELETE /api/session/{session_id}` — only if session belongs to user
- `GET /api/session/{session_id}` — only if session belongs to user
- `GET /api/session/{session_id}/outcomes` — ownership check
- `GET /api/session/{session_id}/messages` — ownership check
- `GET /api/sessions` — filtered to `current_user.id` only
- `GET /api/observability/*` — requires auth (previously unprotected)

**Public routes (no auth):**
- `GET /api/health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/knowledge/topics`
- `GET /api/knowledge/documents`

### Config (`app/config.py`)

New settings:
```python
JWT_SECRET: str = ""           # Required in production
JWT_EXPIRY_HOURS: int = 24
DEFAULT_USER_EMAIL: str = "test@test.com"
DEFAULT_USER_PASSWORD: str = "test1234"
```

### Default test user seeding (`app/main.py`)

On startup, if `DEFAULT_USER_EMAIL` is set and no user with that email exists, insert the user (bcrypt-hashed password). This ensures a working test account is always available in dev without manual setup.

### `APIKeyMiddleware` update

Exempt `/api/auth/register` and `/api/auth/login` from the API key check (they have their own auth).

---

## Frontend

### Changed files

| File | Change |
|---|---|
| `src/lib/auth.ts` | Replace localStorage mock with real API calls |
| `src/lib/api.ts` | Attach `Authorization: Bearer <token>` to all requests |
| `src/hooks/use-auth.tsx` | Make signIn/signUp async, add loading/error state |
| `src/pages/ObservabilityPage.tsx` | Move behind ProtectedRoute in App.tsx |

### `src/lib/auth.ts`

- `signUp(email, password)` → `POST /api/auth/register` → store JWT in localStorage
- `signIn(email, password)` → `POST /api/auth/login` → store JWT in localStorage
- `signOut()` → remove JWT from localStorage
- `getToken()` → read JWT from localStorage

### `src/lib/api.ts`

All fetch/axios calls include:
```typescript
headers: { "Authorization": `Bearer ${getToken()}` }
```

On `401` response → call `signOut()` and redirect to `/signin`.

### `src/hooks/use-auth.tsx`

- `signIn` and `signUp` become async, return `Promise<void>`
- Add `isLoading: boolean` and `error: string | null` to context
- SignIn/SignUp pages show loading state and display error messages from API

### `App.tsx`

Add `/observability` behind `ProtectedRoute` (currently fully public).

---

## Error handling

| Scenario | Response |
|---|---|
| Invalid/expired JWT | 401 Unauthorized |
| Session belongs to different user | 403 Forbidden |
| Email already registered | 400 Bad Request |
| Wrong credentials | 401 Unauthorized (no detail distinguishing email vs password) |
| Frontend receives 401 | Auto sign-out + redirect to /signin |

---

## Testing

- `tests/test_auth.py` — register, login, duplicate email, wrong password, expired token, protected route without token, session ownership enforcement
- Existing tests unaffected: they use `InMemorySessionStore` and don't hit auth middleware

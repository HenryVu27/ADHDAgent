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

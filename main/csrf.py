from __future__ import annotations

import hmac
import secrets

from flask import session, request


def generate_csrf_token() -> str:
    """Return the CSRF token for the current session, creating one if needed."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def validate_csrf_token() -> bool:
    """Return True if the CSRF token in the request matches the session token."""
    try:
        form_token = request.form.get("csrf_token")
    except Exception:
        form_token = None
    token = form_token or request.headers.get("X-CSRF-Token", "")
    expected = session.get("csrf_token", "")
    if not token or not expected:
        return False
    return hmac.compare_digest(str(token), str(expected))

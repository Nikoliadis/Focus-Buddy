from __future__ import annotations

from functools import wraps
from typing import Callable, Any

from flask import session, redirect, url_for, request

def login_required(view: Callable[..., Any]):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login_page", next=next_url))
        return view(*args, **kwargs)
    return wrapper
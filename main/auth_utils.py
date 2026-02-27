from __future__ import annotations

from functools import wraps
from urllib.parse import urlparse, urljoin

from flask import request, redirect, url_for, session, flash


def _is_safe_url(target: str) -> bool:
    """Allow only same-host redirects (prevents open redirect)."""
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return (test_url.scheme in ("http", "https")) and (ref_url.netloc == test_url.netloc)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            nxt = request.full_path if request.query_string else request.path
            if not _is_safe_url(nxt):
                nxt = "/"
            flash("Please log in to continue.", "info")
            return redirect(url_for("auth.login_page", next=nxt))
        return view(*args, **kwargs)

    return wrapped
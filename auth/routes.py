from __future__ import annotations

from flask import render_template, request, redirect, url_for, flash, session

from . import auth_bp
from .service import find_user_by_login, create_user, verify_password, user_exists
from urllib.parse import urlparse, urljoin

def _is_safe_url(target: str) -> bool:
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc

@auth_bp.get("/login")
def login_page():
    if session.get("user_id"):
        return redirect(url_for("main.home"))
    next_url = request.args.get("next", "")
    return render_template("auth/login.html", next=next_url)


@auth_bp.post("/login")
def login_post():
    login = (request.form.get("login") or "").strip()
    password = request.form.get("password") or ""
    next_url = request.form.get("next") or ""

    user = find_user_by_login(login)
    if not user or not verify_password(user, password):
        flash("Invalid username/email or password.", "error")
        return redirect(url_for("auth.login_page"))

    session.permanent = True
    session["user_id"] = user.id
    session["username"] = user.username
    flash("Logged in successfully ✅", "success")

    if _is_safe_url(next_url):
        return redirect(next_url)
    return redirect(url_for("main.home"))


@auth_bp.get("/register")
def register_page():
    if session.get("user_id"):
        return redirect(url_for("main.home"))
    return render_template("auth/register.html")


@auth_bp.post("/register")
def register_post():
    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    # minimal validation (fast + safe)
    if len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
        return redirect(url_for("auth.register_page"))

    if "@" not in email or "." not in email:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("auth.register_page"))

    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
        return redirect(url_for("auth.register_page"))

    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("auth.register_page"))

    if user_exists(username, email):
        flash("Username or email already exists.", "error")
        return redirect(url_for("auth.register_page"))

    user = create_user(username, email, password)
    session["user_id"] = user.id
    session["username"] = user.username
    flash("Account created ✅", "success")
    return redirect(url_for("main.home"))


@auth_bp.get("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return redirect(url_for("main.home"))

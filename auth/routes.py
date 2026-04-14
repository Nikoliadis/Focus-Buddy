from __future__ import annotations

import re

from flask import render_template, request, redirect, url_for, flash, session

from main.auth_utils import login_required
from main.db import db
from main.limiter_ext import limiter
from models.user import User
from . import auth_bp
from .service import (
    find_user_by_login_identifier,
    find_user_by_username,
    find_user_by_email,
    create_user,
    verify_password,
)

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")


@auth_bp.get("/login")
def login_page():
    if session.get("user_id"):
        return redirect(url_for("main.home"))
    next_url = request.args.get("next") or ""
    return render_template("auth/login.html", next=next_url)


@auth_bp.post("/login")
@limiter.limit("10 per minute")
def login_post():
    identifier = (request.form.get("identifier") or "").strip()
    password = request.form.get("password") or ""
    next_url = (request.form.get("next") or "").strip()

    if not identifier or not password:
        flash("Please fill in all fields.", "error")
        return redirect(url_for("auth.login_page", next=next_url))

    if len(identifier) > 255 or len(password) > 128:
        flash("Invalid credentials.", "error")
        return redirect(url_for("auth.login_page", next=next_url))

    user = find_user_by_login_identifier(identifier)
    if not user or not verify_password(user, password):
        flash("Invalid credentials.", "error")
        return redirect(url_for("auth.login_page", next=next_url))

    session.permanent = True
    session["user_id"] = user.id
    session["username"] = user.username

    flash("Logged in successfully ✅", "success")
    if next_url:
        return redirect(next_url)
    return redirect(url_for("main.home"))


@auth_bp.get("/register")
def register_page():
    if session.get("user_id"):
        return redirect(url_for("main.home"))
    return render_template("auth/register.html")


@auth_bp.post("/register")
@limiter.limit("5 per minute")
def register_post():
    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if not _USERNAME_RE.match(username):
        flash("Username must be 3–30 characters: letters, numbers, underscores only.", "error")
        return redirect(url_for("auth.register_page"))

    if len(email) > 255 or "@" not in email or "." not in email.split("@")[-1]:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("auth.register_page"))

    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("auth.register_page"))

    if len(password) > 128:
        flash("Password is too long.", "error")
        return redirect(url_for("auth.register_page"))

    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("auth.register_page"))

    if find_user_by_username(username):
        flash("Username already exists.", "error")
        return redirect(url_for("auth.register_page"))

    if find_user_by_email(email):
        flash("Email already exists.", "error")
        return redirect(url_for("auth.register_page"))

    user = create_user(username, email, password)

    session.permanent = True
    session["user_id"] = user.id
    session["username"] = user.username

    flash("Account created ✅", "success")
    return redirect(url_for("main.home"))


@auth_bp.get("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("main.home"))


@auth_bp.get("/profile")
@login_required
def profile_page():
    user = db.session.get(User, session["user_id"])
    return render_template("auth/profile.html", user=user)


@auth_bp.post("/profile")
@login_required
def profile_post():
    user = db.session.get(User, session["user_id"])
    action = request.form.get("action")

    if action == "username":
        new_username = (request.form.get("username") or "").strip()
        if not _USERNAME_RE.match(new_username):
            flash("Username must be 3–30 characters: letters, numbers, underscores only.", "error")
            return redirect(url_for("auth.profile_page"))
        if new_username != user.username and find_user_by_username(new_username):
            flash("Username already taken.", "error")
            return redirect(url_for("auth.profile_page"))
        user.username = new_username
        session["username"] = new_username
        db.session.commit()
        flash("Username updated ✅", "success")

    elif action == "password":
        current = request.form.get("current_password") or ""
        new_pw = request.form.get("new_password") or ""
        confirm = request.form.get("confirm_password") or ""
        if not verify_password(user, current):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("auth.profile_page"))
        if len(new_pw) < 8:
            flash("New password must be at least 8 characters.", "error")
            return redirect(url_for("auth.profile_page"))
        if len(new_pw) > 128:
            flash("Password is too long.", "error")
            return redirect(url_for("auth.profile_page"))
        if new_pw != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("auth.profile_page"))
        from werkzeug.security import generate_password_hash
        user.password_hash = generate_password_hash(new_pw)
        db.session.commit()
        flash("Password updated ✅", "success")

    elif action == "goal":
        try:
            goal = int(request.form.get("daily_goal_minutes") or "0")
        except ValueError:
            goal = 0
        user.daily_goal_minutes = goal if goal > 0 else None
        db.session.commit()
        flash("Daily goal updated ✅", "success")

    return redirect(url_for("auth.profile_page"))
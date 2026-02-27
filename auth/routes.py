from __future__ import annotations

from flask import render_template, request, redirect, url_for, flash, session

from . import auth_bp
from .service import (
    find_user_by_login_identifier,
    find_user_by_username,
    find_user_by_email,
    create_user,
    verify_password,
)


@auth_bp.get("/login")
def login_page():
    if session.get("user_id"):
        return redirect(url_for("main.home"))
    next_url = request.args.get("next") or ""
    return render_template("auth/login.html", next=next_url)


@auth_bp.post("/login")
def login_post():
    identifier = (request.form.get("identifier") or "").strip()
    password = request.form.get("password") or ""
    next_url = (request.form.get("next") or "").strip()

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
def register_post():
    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
        return redirect(url_for("auth.register_page"))

    if "@" not in email or "." not in email:
        flash("Please enter a valid email.", "error")
        return redirect(url_for("auth.register_page"))

    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
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
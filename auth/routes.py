from __future__ import annotations

import re

from flask import render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash

from main.auth_utils import login_required
from main.db import db
from main.limiter_ext import limiter
from main.email_utils import send_verification_email, send_password_reset_email
from models.user import User
from models.tokens import EmailVerificationToken, PasswordResetToken
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

    if not user.email_verified:
        flash("Please verify your email before logging in. Check your inbox or request a new link.", "error")
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

    tok = EmailVerificationToken.generate(user.id)
    db.session.commit()

    try:
        verify_url = url_for("auth.verify_email", token=tok.token, _external=True)
        send_verification_email(user.email, user.username, verify_url)
    except Exception:
        pass

    return render_template("auth/verify_pending.html", email=email)


@auth_bp.get("/verify-email/<token>")
def verify_email(token: str):
    tok = EmailVerificationToken.query.filter_by(token=token).first()
    if not tok or not tok.is_valid():
        flash("This verification link is invalid or has expired.", "error")
        return redirect(url_for("auth.login_page"))

    tok.used = True
    user = db.session.get(User, tok.user_id)
    user.email_verified = True
    db.session.commit()

    flash("Email verified ✅ You can now log in.", "success")
    return redirect(url_for("auth.login_page"))


@auth_bp.get("/resend-verification")
def resend_verification_page():
    return render_template("auth/resend_verification.html")


@auth_bp.post("/resend-verification")
@limiter.limit("3 per minute")
def resend_verification_post():
    email = (request.form.get("email") or "").strip().lower()
    user = find_user_by_email(email)

    # Always show success to avoid email enumeration
    if user and not user.email_verified:
        tok = EmailVerificationToken.generate(user.id)
        db.session.commit()
        try:
            verify_url = url_for("auth.verify_email", token=tok.token, _external=True)
            send_verification_email(user.email, user.username, verify_url)
        except Exception:
            pass

    flash("If that email is registered and unverified, a new link has been sent.", "info")
    return redirect(url_for("auth.login_page"))


@auth_bp.get("/forgot-password")
def forgot_password_page():
    return render_template("auth/forgot_password.html")


@auth_bp.post("/forgot-password")
@limiter.limit("5 per minute")
def forgot_password_post():
    email = (request.form.get("email") or "").strip().lower()
    user = find_user_by_email(email)

    if user:
        tok = PasswordResetToken.generate(user.id)
        db.session.commit()
        try:
            reset_url = url_for("auth.reset_password_page", token=tok.token, _external=True)
            send_password_reset_email(user.email, user.username, reset_url)
        except Exception:
            pass

    flash("If that email is registered, a password reset link has been sent.", "info")
    return redirect(url_for("auth.login_page"))


@auth_bp.get("/reset-password/<token>")
def reset_password_page(token: str):
    tok = PasswordResetToken.query.filter_by(token=token).first()
    if not tok or not tok.is_valid():
        flash("This reset link is invalid or has expired.", "error")
        return redirect(url_for("auth.forgot_password_page"))
    return render_template("auth/reset_password.html", token=token)


@auth_bp.post("/reset-password/<token>")
@limiter.limit("5 per minute")
def reset_password_post(token: str):
    tok = PasswordResetToken.query.filter_by(token=token).first()
    if not tok or not tok.is_valid():
        flash("This reset link is invalid or has expired.", "error")
        return redirect(url_for("auth.forgot_password_page"))

    new_pw = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""

    if len(new_pw) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("auth.reset_password_page", token=token))

    if len(new_pw) > 128:
        flash("Password is too long.", "error")
        return redirect(url_for("auth.reset_password_page", token=token))

    if new_pw != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("auth.reset_password_page", token=token))

    tok.used = True
    user = db.session.get(User, tok.user_id)
    user.password_hash = generate_password_hash(new_pw)
    db.session.commit()

    flash("Password updated ✅ You can now log in.", "success")
    return redirect(url_for("auth.login_page"))


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

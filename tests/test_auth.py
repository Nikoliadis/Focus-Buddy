from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from werkzeug.security import check_password_hash

from main.db import db
from models.tokens import EmailVerificationToken, PasswordResetToken
from models.user import User


def _register(post_form, **overrides):
    data = {
        "username": "alice",
        "email": "alice@example.com",
        "password": "password123",
        "confirm": "password123",
        **overrides,
    }
    return post_form("/register", data)


class TestRegister:
    def test_creates_unverified_user_and_token(self, post_form):
        res = _register(post_form)

        assert res.status_code == 200
        user = User.query.filter_by(username="alice").one()
        assert user.email_verified is False
        assert user.password_hash != "password123"
        assert EmailVerificationToken.query.filter_by(user_id=user.id).count() == 1

    def test_email_is_normalised_to_lowercase(self, post_form):
        _register(post_form, email="Alice@Example.COM")
        assert User.query.one().email == "alice@example.com"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"username": "ab"},
            {"username": "bad name!"},
            {"email": "not-an-email"},
            {"password": "short", "confirm": "short"},
            {"password": "x" * 129, "confirm": "x" * 129},
            {"confirm": "different123"},
        ],
        ids=["short-username", "invalid-username", "invalid-email", "short-password", "long-password", "mismatch"],
    )
    def test_rejects_invalid_input(self, post_form, overrides):
        res = _register(post_form, **overrides)

        assert res.status_code == 302
        assert User.query.count() == 0

    def test_rejects_duplicate_username(self, post_form, make_user):
        make_user("alice")
        res = _register(post_form, email="other@example.com")

        assert res.status_code == 302
        assert User.query.count() == 1

    def test_rejects_duplicate_email(self, post_form, make_user):
        make_user("alice")
        res = _register(post_form, username="bob")

        assert res.status_code == 302
        assert User.query.count() == 1


class TestCsrf:
    def test_post_without_token_is_forbidden(self, client):
        res = client.post("/login", data={"identifier": "alice", "password": "password123"})
        assert res.status_code == 403

    def test_post_with_wrong_token_is_forbidden(self, client):
        res = client.post("/login", data={"csrf_token": "wrong", "identifier": "a", "password": "b"})
        assert res.status_code == 403

    def test_token_accepted_from_header(self, client, make_user):
        make_user("alice")
        res = client.post(
            "/login",
            data={"identifier": "alice", "password": "password123"},
            headers={"X-CSRF-Token": "test-csrf-token"},
        )
        assert res.status_code == 302


class TestLogin:
    def test_login_with_username(self, client, post_form, make_user):
        user = make_user("alice")
        res = post_form("/login", {"identifier": "alice", "password": "password123"})

        assert res.status_code == 302
        with client.session_transaction() as sess:
            assert sess["user_id"] == user.id

    def test_login_with_email_is_case_insensitive(self, client, post_form, make_user):
        make_user("alice")
        post_form("/login", {"identifier": "ALICE@example.com", "password": "password123"})

        with client.session_transaction() as sess:
            assert sess["username"] == "alice"

    def test_wrong_password(self, client, post_form, make_user):
        make_user("alice")
        post_form("/login", {"identifier": "alice", "password": "wrong-password"})

        with client.session_transaction() as sess:
            assert "user_id" not in sess

    def test_unverified_user_cannot_log_in(self, client, post_form, make_user):
        make_user("alice", verified=False)
        post_form("/login", {"identifier": "alice", "password": "password123"})

        with client.session_transaction() as sess:
            assert "user_id" not in sess

    def test_logout_clears_session(self, client, login, make_user):
        login(make_user())
        client.get("/logout")

        with client.session_transaction() as sess:
            assert "user_id" not in sess

    def test_protected_page_redirects_to_login(self, client):
        res = client.get("/stats")

        assert res.status_code == 302
        assert "/login" in res.headers["Location"]


class TestEmailVerification:
    def test_valid_token_verifies_user(self, client, make_user):
        user = make_user("alice", verified=False)
        tok = EmailVerificationToken.generate(user.id)
        db.session.commit()

        client.get(f"/verify-email/{tok.token}")

        db.session.refresh(user)
        db.session.refresh(tok)
        assert user.email_verified is True
        assert tok.used is True

    def test_token_cannot_be_reused(self, make_user):
        user = make_user("alice", verified=False)
        tok = EmailVerificationToken.generate(user.id)
        tok.used = True
        db.session.commit()

        assert tok.is_valid() is False

    def test_expired_token_is_rejected(self, client, make_user):
        user = make_user("alice", verified=False)
        tok = EmailVerificationToken.generate(user.id)
        tok.created_at = datetime.utcnow() - timedelta(hours=25)
        db.session.commit()

        client.get(f"/verify-email/{tok.token}")

        db.session.refresh(user)
        assert user.email_verified is False


class TestPasswordReset:
    def test_forgot_password_creates_token(self, post_form, make_user):
        user = make_user("alice")
        post_form("/forgot-password", {"email": "alice@example.com"})

        assert PasswordResetToken.query.filter_by(user_id=user.id).count() == 1

    def test_forgot_password_unknown_email_gives_same_response(self, post_form):
        res = post_form("/forgot-password", {"email": "nobody@example.com"})

        assert res.status_code == 302
        assert PasswordResetToken.query.count() == 0

    def test_reset_changes_password(self, post_form, make_user):
        user = make_user("alice")
        tok = PasswordResetToken.generate(user.id)
        db.session.commit()

        post_form(f"/reset-password/{tok.token}", {"password": "brand-new-pass", "confirm": "brand-new-pass"})

        db.session.refresh(user)
        assert check_password_hash(user.password_hash, "brand-new-pass")
        db.session.refresh(tok)
        assert tok.used is True

    def test_reset_token_expires_after_one_hour(self, post_form, make_user):
        user = make_user("alice")
        tok = PasswordResetToken.generate(user.id)
        tok.created_at = datetime.utcnow() - timedelta(minutes=61)
        db.session.commit()

        post_form(f"/reset-password/{tok.token}", {"password": "brand-new-pass", "confirm": "brand-new-pass"})

        db.session.refresh(user)
        assert check_password_hash(user.password_hash, "password123")


class TestProfile:
    def test_change_username(self, client, login, post_form, make_user):
        user = make_user("alice")
        login(user)

        post_form("/profile", {"action": "username", "username": "alice_new"})

        db.session.refresh(user)
        assert user.username == "alice_new"
        with client.session_transaction() as sess:
            assert sess["username"] == "alice_new"

    def test_cannot_take_existing_username(self, login, post_form, make_user):
        user = make_user("alice")
        make_user("bob")
        login(user)

        post_form("/profile", {"action": "username", "username": "bob"})

        db.session.refresh(user)
        assert user.username == "alice"

    def test_change_password_requires_current_password(self, login, post_form, make_user):
        user = make_user("alice")
        login(user)
        form = {"action": "password", "new_password": "new-password-1", "confirm_password": "new-password-1"}

        post_form("/profile", {**form, "current_password": "wrong"})
        db.session.refresh(user)
        assert check_password_hash(user.password_hash, "password123")

        post_form("/profile", {**form, "current_password": "password123"})
        db.session.refresh(user)
        assert check_password_hash(user.password_hash, "new-password-1")

    @pytest.mark.parametrize(("value", "expected"), [("45", 45), ("0", None), ("abc", None)])
    def test_set_daily_goal(self, login, post_form, make_user, value, expected):
        user = make_user("alice")
        login(user)

        post_form("/profile", {"action": "goal", "daily_goal_minutes": value})

        db.session.refresh(user)
        assert user.daily_goal_minutes == expected

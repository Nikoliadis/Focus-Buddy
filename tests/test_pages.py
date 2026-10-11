from __future__ import annotations

import pytest


@pytest.mark.parametrize("path", ["/", "/login", "/register", "/forgot-password", "/resend-verification"])
def test_public_pages_render(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/rooms", "/rooms/create", "/rooms/join", "/stats", "/leaderboard", "/profile"])
def test_logged_in_pages_render(client, login, make_user, path):
    login(make_user())
    assert client.get(path).status_code == 200


def test_custom_404_page(client):
    res = client.get("/does-not-exist")
    assert res.status_code == 404


def test_logged_in_user_is_redirected_away_from_login(client, login, make_user):
    login(make_user())
    assert client.get("/login").status_code == 302

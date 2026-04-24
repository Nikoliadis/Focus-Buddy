import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    # Sessions
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # (enable in production only)
    SESSION_COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"

    # DB — Railway gives postgresql:// or postgres://, psycopg3 needs postgresql+psycopg://
    _db_url = os.getenv("DATABASE_URL", "")
    if _db_url.startswith("postgres://"):
        _db_url = "postgresql+psycopg://" + _db_url[len("postgres://"):]
    elif _db_url.startswith("postgresql://"):
        _db_url = "postgresql+psycopg://" + _db_url[len("postgresql://"):]
    SQLALCHEMY_DATABASE_URI = _db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False
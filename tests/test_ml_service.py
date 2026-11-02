"""Runs tests_service/ in a clean process (see the docstring there)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parent.parent


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "GROQ_API_KEY")}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=400)


def test_service_suite_in_a_clean_process():
    result = _run("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests_service")
    assert result.returncode == 0, result.stdout[-4000:]


def test_the_service_never_applies_gevent_patching():
    code = "import ml_api.main; from gevent import monkey; print(monkey.is_module_patched('socket'))"
    result = _run("-W", "ignore", "-c", code)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("False")


def test_the_web_app_does_not_import_torch_or_the_service():
    code = ("import os, sys; os.environ['DATABASE_URL'] = 'sqlite://'; import app; "
            "print(sorted(m for m in ('torch', 'sentence_transformers', 'ml_api', 'sklearn') if m in sys.modules))")
    result = _run("-W", "ignore", "-c", code)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("[]")

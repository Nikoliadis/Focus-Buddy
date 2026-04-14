from flask import render_template, session

from main.auth_utils import login_required
from main.stats_service import get_user_stats
from . import main_bp


@main_bp.get("/")
def home():
    return render_template("home.html")


@main_bp.get("/stats")
@login_required
def stats():
    data = get_user_stats(session["user_id"])
    return render_template("stats.html", **data)

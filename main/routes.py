from flask import render_template
from . import main_bp


@main_bp.get("/")
def home():
    return render_template("home.html")

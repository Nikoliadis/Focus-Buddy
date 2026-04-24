from gevent import monkey
monkey.patch_all()

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime

from flask import Flask, render_template, abort, request

from config import Config
from main.db import db
from main.socketio_ext import socketio
from main.limiter_ext import limiter
from main.csrf import generate_csrf_token, validate_csrf_token

from main import main_bp
from auth import auth_bp
from rooms import rooms_bp

import rooms.sockets

from models.user import User
from models.focus import FocusSession, FocusLog
from models.chat import ChatMessage
from models.todo import TodoItem


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # init extensions
    db.init_app(app)
    socketio.init_app(app)
    limiter.init_app(app)

    # blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(rooms_bp)

    # ------------------------------------------------------------------ #
    # CSRF protection — validate token on every POST (skip SocketIO path) #
    # ------------------------------------------------------------------ #
    @app.before_request
    def csrf_protect():
        if request.method == "POST":
            if request.path.startswith("/socket.io/"):
                return
            if not validate_csrf_token():
                abort(403)

    # ------------------------------------------------------------------ #
    # Template globals                                                     #
    # ------------------------------------------------------------------ #
    @app.context_processor
    def inject_globals():
        return {
            "year": datetime.now().year,
            "csrf_token": generate_csrf_token,
            "STATIC_VERSION": "4",
        }

    # ------------------------------------------------------------------ #
    # Error handlers                                                       #
    # ------------------------------------------------------------------ #
    @app.errorhandler(400)
    def bad_request(_e):
        return render_template("errors/400.html"), 400

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(429)
    def too_many_requests(_e):
        return render_template("errors/429.html"), 429

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("errors/500.html"), 500

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
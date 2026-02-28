from gevent import monkey
monkey.patch_all()

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime

from flask import Flask

from config import Config
from main.db import db
from main.socketio_ext import socketio

from main import main_bp
from auth import auth_bp
from rooms import rooms_bp

import rooms.sockets

from models.user import User
from models.focus import FocusSession, FocusLog 


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # init extensions
    db.init_app(app)
    socketio.init_app(app)

    # blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(rooms_bp)

    @app.context_processor
    def inject_globals():
        return {"year": datetime.now().year}

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
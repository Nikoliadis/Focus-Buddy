from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from flask import Flask
from config import Config
from main.db import db
from main import main_bp
from models.user import User 



def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    @app.context_processor
    def inject_globals():
        return {"year": datetime.now().year}

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)

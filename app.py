from datetime import datetime
from flask import Flask
from config import Config

from main import main_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    @app.context_processor
    def inject_globals():
        return {"year": datetime.now().year}

    app.register_blueprint(main_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)

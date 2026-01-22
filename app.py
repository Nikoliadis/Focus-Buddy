from datetime import datetime
from flask import Flask, render_template
from config import Config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    @app.context_processor
    def inject_globals():
        return {"year": datetime.now().year}

    @app.get("/")
    def home():
        return render_template("home.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)

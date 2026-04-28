from main.db import db

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)

    password_hash = db.Column(db.String(255), nullable=False)
    email_verified = db.Column(db.Boolean, nullable=False, default=False, server_default="true")
    daily_goal_minutes = db.Column(db.Integer, nullable=True)  # None = no goal set

    def __repr__(self) -> str:
        return f"<User {self.username}>"

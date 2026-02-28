from flask import Blueprint
import rooms.sockets  # noqa

rooms_bp = Blueprint("rooms", __name__)

from . import routes  # noqa
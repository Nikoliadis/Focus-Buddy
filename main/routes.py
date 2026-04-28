import csv
import io
import os

from flask import render_template, session, Response, request, redirect, url_for, jsonify

from main.auth_utils import login_required
from main.db import db
from main.limiter_ext import limiter
from main.stats_service import get_user_stats, get_leaderboard
from models.focus import FocusLog
from models.todo import TodoItem
from rooms.models import Room
from . import main_bp

_AI_SYSTEM = (
    "You are FocusBot, a friendly study assistant inside FocusBuddy — a real-time focus and study app. "
    "Help users with study questions, productivity tips, exam prep, and anything related to learning. "
    "Be concise, encouraging, and use simple language. Keep responses under 200 words unless more detail is needed."
)


@main_bp.get("/")
def home():
    return render_template("home.html")


@main_bp.get("/stats")
@login_required
def stats():
    data = get_user_stats(session["user_id"])
    return render_template("stats.html", **data)


@main_bp.get("/stats/export")
@login_required
def stats_export():
    rows = (
        db.session.query(FocusLog, Room.name)
        .join(Room, Room.id == FocusLog.room_id)
        .filter(FocusLog.user_id == session["user_id"])
        .order_by(FocusLog.created_at.desc())
        .all()
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "room", "duration_seconds", "duration_minutes"])
    for log, room_name in rows:
        writer.writerow([
            log.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            room_name,
            log.focused_seconds,
            round(log.focused_seconds / 60, 1),
        ])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=focus_stats.csv"},
    )


@main_bp.get("/leaderboard")
@login_required
def leaderboard():
    entries = get_leaderboard()
    return render_template("leaderboard.html", entries=entries)


# ── Todo JSON API (used by the in-room widget) ────────────────────────────────

@main_bp.get("/api/todos")
@login_required
def api_todos_list():
    items = (
        TodoItem.query
        .filter_by(user_id=session["user_id"])
        .order_by(TodoItem.created_at.asc())
        .all()
    )
    return jsonify([
        {"id": i.id, "body": i.body, "done": i.done, "is_public": i.is_public}
        for i in items
    ])


@main_bp.post("/api/todos")
@login_required
def api_todos_create():
    data = request.get_json(silent=True) or {}
    body = (data.get("body") or "").strip()
    if not body or len(body) > 300:
        return jsonify({"error": "Invalid body"}), 400
    item = TodoItem(
        user_id=session["user_id"],
        body=body,
        is_public=bool(data.get("is_public", False)),
    )
    db.session.add(item)
    db.session.commit()
    return jsonify({"id": item.id, "body": item.body, "done": item.done, "is_public": item.is_public})


@main_bp.post("/api/todos/<int:item_id>/toggle")
@login_required
def api_todos_toggle(item_id: int):
    item = TodoItem.query.filter_by(id=item_id, user_id=session["user_id"]).first_or_404()
    item.done = not item.done
    db.session.commit()
    return jsonify({"id": item.id, "done": item.done})


@main_bp.post("/api/todos/<int:item_id>/delete")
@login_required
def api_todos_delete(item_id: int):
    item = TodoItem.query.filter_by(id=item_id, user_id=session["user_id"]).first_or_404()
    db.session.delete(item)
    db.session.commit()
    return jsonify({"ok": True})


# ── AI Chat ───────────────────────────────────────────────────────────────────

@main_bp.post("/api/ai/chat")
@login_required
@limiter.limit("20 per minute")
def api_ai_chat():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return jsonify({"error": "AI not configured"}), 503

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message or len(message) > 1000:
        return jsonify({"error": "Invalid message"}), 400

    history = data.get("history") or []
    # Keep last 10 exchanges to limit token usage
    history = history[-10:]
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    messages.append({"role": "user", "content": message})

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=512,
            messages=[{"role": "system", "content": _AI_SYSTEM}] + messages,
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": "AI error", "detail": str(e)}), 500

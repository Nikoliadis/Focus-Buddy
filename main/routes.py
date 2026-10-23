import csv
import io
import os
from functools import lru_cache

from flask import Response, current_app, jsonify, render_template, request, session

from main.auth_utils import login_required
from main.db import db
from main.limiter_ext import limiter
from main.stats_service import get_leaderboard, get_user_stats
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

    history = data.get("history") if isinstance(data.get("history"), list) else []
    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in history[-10:]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]

    if rag_enabled():
        rag = _rag_pipeline()
        if rag is not None:
            from ai.llm.client import LLMUnavailable

            try:
                return jsonify(rag.answer(message, history).to_json())
            except LLMUnavailable:
                current_app.logger.exception("FocusBot RAG request failed")
                return jsonify({"error": "AI error"}), 500

    messages = [*history, {"role": "user", "content": message}]
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
    except Exception:
        # Log the cause server-side; never send provider error details to the browser.
        current_app.logger.exception("FocusBot request failed")
        return jsonify({"error": "AI error"}), 500


def rag_enabled() -> bool:
    """RAG needs sentence-transformers/torch, which the web image does not install by default."""
    return os.getenv("FOCUSBOT_RAG") == "1"


@lru_cache(maxsize=1)
def _rag_pipeline():
    """Built once per process. Loading the embedding model takes seconds and ~500 MB of RAM,
    which is why Phase 19 moves this into a separate ML service. Returns None (plain FocusBot)
    if the ML dependencies are not installed."""
    try:
        from ai.llm.client import GroqLLM
        from ai.rag.pipeline import RAGPipeline
        from ai.rag.retriever import KnowledgeRetriever
        from nlp.embeddings import SentenceTransformerEmbedder
    except ImportError:
        current_app.logger.warning("FOCUSBOT_RAG=1 but ML dependencies are missing; using plain FocusBot")
        return None
    return RAGPipeline(KnowledgeRetriever(SentenceTransformerEmbedder(device="cpu")), GroqLLM())

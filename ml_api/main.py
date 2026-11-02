"""FocusBuddy ML service: inference over HTTP, in its own process.

Why a separate service: the web app runs on gevent, where CPU work (embeddings, forests) would
stall every WebSocket, and torch would add gigabytes to its image. This service only serves.
It loads trained artifacts once at startup and never trains.

Endpoints are plain ``def``: FastAPI runs them in a thread pool, which is right for blocking
work such as model inference and LLM calls. ``async def`` would block the event loop.

It is an internal service. Every endpoint except /health requires X-API-Key, and the service
refuses all of them if no key is configured. The caller vouches for ``user_id``, so the port
must not be exposed publicly; docker-compose publishes none.

Run:
    ML_API_KEY=... uvicorn ml_api.main:create_app --factory --port 8000
    Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ai.llm.client import LLMNotConfigured, LLMRateLimited, LLMTimeout, LLMUnavailable
from ai.llm.telemetry import configure_logging
from ml_api import schemas as s
from ml_api.registry import LoadedModel, ModelRegistry
from nlp.preprocessing import redact, tokenize

logger = logging.getLogger("focusbuddy.ml_api")

DEFAULT_ARTIFACTS = Path("artifacts")


@dataclass
class Services:
    """Everything endpoints use, built once. Tests construct their own."""

    registry: ModelRegistry
    api_key: str = ""
    embedder: Any = None
    rag: Any = None
    engine: Engine | None = None
    data_scope: Callable | None = None  # database context for tools over the user's own data
    errors: dict[str, str] = field(default_factory=dict)


def _data_access(url: str) -> tuple[Engine, Callable]:
    """Read-only database access. Tools use the app's Flask-SQLAlchemy models, which need an app
    context, so a minimal Flask app (no routes, no create_all) provides one."""
    from flask import Flask

    from analytics.quality import make_readonly_engine
    from main.db import db

    flask_app = Flask("focusbuddy_ml")
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = url
    if url.startswith("postgresql"):
        flask_app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "connect_args": {"options": "-c default_transaction_read_only=on"}}
    db.init_app(flask_app)
    return make_readonly_engine(url), flask_app.app_context


def build_services() -> Services:
    """Production wiring. A missing optional part is recorded and its endpoints answer 503."""
    from dotenv import load_dotenv

    configure_logging()
    load_dotenv()
    from config import Config  # after load_dotenv

    services = Services(
        registry=ModelRegistry(Path(os.getenv("ML_ARTIFACTS_DIR", DEFAULT_ARTIFACTS))).load_all(),
        api_key=os.getenv("ML_API_KEY", ""),
    )
    if Config.SQLALCHEMY_DATABASE_URI:
        try:
            services.engine, services.data_scope = _data_access(Config.SQLALCHEMY_DATABASE_URI)
        except Exception:  # noqa: BLE001
            logger.exception("database access is unavailable")
            services.errors["database"] = "unavailable"
    try:
        from ai.llm.client import GroqLLM
        from ai.rag.pipeline import RAGPipeline
        from ai.rag.retriever import KnowledgeRetriever
        from nlp.embeddings import SentenceTransformerEmbedder

        services.embedder = SentenceTransformerEmbedder()
        # Embeds the knowledge base now, so the first user does not wait for the model to load.
        services.rag = RAGPipeline(KnowledgeRetriever(services.embedder), GroqLLM())
    except Exception:  # noqa: BLE001
        logger.exception("the RAG pipeline is unavailable")
        services.errors["rag"] = "unavailable"
    return services


def search_available(services: Services) -> bool:
    """pgvector exists only on PostgreSQL."""
    return (services.embedder is not None and services.engine is not None
            and services.engine.dialect.name == "postgresql")


def _services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(_services)]


def require_api_key(services: ServicesDep, x_api_key: Annotated[str, Header()] = "") -> None:
    if not services.api_key:
        raise HTTPException(503, "ML_API_KEY is not configured")
    if not hmac.compare_digest(x_api_key.encode(), services.api_key.encode()):
        raise HTTPException(401, "invalid API key")


def _require_model(services: Services, task: str) -> LoadedModel:
    loaded = services.registry.models.get(task)
    if loaded is None:
        raise HTTPException(503, f"no {task} model is available")
    return loaded


def _info(loaded: LoadedModel) -> s.ModelInfo:
    return s.ModelInfo(name=loaded.name, version=loaded.version, data_source=loaded.data_source)


def rooms_of_user(engine: Engine, user_id: int) -> list[int]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT room_id FROM room_members WHERE user_id = :u"), {"u": user_id})
        return [int(r[0]) for r in rows]


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        # Loaded at startup, once. Nothing here trains, and predictions only read.
        app.state.services = services if services is not None else build_services()
        yield

    app = FastAPI(title="FocusBuddy ML service", version="1.0.0", lifespan=lifespan)
    guarded = [Depends(require_api_key)]

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Method, path, status and latency only: bodies contain user text."""
        started = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response

    @app.exception_handler(LLMUnavailable)
    async def llm_error(_request: Request, exc: LLMUnavailable) -> JSONResponse:
        status = {LLMRateLimited: 429, LLMTimeout: 504, LLMNotConfigured: 503}.get(type(exc), 502)
        logger.warning("LLM failure: %s", type(exc).__name__)
        return JSONResponse({"detail": "the language model is unavailable"}, status_code=status)

    @app.get("/health", response_model=s.HealthResponse)
    def health(services: ServicesDep) -> s.HealthResponse:
        """Open, for orchestrators, and it reveals no versions or error details."""
        loaded = sorted(services.registry.models)
        complete = len(loaded) == 3 and services.rag is not None
        return s.HealthResponse(status="ok" if complete else "degraded", models_loaded=loaded,
                                rag_ready=services.rag is not None,
                                search_ready=search_available(services))

    @app.get("/models", response_model=s.ModelsResponse, dependencies=guarded)
    def models(services: ServicesDep) -> s.ModelsResponse:
        registry = services.registry
        return s.ModelsResponse(
            models=[s.ModelSummary(task=m.task, name=m.name, version=m.version, data_source=m.data_source,
                                   sklearn_version=m.metadata.get("sklearn_version"),
                                   sklearn_matches=m.sklearn_matches, notes=registry.notes(m))
                    for m in registry.models.values()],
            errors=registry.errors,
        )

    @app.post("/predict/productivity", response_model=s.ProductivityPrediction, dependencies=guarded)
    def predict_productivity(body: s.SessionFeatures, services: ServicesDep) -> s.ProductivityPrediction:
        loaded = _require_model(services, "regression")
        return s.ProductivityPrediction(predicted_focused_minutes=float(loaded.model.predict(body.to_frame())[0]),
                                        model=_info(loaded))

    @app.post("/predict/session-completion", response_model=s.CompletionPrediction, dependencies=guarded)
    def predict_completion(body: s.SessionFeatures, services: ServicesDep) -> s.CompletionPrediction:
        loaded = _require_model(services, "classification")
        probability = float(loaded.model.predict_proba(body.to_frame())[0, 1])
        threshold = float(loaded.metadata["threshold"])
        return s.CompletionPrediction(probability_ended_early=probability,
                                      predicted_ended_early=probability >= threshold,
                                      threshold=threshold, model=_info(loaded))

    @app.post("/cluster/user", response_model=s.ClusterAssignment, dependencies=guarded)
    def cluster_user(body: s.UserBehaviour, services: ServicesDep) -> s.ClusterAssignment:
        loaded = _require_model(services, "clustering")
        cluster = int(loaded.model.predict(body.to_frame())[0])
        description = loaded.metadata.get("descriptions", {}).get(str(cluster))
        return s.ClusterAssignment(cluster=cluster, description=description,
                                   weak_structure=bool(loaded.metadata.get("weak_structure", False)),
                                   model=_info(loaded))

    @app.post("/nlp/analyze", response_model=s.AnalyzeResponse, dependencies=guarded)
    def nlp_analyze(body: s.AnalyzeRequest) -> s.AnalyzeResponse:
        redacted = redact(body.text)
        return s.AnalyzeResponse(redacted_text=" ".join(redacted.split()),
                                 contains_personal_data=redacted != body.text, tokens=tokenize(body.text))

    @app.post("/embeddings/search", response_model=s.SearchResponse, dependencies=guarded)
    def embeddings_search(body: s.SearchRequest, services: ServicesDep) -> s.SearchResponse:
        if not search_available(services):
            raise HTTPException(503, "semantic search is not available")
        from ai.retrieval import store

        try:
            if services.embedder.name not in set(store.stats(services.engine)["model"]):
                raise HTTPException(503, "the search index has not been built")
            # The database decides which rooms are allowed; room_ids can only narrow them.
            allowed = set(rooms_of_user(services.engine, body.user_id))
            rooms = sorted(allowed & set(body.room_ids)) if body.room_ids is not None else sorted(allowed)
            hits = store.search_text(services.engine, services.embedder, body.query, rooms, body.k)
        except SQLAlchemyError as exc:
            logger.warning("search failed: %s", type(exc).__name__)
            raise HTTPException(503, "semantic search is not available") from exc
        return s.SearchResponse(hits=[
            s.SearchHit(message_id=int(h.message_id), room_id=int(h.room_id), body=" ".join(redact(h.body).split()),
                        similarity=float(h.similarity))
            for h in hits.itertuples()
        ])

    @app.post("/rag/query", response_model=s.RagResponse, dependencies=guarded)
    def rag_query(body: s.RagRequest, services: ServicesDep) -> s.RagResponse:
        if services.rag is None:
            raise HTTPException(503, "the RAG pipeline is not available")
        tool_context = None
        if body.user_id is not None and services.data_scope is not None:
            from ai.rag.tools import ToolContext

            tool_context = ToolContext(body.user_id, scope=services.data_scope)
        history = [t.model_dump() for t in body.history]
        return s.RagResponse(**services.rag.answer(body.question, history, tool_context=tool_context).to_json())

    return app

#backend/app.py
from __future__ import annotations

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.pipeline.expedition_pipeline import ExpeditionPipeline
from backend.services.chat_memory_repository import ChatMemoryRepository
from backend.services.context_chat_service import ContextAwareChatService
from backend.services.expedition_repository import ExpeditionRepository

app = Flask(__name__)
CORS(app)

pipeline = ExpeditionPipeline()
repo = ExpeditionRepository()
chat_repo = ChatMemoryRepository()
chat_service = ContextAwareChatService(repo, chat_repo)

DEFAULT_EMAIL = os.getenv("DEFAULT_USER_EMAIL", "guest@mars.local")
DEFAULT_NAME = os.getenv("DEFAULT_USER_NAME", "Local User")


def _identity_from_request():
    data = request.get_json(silent=True) or {}
    email = (
        request.headers.get("X-User-Email")
        or data.get("user_email")
        or request.args.get("user_email")
        or DEFAULT_EMAIL
    )
    display_name = (
        request.headers.get("X-User-Name")
        or data.get("display_name")
        or request.args.get("display_name")
        or DEFAULT_NAME
    )
    return str(email).strip(), str(display_name).strip()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/research")
def research():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "") or "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400

    user_email, display_name = _identity_from_request()
    existing_expedition_id = data.get("expedition_id")

    result = pipeline.run(query)

    saved = repo.save_expedition(
        user_email=user_email,
        display_name=display_name,
        expedition=result,
        expedition_id=existing_expedition_id,
    )

    result["expedition_id"] = str(saved["id"])
    result["saved_to_db"] = True
    result["user_email"] = user_email
    result["display_name"] = display_name
    return jsonify(result)


@app.post("/expeditions/save")
def save_expedition():
    data = request.get_json(silent=True) or {}
    expedition = data.get("expedition", {})
    expedition_id = data.get("expedition_id")

    if not isinstance(expedition, dict):
        return jsonify({"error": "expedition must be an object"}), 400

    user_email, display_name = _identity_from_request()

    saved = repo.save_expedition(
        user_email=user_email,
        display_name=display_name,
        expedition=expedition,
        expedition_id=expedition_id,
    )

    return jsonify({"ok": True, "expedition_id": str(saved["id"]), "user_email": user_email})


@app.get("/expeditions")
def list_expeditions():
    user_email, _ = _identity_from_request()
    items = repo.list_expeditions(user_email)
    return jsonify(items)


@app.get("/expeditions/<expedition_id>")
def get_expedition(expedition_id: str):
    try:
        return jsonify(repo.get_expedition_detail(expedition_id))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.delete("/expeditions/<expedition_id>")
def delete_expedition(expedition_id: str):
    try:
        repo.delete_expedition(expedition_id)
        return jsonify({"ok": True})
    except Exception as exc:
        app.logger.exception("Delete failed")
        return jsonify({"error": str(exc)}), 500


@app.get("/expeditions/<expedition_id>/chat")
def get_chat_memory(expedition_id: str):
    try:
        messages = chat_repo.list_messages(expedition_id)
        return jsonify(messages)
    except Exception as exc:
        app.logger.exception("Chat memory load failed")
        return jsonify({"error": str(exc)}), 500


@app.post("/expeditions/<expedition_id>/chat")
def chat(expedition_id: str):
    data = request.get_json(silent=True) or {}
    question = str(data.get("message", "") or "").strip()
    if not question:
        return jsonify({"error": "message required"}), 400

    try:
        result = chat_service.ask(expedition_id, question)
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Chat failed")
        return jsonify({"error": str(exc)}), 500
    
from backend.services.cfg_service import CFGService

try:
    cfg_service = CFGService()
except Exception as exc:
    cfg_service = None
    app.logger.warning(f"CFG service disabled: {exc}")


def _sync_cfg_from_expedition(expedition_id: str):
    if not cfg_service:
        return
    try:
        detail = repo.get_expedition_detail(expedition_id)
        cfg_service.sync_from_detail(detail)
    except Exception as exc:
        app.logger.warning(f"CFG sync failed for {expedition_id}: {exc}")


@app.get("/cfg/<expedition_id>")
def get_cfg(expedition_id: str):
    if not cfg_service:
        return jsonify({
            "expedition_id": expedition_id,
            "ready": False,
            "nodes": [],
            "links": [],
            "learn_next": [],
            "trends": [],
            "forgotten_curve": [],
            "message": "CFG service not configured"
        }), 200

    try:
        graph = cfg_service.get_graph(expedition_id)

        if not graph:
            graph = {}

        graph.setdefault("expedition_id", expedition_id)
        graph.setdefault("ready", bool(graph.get("nodes")))
        graph.setdefault("nodes", [])
        graph.setdefault("links", [])
        graph.setdefault("learn_next", [])
        graph.setdefault("trends", [])
        graph.setdefault("forgotten_curve", [])

        if not graph["ready"]:
            graph["message"] = "CFG not built yet"

        return jsonify(graph), 200

    except Exception as exc:
        app.logger.warning(f"CFG not ready for {expedition_id}: {exc}")
        return jsonify({
            "expedition_id": expedition_id,
            "ready": False,
            "nodes": [],
            "links": [],
            "learn_next": [],
            "trends": [],
            "forgotten_curve": [],
            "message": "CFG not built yet"
        }), 200


@app.post("/cfg/<expedition_id>/rebuild")
def rebuild_cfg(expedition_id: str):
    if not cfg_service:
        return jsonify({"error": "CFG service not configured"}), 503
    try:
        detail = repo.get_expedition_detail(expedition_id)
        cfg_service.sync_from_detail(detail)
        return jsonify(cfg_service.get_graph(expedition_id)), 200
    except Exception as exc:
        app.logger.exception("CFG rebuild failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
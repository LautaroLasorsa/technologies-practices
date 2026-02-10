"""Task Tracker API — minimal Flask app for Kubernetes practice."""

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request

app = Flask(__name__)

# Configuration from environment variables (injected via ConfigMap / Secret)
APP_ENV = os.environ.get("APP_ENV", "development")
APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
SECRET_API_KEY = os.environ.get("SECRET_API_KEY", "not-set")
SECRET_DB_PASSWORD = os.environ.get("SECRET_DB_PASSWORD", "not-set")

# In-memory task store (resets on pod restart — intentional for learning)
tasks: dict[str, dict] = {}


@app.route("/health", methods=["GET"])
def health():
    """Liveness/readiness probe endpoint."""
    return jsonify({
        "status": "healthy",
        "version": APP_VERSION,
        "environment": APP_ENV,
        "log_level": LOG_LEVEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/config", methods=["GET"])
def config():
    """Show injected configuration (for verifying ConfigMap/Secret mounts)."""
    return jsonify({
        "APP_ENV": APP_ENV,
        "APP_VERSION": APP_VERSION,
        "LOG_LEVEL": LOG_LEVEL,
        "SECRET_API_KEY": f"{SECRET_API_KEY[:4]}****" if len(SECRET_API_KEY) > 4 else "****",
        "SECRET_DB_PASSWORD": "****" if SECRET_DB_PASSWORD != "not-set" else "not-set",
    })


@app.route("/tasks", methods=["GET"])
def list_tasks():
    """List all tasks."""
    return jsonify({"tasks": list(tasks.values()), "count": len(tasks)})


@app.route("/tasks", methods=["POST"])
def create_task():
    """Create a new task. Body: {"title": "...", "description": "..."}"""
    data = request.get_json(silent=True)
    if not data or "title" not in data:
        return jsonify({"error": "Missing required field: title"}), 400

    task_id = uuid.uuid4().hex[:8]
    task = {
        "id": task_id,
        "title": data["title"],
        "description": data.get("description", ""),
        "done": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pod": os.environ.get("HOSTNAME", "unknown"),
    }
    tasks[task_id] = task
    return jsonify(task), 201


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id: str):
    """Get a single task by ID."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": f"Task '{task_id}' not found"}), 404
    return jsonify(task)


@app.route("/tasks/<task_id>", methods=["PATCH"])
def update_task(task_id: str):
    """Toggle task completion. Body: {"done": true/false}"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": f"Task '{task_id}' not found"}), 404

    data = request.get_json(silent=True) or {}
    if "done" in data:
        task["done"] = bool(data["done"])
    if "title" in data:
        task["title"] = data["title"]
    return jsonify(task)


@app.route("/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id: str):
    """Delete a task by ID."""
    task = tasks.pop(task_id, None)
    if not task:
        return jsonify({"error": f"Task '{task_id}' not found"}), 404
    return jsonify({"deleted": task_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

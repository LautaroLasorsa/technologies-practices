"""TaskManager gRPC server.

Run with:
    cd practice_004a_grpc_python/app
    uv run python -m app.server

The server listens on localhost:50051 by default.
"""

from __future__ import annotations

import logging
import sys
from concurrent import futures
from pathlib import Path

import grpc

# -- Make sure the practice root is on sys.path so `app.generated` resolves ---
_practice_root = str(Path(__file__).resolve().parent.parent)
if _practice_root not in sys.path:
    sys.path.insert(0, _practice_root)

from app.generated import task_manager_pb2 as pb2
from app.generated import task_manager_pb2_grpc as pb2_grpc
from app.store import TaskStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = "localhost:50051"


# ──────────────────────────────────────────────────────────────────────────────
# Servicer implementation
# ──────────────────────────────────────────────────────────────────────────────

class TaskManagerServicer(pb2_grpc.TaskManagerServicer):
    """Implements the TaskManager service defined in task_manager.proto.

    The in-memory `TaskStore` is injected via the constructor (Dependency
    Inversion). Each RPC method receives:
      - `request`: the deserialized protobuf request message
      - `context`: a `grpc.ServicerContext` for setting status codes, metadata, etc.

    Error handling pattern:
      - Use `context.abort(grpc.StatusCode.XXX, "details")` to fail an RPC.
        This raises an exception internally and sends the status to the client.
      - Common codes: NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS, INTERNAL.
      - Full list: https://grpc.io/docs/guides/status-codes/
    """

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    # ── Unary RPCs ────────────────────────────────

    def CreateTask(
        self,
        request: pb2.CreateTaskRequest,
        context: grpc.ServicerContext,
    ) -> pb2.CreateTaskResponse:
        # TODO(human): Implement CreateTask.
        #
        # Steps:
        #   1. Validate: title must be non-empty. If empty, abort with
        #      grpc.StatusCode.INVALID_ARGUMENT and a descriptive message.
        #   2. Call self._store.create(title, description) to persist the task.
        #   3. Log the creation.
        #   4. Return a CreateTaskResponse wrapping the new task.
        #
        # Hint:
        #   context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Title is required")
        #   return pb2.CreateTaskResponse(task=task)
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "CreateTask not yet implemented")

    def GetTask(
        self,
        request: pb2.GetTaskRequest,
        context: grpc.ServicerContext,
    ) -> pb2.GetTaskResponse:
        # TODO(human): Implement GetTask.
        #
        # Steps:
        #   1. Call self._store.get(request.id)
        #   2. If None, abort with grpc.StatusCode.NOT_FOUND
        #   3. Return GetTaskResponse wrapping the task.
        #
        # Hint:
        #   context.abort(grpc.StatusCode.NOT_FOUND, f"Task '{request.id}' not found")
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "GetTask not yet implemented")

    def UpdateTask(
        self,
        request: pb2.UpdateTaskRequest,
        context: grpc.ServicerContext,
    ) -> pb2.UpdateTaskResponse:
        # TODO(human): Implement UpdateTask.
        #
        # Steps:
        #   1. Call self._store.update(request.id, request.title,
        #      request.description, request.status)
        #   2. If None (task not found), abort with NOT_FOUND.
        #   3. Log the update.
        #   4. Return UpdateTaskResponse wrapping the updated task.
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "UpdateTask not yet implemented")

    def DeleteTask(
        self,
        request: pb2.DeleteTaskRequest,
        context: grpc.ServicerContext,
    ) -> pb2.DeleteTaskResponse:
        # TODO(human): Implement DeleteTask.
        #
        # Steps:
        #   1. Call self._store.delete(request.id)
        #   2. If None (task not found), abort with NOT_FOUND.
        #   3. Log the deletion.
        #   4. Return DeleteTaskResponse wrapping the deleted task.
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "DeleteTask not yet implemented")

    # ── Server streaming RPC ──────────────────────

    def WatchTasks(
        self,
        request: pb2.WatchTasksRequest,
        context: grpc.ServicerContext,
    ):
        # TODO(human): Implement WatchTasks (server streaming).
        #
        # This is a *generator* — you `yield` TaskEvent messages one at a time.
        # The client receives them as an iterator.
        #
        # Steps:
        #   1. Subscribe to the store: q = self._store.subscribe()
        #   2. Use a try/finally to ensure unsubscribe on client disconnect.
        #   3. In a loop, pull events from the queue with a timeout (e.g., 1s).
        #      - q.get(timeout=1.0) blocks until an event arrives or times out.
        #      - On timeout (queue.Empty), just continue the loop.
        #      - On receiving an event, `yield` it.
        #   4. Check `context.is_active()` each iteration — if False, the client
        #      disconnected, so break out of the loop.
        #   5. In `finally`, call self._store.unsubscribe(q).
        #
        # Hint — the skeleton looks like:
        #   q = self._store.subscribe()
        #   try:
        #       while context.is_active():
        #           try:
        #               event = q.get(timeout=1.0)
        #               yield event
        #           except queue.Empty:
        #               continue
        #   finally:
        #       self._store.unsubscribe(q)
        #
        # Remember to `import queue` at the top of the file.
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "WatchTasks not yet implemented")


# ──────────────────────────────────────────────────────────────────────────────
# Server bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def create_server(store: TaskStore) -> grpc.Server:
    """Build and configure the gRPC server.

    Wires up:
      - Thread pool executor (handles concurrent RPCs)
      - Interceptors (logging, etc.)
      - Service registration
    """
    # TODO(human): Uncomment the interceptor import and add it to the server
    # once you've implemented LoggingInterceptor in interceptors.py.
    #
    # from app.interceptors import LoggingInterceptor
    # interceptors = [LoggingInterceptor()]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        # TODO(human): Pass interceptors here once implemented:
        # interceptors=interceptors,
    )

    servicer = TaskManagerServicer(store)
    pb2_grpc.add_TaskManagerServicer_to_server(servicer, server)

    server.add_insecure_port(SERVER_ADDRESS)
    return server


def serve() -> None:
    """Start the server and block until interrupted."""
    store = TaskStore()
    server = create_server(store)
    server.start()
    logger.info("TaskManager gRPC server listening on %s", SERVER_ADDRESS)
    logger.info("Press Ctrl+C to stop.")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=3).wait()
        logger.info("Server stopped.")


if __name__ == "__main__":
    serve()

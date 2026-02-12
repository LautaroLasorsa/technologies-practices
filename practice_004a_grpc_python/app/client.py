"""TaskManager gRPC client — exercises every RPC method.

Run with:
    cd practice_004a_grpc_python/app
    uv run python -m app.client

Make sure the server is running in another terminal first.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import grpc

# -- Make sure the practice root is on sys.path so `app.generated` resolves ---
_practice_root = str(Path(__file__).resolve().parent.parent)
if _practice_root not in sys.path:
    sys.path.insert(0, _practice_root)

from app.generated import task_manager_pb2 as pb2
from app.generated import task_manager_pb2_grpc as pb2_grpc

SERVER_ADDRESS = "localhost:50051"


# ──────────────────────────────────────────────────────────────────────────────
# Client helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_stub() -> tuple[grpc.Channel, pb2_grpc.TaskManagerStub]:
    """Create a channel + stub. Returns (channel, stub) so caller can close."""
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = pb2_grpc.TaskManagerStub(channel)
    return channel, stub


def print_task(label: str, task: pb2.Task) -> None:
    """Pretty-print a Task protobuf message."""
    status_name = pb2.TaskStatus.Name(task.status)
    print(f"  [{label}] id={task.id}  title={task.title!r}  "
          f"status={status_name}  desc={task.description!r}")


# ──────────────────────────────────────────────────────────────────────────────
# CRUD demonstrations
# ──────────────────────────────────────────────────────────────────────────────

def demo_create(stub: pb2_grpc.TaskManagerStub) -> str:
    """Create a task and return its ID."""
    print("\n--- CreateTask ---")
    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches making unary RPC calls from the client: building
    # protobuf request messages, calling stub methods, and accessing response fields.
    # It's the client-side counterpart to the server's CreateTask implementation.

    # TODO(human): Call stub.CreateTask() with a CreateTaskRequest.
    #
    # Steps:
    #   1. Build the request: pb2.CreateTaskRequest(title=..., description=...)
    #   2. Call: response = stub.CreateTask(request)
    #   3. Print the task with print_task("created", response.task)
    #   4. Return response.task.id
    #
    # Hint: Pick any title/description you like.
    raise NotImplementedError("TODO(human): implement demo_create")


def demo_get(stub: pb2_grpc.TaskManagerStub, task_id: str) -> None:
    """Fetch a task by ID."""
    print("\n--- GetTask ---")
    # TODO(human): Call stub.GetTask() with a GetTaskRequest.
    #
    # Steps:
    #   1. Build the request: pb2.GetTaskRequest(id=task_id)
    #   2. Call: response = stub.GetTask(request)
    #   3. Print with print_task("fetched", response.task)
    raise NotImplementedError("TODO(human): implement demo_get")


def demo_update(stub: pb2_grpc.TaskManagerStub, task_id: str) -> None:
    """Update a task's status to IN_PROGRESS."""
    print("\n--- UpdateTask ---")
    # TODO(human): Call stub.UpdateTask() with an UpdateTaskRequest.
    #
    # Steps:
    #   1. Build the request with the task's ID and new status:
    #      pb2.UpdateTaskRequest(
    #          id=task_id,
    #          status=pb2.TASK_STATUS_IN_PROGRESS,
    #      )
    #   2. Call: response = stub.UpdateTask(request)
    #   3. Print with print_task("updated", response.task)
    raise NotImplementedError("TODO(human): implement demo_update")


def demo_delete(stub: pb2_grpc.TaskManagerStub, task_id: str) -> None:
    """Delete a task by ID."""
    print("\n--- DeleteTask ---")
    # TODO(human): Call stub.DeleteTask() with a DeleteTaskRequest.
    #
    # Steps:
    #   1. Build the request: pb2.DeleteTaskRequest(id=task_id)
    #   2. Call: response = stub.DeleteTask(request)
    #   3. Print with print_task("deleted", response.task)
    raise NotImplementedError("TODO(human): implement demo_delete")


def demo_get_not_found(stub: pb2_grpc.TaskManagerStub) -> None:
    """Try to fetch a non-existent task — exercise error handling."""
    print("\n--- GetTask (NOT_FOUND) ---")
    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches client-side error handling: catching grpc.RpcError,
    # inspecting status codes, and extracting error details. This is critical for
    # building resilient clients that handle network failures, invalid requests, etc.

    # TODO(human): Handle the gRPC error.
    #
    # Steps:
    #   1. Call stub.GetTask with a fake ID (e.g., "nonexistent-id")
    #   2. Wrap the call in try/except grpc.RpcError as e:
    #   3. In the except block, print:
    #      - e.code()      → the grpc.StatusCode enum (e.g., NOT_FOUND)
    #      - e.details()   → the human-readable error message
    #
    # Hint:
    #   try:
    #       response = stub.GetTask(pb2.GetTaskRequest(id="nonexistent"))
    #   except grpc.RpcError as e:
    #       print(f"  gRPC error: code={e.code()}, details={e.details()}")
    raise NotImplementedError("TODO(human): implement demo_get_not_found")


# ──────────────────────────────────────────────────────────────────────────────
# Streaming demonstration
# ──────────────────────────────────────────────────────────────────────────────

def demo_watch(stub: pb2_grpc.TaskManagerStub) -> None:
    """Subscribe to WatchTasks in a background thread, then do some CRUD.

    This demonstrates server streaming: the client starts the stream,
    then performs mutations in the main thread. The background thread
    prints events as they arrive.
    """
    print("\n--- WatchTasks (server streaming) ---")
    print("  Starting watcher in background thread...")

    stop_event = threading.Event()

    def watch_loop() -> None:
        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches consuming server-streaming RPCs: iterating over an
        # iterator of responses from the server. This is the client-side pattern for
        # real-time feeds, progress updates, or any scenario where the server streams events.

        # TODO(human): Implement the streaming consumer.
        #
        # Steps:
        #   1. Call stub.WatchTasks(pb2.WatchTasksRequest()) — this returns
        #      an *iterator* of TaskEvent messages.
        #   2. Iterate with `for event in response_iterator:`
        #   3. For each event, print the event type and task info.
        #   4. Wrap in try/except grpc.RpcError to handle cancellation.
        #   5. Set stop_event when done.
        #
        # Hint:
        #   try:
        #       for event in stub.WatchTasks(pb2.WatchTasksRequest()):
        #           event_name = pb2.TaskEventType.Name(event.event_type)
        #           print(f"  [WATCH] {event_name}: {event.task.title!r} "
        #                 f"(id={event.task.id})")
        #   except grpc.RpcError as e:
        #       if e.code() != grpc.StatusCode.CANCELLED:
        #           print(f"  [WATCH] error: {e.code()} - {e.details()}")
        #   finally:
        #       stop_event.set()
        stop_event.set()  # Remove this line once you implement the loop
        raise NotImplementedError("TODO(human): implement watch_loop")

    watcher = threading.Thread(target=watch_loop, daemon=True)
    watcher.start()

    # Give the watcher a moment to connect
    import time
    time.sleep(0.5)

    # Perform some CRUD — the watcher should print events for each
    print("  Performing CRUD operations (watcher should print events)...")
    task_id = demo_create(stub)
    demo_update(stub, task_id)
    demo_delete(stub, task_id)

    # Wait a bit for events to arrive, then we're done
    time.sleep(1.0)
    print("  Watcher demo complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    channel, stub = make_stub()
    try:
        # Phase 2: Unary CRUD
        task_id = demo_create(stub)
        demo_get(stub, task_id)
        demo_update(stub, task_id)
        demo_delete(stub, task_id)

        # Phase 4: Error handling
        demo_get_not_found(stub)

        # Phase 3: Server streaming (uncomment when WatchTasks is implemented)
        # demo_watch(stub)

    finally:
        channel.close()

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()

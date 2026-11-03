"""FastAPI Backend for Practice 116: Real-Time & Streaming UI.

Provides:
1. POST SSE endpoint (/api/solver/stream-reasoning) with chunk-fragmentation and
   client-disconnect detection (Exercise 4, Part A).
2. Bidirectional WebSocket (/ws/telemetry) emitting high-frequency solver telemetry
   (10Hz - 500Hz) with heartbeat ping/pong support (Exercise 4, Part B).
"""

import asyncio
import json
import math
import random
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Practice 116 Real-Time Solver Backend")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolverPromptRequest(BaseModel):
    prompt: str = "Solve TSP-100 instance with simulated annealing"
    problem_type: str = "Traveling Salesperson"
    simulated_delay_ms: int = 40


# Pre-baked reasoning tokens simulating a complex optimization solver
REASONING_STEPS = [
    "[Phase 1: Ingestion & Problem Formulation]",
    "Parsing graph topology: 100 vertices, 4,950 fully connected Euclidean edges.",
    "Formulating Miller-Tucker-Zemlin (MTZ) subtour elimination constraints.",
    "Computing 2-opt initial heuristic tour length: 1,428.65 km.",
    "[Phase 2: Linear Programming Relaxation]",
    "Constructing sparse constraint matrix: A_eq rank 100, A_ub rank 4851.",
    "Executing Primal Simplex on relaxed LP tableau...",
    "Pivot 1,240: lower bound objective = 982.14 km.",
    "Duality gap at root node: 31.25%. Initiating Branch & Cut.",
    "[Phase 3: Branch & Bound Exploration]",
    "Branching on fractional edge (v14 -> v27, x_ij = 0.542).",
    "Generating Gomory mixed-integer cuts (34 valid inequalities added).",
    "Pruning suboptimal branch 4B: bound 1,510 km exceeds incumbent 1,320 km.",
    "[Phase 4: Simulated Annealing Refinement]",
    "Annealing temperature T=100.0, cooling rate alpha=0.995.",
    "Iterating stochastic 3-opt perturbation swaps...",
    "Metropolis criterion accepted 42 uphill moves to escape local minima.",
    "[Phase 5: Convergence & Optimal Tour Verification]",
    "Integer feasibility confirmed. Duality gap closed to 0.00%.",
    "Final optimal tour length: 1,124.80 km (verified by Karush-Kuhn-Tucker conditions).",
    "Solver finished successfully.",
]


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "practice-116-backend", "port": 8116}


@app.post("/api/solver/stream-reasoning")
async def stream_solver_reasoning(request: Request, body: SolverPromptRequest):
    """Exercise 4 (Part A): Server-Sent Events (SSE) Streaming Endpoint.

    Streams reasoning tokens as 'text/event-stream'.
    Key Production Requirements:
    1. Detect client aborts via `await request.is_disconnected()`.
       If the user clicks 'Abort' in React, the backend must immediately halt
       computation to save resources.
    2. Format payloads conforming to the SSE wire standard:
       id: <event_id>\n
       event: token | step | done\n
       data: <json_string>\n\n
    3. Willfully simulate TCP chunk fragmentation: yield partial chunks across
       multiple yields so the frontend's stream buffer (Exercise 1) is thoroughly tested!
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        # --- TODO(human): Exercise 4 (Part A) ---
        # Task:
        # 1. Iterate over REASONING_STEPS or words.
        # 2. Before each step/yield, check:
        #      if await request.is_disconnected():
        #          print("[Backend SSE] Client disconnected/aborted. Halting computation.")
        #          break
        # 3. For each token/sentence:
        #    Construct the SSE wire format string:
        #      f"id: {idx}\nevent: token\ndata: {json.dumps({'token': token, 'idx': idx})}\n\n"
        # 4. (Bonus/Educational): Occasionally split the above string into two halves and yield
        #    each half separately with a tiny `await asyncio.sleep(0.01)` to emulate TCP packet fragmentation.
        # 5. At the end, yield an event with `event: done` and `data: {"finished": true}`.
        #
        # Starter Implementation:
        print(f"[Backend SSE] Started streaming for prompt: {body.prompt!r}")

        event_id = 0
        try:
            for step_idx, step_text in enumerate(REASONING_STEPS):
                # Check for client abort
                if await request.is_disconnected():
                    print(f"[Backend SSE] Client aborted at step {step_idx}! Aborting backend.")
                    return

                # Break step into individual tokens for live streaming feel
                words = step_text.split(" ")
                for word in words:
                    if await request.is_disconnected():
                        print(f"[Backend SSE] Client aborted mid-sentence! Aborting backend.")
                        return

                    event_id += 1
                    token_payload = json.dumps({
                        "token": word + " ",
                        "step": step_idx,
                        "id": event_id,
                        "timestamp": time.time(),
                    })
                    raw_event = f"id: {event_id}\nevent: token\ndata: {token_payload}\n\n"

                    # 30% of the time, intentionally split across two chunks to test client TCP buffering!
                    if random.random() < 0.3 and len(raw_event) > 8:
                        split_pt = random.randint(3, len(raw_event) - 3)
                        yield raw_event[:split_pt]
                        await asyncio.sleep(0.005)
                        yield raw_event[split_pt:]
                    else:
                        yield raw_event

                    await asyncio.sleep(body.simulated_delay_ms / 1000.0)

                # Send step boundary event
                event_id += 1
                yield f"id: {event_id}\nevent: step_complete\ndata: {json.dumps({'step': step_idx})}\n\n"

            # Completion event
            event_id += 1
            yield f"id: {event_id}\nevent: done\ndata: {json.dumps({'status': 'completed', 'total_events': event_id})}\n\n"
            print(f"[Backend SSE] Completed stream of {event_id} events.")

        except asyncio.CancelledError:
            print("[Backend SSE] Request cancelled by server/framework.")
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering (Nginx)
        },
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Exercise 4 (Part B): High-Frequency Telemetry & Control WebSocket.

    Emits live solver convergence telemetry at 10Hz - 500Hz.
    Handles bidirectional commands:
    - {"type": "SET_RATE", "hz": 250}: Adjusts broadcast rate dynamically.
    - {"type": "PING", "timestamp": ...}: Answers with {"type": "PONG", "seq": ...}.
    - {"type": "SET_PARAM", ...}: Updates solver temperature or duality target.
    """
    await websocket.accept()
    print("[Backend WS] Client connected to telemetry socket.")

    state = {
        "hz": 200,             # Default: 200 messages per second (stress tests React!)
        "iteration": 0,
        "objective": 2500.0,
        "duality_gap": 0.45,
        "temperature": 100.0,
        "cooling_rate": 0.9995,
        "running": True,
    }

    async def incoming_listener():
        """Handles incoming messages from the frontend."""
        try:
            while state["running"]:
                text = await websocket.receive_text()
                msg = json.loads(text)
                msg_type = msg.get("type")

                if msg_type == "PING":
                    # Echo back PONG for heartbeat latency calculation
                    await websocket.send_text(json.dumps({
                        "type": "PONG",
                        "client_timestamp": msg.get("timestamp"),
                        "server_timestamp": time.time(),
                        "seq": msg.get("seq", 0),
                    }))

                elif msg_type == "SET_RATE":
                    new_hz = max(1, min(1000, int(msg.get("hz", 100))))
                    state["hz"] = new_hz
                    print(f"[Backend WS] Telemetry rate adjusted to {new_hz} Hz")

                elif msg_type == "SET_PARAM":
                    if "cooling_rate" in msg:
                        state["cooling_rate"] = float(msg["cooling_rate"])
                    if "temperature" in msg:
                        state["temperature"] = float(msg["temperature"])

        except WebSocketDisconnect:
            state["running"] = False
        except Exception as e:
            print(f"[Backend WS] Listener error: {e}")
            state["running"] = False

    async def telemetry_broadcaster():
        """Broadcasts telemetry packets at state['hz'] frequency."""
        # --- TODO(human): Exercise 4 (Part B) ---
        # Task:
        # Implement the simulation math and high-frequency emission loop.
        # Ensure packet payload includes:
        # - iteration: monotonically increasing integer
        # - objective: descending curve towards optimum
        # - duality_gap: shrinking towards 0.0
        # - temperature: decaying by cooling_rate
        # - timestamp: time.time()
        # - hz: current emission frequency
        # Sleep for `1.0 / state["hz"]` between ticks.
        try:
            while state["running"]:
                state["iteration"] += 1
                hz = state["hz"]

                # Simulate optimization convergence
                state["temperature"] = max(0.01, state["temperature"] * state["cooling_rate"])
                decay = math.exp(-state["iteration"] / 5000.0)
                noise = (random.random() - 0.5) * 5.0 * state["temperature"]
                state["objective"] = 1124.80 + 1375.20 * decay + noise
                state["duality_gap"] = max(0.0, 0.45 * decay + (random.random() - 0.5) * 0.01)

                packet = {
                    "type": "TELEMETRY",
                    "iteration": state["iteration"],
                    "objective": round(state["objective"], 2),
                    "duality_gap": round(state["duality_gap"] * 100, 3),
                    "temperature": round(state["temperature"], 2),
                    "hz": hz,
                    "timestamp": time.time(),
                }

                await websocket.send_text(json.dumps(packet))
                # High-frequency sleep
                await asyncio.sleep(1.0 / hz)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[Backend WS] Broadcaster error: {e}")
        finally:
            state["running"] = False

    listener_task = asyncio.create_task(incoming_listener())
    broadcaster_task = asyncio.create_task(telemetry_broadcaster())

    done, pending = await asyncio.wait(
        [listener_task, broadcaster_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    print("[Backend WS] WebSocket connection closed cleanly.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8116, reload=True)

"""
SightLine — FastAPI Backend Server
Runs on AMD Cloud GPU. Handles vision, context memory, and WebSocket streaming.

Start with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import json
import os
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vision import VisionPipeline
from voice import VoiceAgent

app = FastAPI(title="SightLine", version="0.1.0")

# ── Context memory: last 5 scene descriptions ─────────────────────────────────
context_memory: deque[str] = deque(maxlen=5)

# ── Singletons (initialised once at startup) ──────────────────────────────────
vision = VisionPipeline()
voice = VoiceAgent()


# ── Pydantic models matching the agreed API contracts ─────────────────────────

class DescribeRequest(BaseModel):
    image: str  # base64-encoded JPEG
    mode: str = "general"  # "general" | "ocr" | "navigation" | "safety"


class DescribeResponse(BaseModel):
    description: str
    safety_alerts: list[str]
    confidence: float


class AskRequest(BaseModel):
    image: str  # base64-encoded JPEG
    question: str
    context: list[str] = []


class AskResponse(BaseModel):
    answer: str
    confidence: float


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": vision.model_name}


@app.post("/describe", response_model=DescribeResponse)
async def describe(req: DescribeRequest):
    """
    Takes a base64 image, runs Llama vision, returns a scene description.
    Mode controls the prompt style:
      general    — full scene overview
      ocr        — focus on reading text
      navigation — spatial layout, obstacles
      safety     — hazards first (stairs, vehicles, drops)
    """
    image_bytes = base64.b64decode(req.image)
    result = await vision.describe(image_bytes, mode=req.mode)

    # Store in context memory so /ask can use it
    context_memory.append(result["description"])

    return DescribeResponse(
        description=result["description"],
        safety_alerts=result.get("safety_alerts", []),
        confidence=result.get("confidence", 0.0),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Answers a user's specific question about the current scene.
    Passes recent context so follow-up questions work naturally.
    """
    image_bytes = base64.b64decode(req.image)

    # Merge caller-supplied context with server-side memory
    full_context = list(context_memory) + req.context

    result = await vision.ask(image_bytes, question=req.question, context=full_context)

    return AskResponse(
        answer=result["answer"],
        confidence=result.get("confidence", 0.0),
    )


# ── WebSocket endpoint for real-time streaming ────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Bidirectional WebSocket.

    Client → Server message shapes:
        {"type": "frame",    "image": "<base64>"}
        {"type": "question", "text":  "<string>"}

    Server → Client message shapes:
        {"type": "description", "text": "...", "safety_alerts": [...]}
        {"type": "answer",      "text": "..."}
        {"type": "error",       "message": "..."}
    """
    await ws.accept()
    print("[WS] client connected")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg["type"] == "frame":
                image_bytes = base64.b64decode(msg["image"])
                result = await vision.describe(image_bytes, mode="general")
                context_memory.append(result["description"])

                await ws.send_json({
                    "type": "description",
                    "text": result["description"],
                    "safety_alerts": result.get("safety_alerts", []),
                })

            elif msg["type"] == "question":
                # Use the most recent frame stored in context — caller must have
                # sent a frame first, or we answer from memory only.
                result = await vision.ask(
                    image_bytes=None,
                    question=msg["text"],
                    context=list(context_memory),
                )
                await ws.send_json({
                    "type": "answer",
                    "text": result["answer"],
                })

            else:
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg['type']}",
                })

    except WebSocketDisconnect:
        print("[WS] client disconnected")
    except Exception as e:
        print(f"[WS] error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})


# ── Serve the phone bridge as a static page ───────────────────────────────────

BRIDGE_DIR = Path(__file__).parent.parent / "bridge"

if BRIDGE_DIR.exists():
    app.mount("/bridge", StaticFiles(directory=str(BRIDGE_DIR), html=True), name="bridge")

    @app.get("/bridge", response_class=HTMLResponse, include_in_schema=False)
    async def bridge_root():
        return (BRIDGE_DIR / "index.html").read_text()

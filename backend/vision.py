"""
SightLine — Vision Pipeline
Wraps Llama 3.2 Vision (11B) via vLLM (AMD GPU) or Ollama (local fallback).

Swap the backend by setting VISION_BACKEND env var:
    VISION_BACKEND=vllm    → uses vLLM OpenAI-compatible API (AMD Cloud)
    VISION_BACKEND=ollama  → uses local Ollama
    VISION_BACKEND=mock    → returns fake data (useful for front-end dev)
"""

import base64
import io
import os
from typing import Optional

import httpx
from PIL import Image

VISION_BACKEND = os.getenv("VISION_BACKEND", "mock")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision:11b")

# ── Prompt templates per mode ──────────────────────────────────────────────────

PROMPTS = {
    "general": (
        "You are an AI assistant helping a blind person understand their surroundings. "
        "Describe what you see in 2-3 concise sentences. Focus on: people and their positions, "
        "objects, spatial layout, and anything actionable. Be warm and direct."
    ),
    "ocr": (
        "Read all text visible in this image. Quote it exactly, left-to-right, top-to-bottom. "
        "If no text is visible, say so briefly."
    ),
    "navigation": (
        "You are helping a blind person navigate safely. Describe the spatial layout: "
        "what is directly ahead, to the left, to the right. Estimate distances (e.g. '2 feet ahead'). "
        "Mention any openings, corridors, or paths. Be precise and concise."
    ),
    "safety": (
        "You are a safety assistant for a blind person. Scan this image for hazards FIRST: "
        "stairs, steps, drops, moving vehicles, wet floors, obstacles at head/shin height, crowds. "
        "State hazards immediately, then give a brief scene overview. Use alert language for dangers."
    ),
}


class VisionPipeline:
    def __init__(self):
        self.model_name = self._detect_model()
        print(f"[Vision] Backend: {VISION_BACKEND} | Model: {self.model_name}")

    def _detect_model(self) -> str:
        if VISION_BACKEND == "vllm":
            return "meta-llama/Llama-3.2-11B-Vision-Instruct"
        if VISION_BACKEND == "ollama":
            return OLLAMA_MODEL
        return "mock"

    # ── Public interface ───────────────────────────────────────────────────────

    async def describe(self, image_bytes: bytes, mode: str = "general") -> dict:
        """Returns {"description": str, "safety_alerts": list[str], "confidence": float}"""
        prompt = PROMPTS.get(mode, PROMPTS["general"])

        if VISION_BACKEND == "vllm":
            text = await self._call_vllm(image_bytes, prompt)
        elif VISION_BACKEND == "ollama":
            text = await self._call_ollama(image_bytes, prompt)
        else:
            text = self._mock_describe(mode)

        safety_alerts = self._extract_safety_alerts(text, mode)
        return {
            "description": text,
            "safety_alerts": safety_alerts,
            "confidence": 0.9 if VISION_BACKEND != "mock" else 1.0,
        }

    async def ask(
        self,
        image_bytes: Optional[bytes],
        question: str,
        context: list[str],
    ) -> dict:
        """Returns {"answer": str, "confidence": float}"""
        context_block = ""
        if context:
            context_block = (
                "Recent scene descriptions for context:\n"
                + "\n".join(f"- {c}" for c in context[-3:])
                + "\n\n"
            )

        prompt = (
            f"{context_block}"
            f"The user asks: {question}\n"
            "Answer concisely and helpfully. If the answer is visible in the image, use it. "
            "If relying on context, say so briefly."
        )

        if image_bytes and VISION_BACKEND == "vllm":
            text = await self._call_vllm(image_bytes, prompt)
        elif image_bytes and VISION_BACKEND == "ollama":
            text = await self._call_ollama(image_bytes, prompt)
        elif VISION_BACKEND == "mock" or not image_bytes:
            text = self._mock_ask(question, context)
        else:
            text = "I'm not sure — I don't have a current view."

        return {"answer": text, "confidence": 0.88}

    # ── vLLM backend (AMD GPU / OpenAI-compatible API) ─────────────────────────

    async def _call_vllm(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Ollama backend (local) ─────────────────────────────────────────────────

    async def _call_ollama(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 256},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"].strip()

    # ── Safety alert extraction ────────────────────────────────────────────────

    def _extract_safety_alerts(self, text: str, mode: str) -> list[str]:
        alerts = []
        keywords = {
            "stairs_detected": ["stair", "step", "steps"],
            "obstacle_ahead": ["obstacle", "blocking", "in your path"],
            "vehicle_nearby": ["car", "vehicle", "bus", "truck", "moving"],
            "drop_ahead": ["drop", "edge", "ledge", "cliff"],
            "wet_floor": ["wet", "slippery", "puddle"],
        }
        lower = text.lower()
        for alert_key, words in keywords.items():
            if any(w in lower for w in words):
                alerts.append(alert_key)
        return alerts

    # ── Mock backend (no GPU needed) ──────────────────────────────────────────

    def _mock_describe(self, mode: str) -> str:
        mocks = {
            "general": "You're in a medium-sized room with several people seated at tables. There's an open path ahead of you.",
            "ocr": "The sign reads: 'EXIT — Push bar to open'.",
            "navigation": "The corridor extends straight ahead for about 10 feet, then turns left. There's a door on your right.",
            "safety": "Clear path ahead. No immediate hazards detected. Two people are seated to your left.",
        }
        return mocks.get(mode, mocks["general"])

    def _mock_ask(self, question: str, context: list[str]) -> str:
        return f"[Mock] Based on the scene, the answer to '{question}' is: this is a placeholder response for development."

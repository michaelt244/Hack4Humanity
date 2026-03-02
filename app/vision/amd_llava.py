import httpx

from app.vision.base import VisionEngine
from app.vision.vision import trim_to_sentence

AMD_ENDPOINT = "http://165.245.140.111:8000/v1/chat/completions"
AMD_HEALTH = "http://165.245.140.111:8000/health"
AMD_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
AMD_TIMEOUT = 15.0

SYSTEM_PROMPT = (
    "You are a concise assistant for blind people. "
    "Respond in 1-2 short sentences, ideally under 30 words. "
    "Never start with 'The image shows' or 'I can see' or 'I see'."
)


class AmdLlavaEngine(VisionEngine):
    @property
    def name(self) -> str:
        return "AMD"

    def describe(self, frame_b64: str, prompt: str) -> str:
        payload = {
            "model": AMD_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": 120,
            "temperature": 0.1,
        }
        resp = httpx.post(AMD_ENDPOINT, json=payload, timeout=AMD_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return trim_to_sentence(raw)

    @staticmethod
    def available() -> bool:
        try:
            return httpx.get(AMD_HEALTH, timeout=5.0).status_code == 200
        except Exception:
            return False


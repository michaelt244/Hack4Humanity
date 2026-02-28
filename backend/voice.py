"""
SightLine — ElevenLabs Voice Agent
Converts text descriptions to natural speech and sends audio to the phone bridge.

Set ELEVENLABS_API_KEY in your .env file.
The agent persona is "Sightline" — warm, calm, concise.
"""

import os
from typing import Optional

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "")  # Conversational agent ID
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # "Bella" — warm female


class VoiceAgent:
    """
    Wraps ElevenLabs TTS.

    Two modes:
      speak(text)       → generates audio bytes from text (used for descriptions)
      agent_session()   → returns a signed URL for a live ElevenLabs Conversational
                          Agent session (used for real-time Q&A via the phone bridge)
    """

    def __init__(self):
        if not ELEVENLABS_API_KEY:
            print("[Voice] WARNING: ELEVENLABS_API_KEY not set — voice disabled")
            self._enabled = False
        else:
            self._enabled = True
            self._init_client()

    def _init_client(self):
        try:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            print("[Voice] ElevenLabs client initialised")
        except ImportError:
            print("[Voice] elevenlabs package not installed — pip install elevenlabs")
            self._enabled = False

    # ── Text-to-Speech ─────────────────────────────────────────────────────────

    def speak(self, text: str) -> Optional[bytes]:
        """
        Convert text to audio bytes (MP3).
        Returns None if voice is disabled.

        Usage:
            audio = voice.speak("You are in a large hall.")
            # stream audio_bytes to phone bridge
        """
        if not self._enabled:
            print(f"[Voice Mock] Would say: {text}")
            return None

        audio = self._client.generate(
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_turbo_v2",  # lowest latency model
        )
        # elevenlabs SDK returns a generator of bytes chunks
        return b"".join(audio)

    # ── Conversational Agent session ───────────────────────────────────────────

    def get_agent_signed_url(self) -> Optional[str]:
        """
        Returns a signed WebSocket URL for a live ElevenLabs Conversational Agent
        session. The phone bridge opens this URL directly so the user can speak
        and the agent responds in real time.

        Requires ELEVENLABS_AGENT_ID to be set.
        """
        if not self._enabled or not ELEVENLABS_AGENT_ID:
            print("[Voice] Agent ID not configured — skipping signed URL")
            return None

        try:
            result = self._client.conversational_ai.get_signed_url(
                agent_id=ELEVENLABS_AGENT_ID
            )
            return result.signed_url
        except Exception as e:
            print(f"[Voice] Failed to get signed URL: {e}")
            return None

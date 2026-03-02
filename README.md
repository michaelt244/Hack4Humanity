# SightLine

SightLine is a real-time vision assistant for blind and low-vision users. It captures live frames from a shared WhatsApp call window, describes the scene with AI vision models, and delivers low-latency voice feedback.

## Why It Matters

- Accessibility-first: prioritizes immediate hazards, orientation, and nearby objects.
- Real-time loop: continuous frame ingestion, model inference, and spoken guidance.
- Multi-backend: AMD LLaVA primary path with Gemini fallback support.
- Voice flexibility: macOS `say`, ElevenLabs, or silent mode.

## Architecture

```text
Browser Dashboard (app/static/index.html)
  -> Screen capture via getDisplayMedia()
  -> POST /upload-frame (every ~3s)
  -> WebSocket /ws/live (live events)

FastAPI App (app/main.py)
  -> app/services/vision_service.py
      -> app/vision/amd_llava.py
      -> app/vision/gemini_flash.py
  -> app/services/tts_service.py
      -> app/tts/mac_say.py
      -> app/tts/elevenlabs.py
  -> app/core/config.py (centralized settings)
  -> app/core/logger.py (structured logging)

Webhook Service (webhook/webhook_server.py)
  -> /upload-frame
  -> /tools/describe_scene
  -> /tools/control
```

## Repository Layout

```text
app/
  core/        # config + logging
  services/    # orchestration layer
  vision/      # vision engines + filters/prompts
  tts/         # TTS engines
  static/      # dashboard frontend
  main.py      # FastAPI entrypoint
webhook/       # webhook service for conversational agent tools
scripts/       # operational scripts (tool registration, automation)
tests/         # test suite
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Required environment variables in `.env`:

- `GEMINI_API_KEY`
- `ELEVENLABS_API_KEY` (required only for `--voice elevenlabs`)

Common optional values:

- `AMD_BASE_URL` (default `http://127.0.0.1:8000`)
- `WEBHOOK_PUBLIC_BASE_URL`
- `SIGHTLINE_ENGINE`, `SIGHTLINE_VOICE`, `SIGHTLINE_FOCUS`, `SIGHTLINE_PORT`

## Running

Main app:

```bash
python app/main.py
python app/main.py --engine gemini --voice elevenlabs
python app/main.py --engine amd --voice mac --focus safety --port 8080
```

Webhook service:

```bash
python webhook/webhook_server.py
```

Register ElevenLabs tools:

```bash
python scripts/setup_agent_tool.py --webhook-url https://<public-host>/tools/describe_scene
```

## CLI Reference

`python app/main.py [options]`

- `--engine` : `amd | gemini`
- `--voice` : `mac | elevenlabs | none`
- `--focus` : `general | ocr | navigation | safety`
- `--port` : FastAPI server port

CLI arguments override `.env` defaults.

## Security

- No API keys are hardcoded in source.
- Secrets are loaded only via environment variables (`app/core/config.py`).
- `.env` and runtime logs are excluded from version control.
- Structured logs avoid printing secrets and sensitive tokens.

If a key was previously exposed, rotate it immediately in the provider console.

## Responsible AI

- Assistive, not autonomous: SightLine provides guidance, not final navigation decisions.
- Safety-first prompting: outputs prioritize hazards and spatial context.
- Human-in-the-loop design: intended for caregiver-assisted or user-verified operation.
- Known limitations: model errors, latency spikes, and low-light scenes can reduce reliability.

Do not use SightLine as the sole source for high-risk mobility decisions.

## Notes for Recruiters

SightLine demonstrates production-minded software engineering across:

- Real-time systems design
- FastAPI backend architecture
- Provider-agnostic AI integration
- Structured logging and config hardening
- Accessibility-driven product thinking

"""
SightLine — Instagram Live Stream Capture
Captures frames from an Instagram Live stream every N seconds and sends
them to the backend /describe endpoint.

Usage:
    python stream.py --username YOUR_IG_USERNAME
    python stream.py --url <direct-stream-url>  # for testing with any HLS/RTMP URL

Dependencies:
    yt-dlp   — extracts the live stream URL from Instagram
    ffmpeg   — must be installed on the system (apt install ffmpeg)
    httpx    — sends frames to the backend

Environment variables:
    BACKEND_URL   — default: http://localhost:8000
    FRAME_INTERVAL — seconds between captures (default: 2)
"""

import argparse
import asyncio
import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "2"))


# ── Stream URL resolution ──────────────────────────────────────────────────────

def get_stream_url(username: str) -> str:
    """
    Use yt-dlp to extract the live HLS/RTMP URL from an Instagram Live.
    The user must be currently live for this to work.
    """
    ig_url = f"https://www.instagram.com/{username}/live/"
    print(f"[Capture] Resolving stream URL for {ig_url} ...")

    result = subprocess.run(
        ["yt-dlp", "--get-url", "--no-warnings", ig_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed: {result.stderr.strip()}\n"
            "Make sure the account is currently live and you are logged in."
        )

    url = result.stdout.strip().splitlines()[0]
    print(f"[Capture] Stream URL: {url[:80]}...")
    return url


# ── Frame extraction ───────────────────────────────────────────────────────────

def capture_frame(stream_url: str, output_path: str) -> bool:
    """
    Use ffmpeg to grab one frame from the live stream and save it as a JPEG.
    Returns True on success.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",                    # overwrite output
            "-i", stream_url,        # input stream
            "-frames:v", "1",        # grab exactly one frame
            "-q:v", "2",             # JPEG quality (2 = high, 31 = low)
            "-vf", "scale=640:-1",   # resize to 640px wide (keeps aspect)
            output_path,
        ],
        capture_output=True,
        timeout=15,
    )
    return result.returncode == 0


# ── Backend communication ──────────────────────────────────────────────────────

async def send_frame(client: httpx.AsyncClient, image_bytes: bytes, mode: str = "general") -> dict:
    """POST a frame to the backend /describe endpoint."""
    b64 = base64.b64encode(image_bytes).decode()
    resp = await client.post(
        f"{BACKEND_URL}/describe",
        json={"image": b64, "mode": mode},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Main capture loop ─────────────────────────────────────────────────────────

async def capture_loop(stream_url: str, mode: str = "general"):
    print(f"[Capture] Starting capture loop — interval: {FRAME_INTERVAL}s, mode: {mode}")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = str(Path(tmpdir) / "frame.jpg")

        async with httpx.AsyncClient() as client:
            while True:
                loop_start = time.monotonic()

                # 1. Grab a frame from the live stream
                if not capture_frame(stream_url, frame_path):
                    print("[Capture] ffmpeg failed to capture frame — retrying...")
                    await asyncio.sleep(1)
                    continue

                # 2. Read the JPEG bytes
                image_bytes = Path(frame_path).read_bytes()
                print(f"[Capture] Frame captured ({len(image_bytes):,} bytes)")

                # 3. Send to backend
                try:
                    result = await send_frame(client, image_bytes, mode=mode)
                    description = result.get("description", "")
                    alerts = result.get("safety_alerts", [])

                    print(f"[Capture] Description: {description[:120]}")
                    if alerts:
                        print(f"[Capture] SAFETY ALERTS: {alerts}")

                except httpx.HTTPError as e:
                    print(f"[Capture] Backend error: {e}")

                # 4. Wait for the remainder of the interval
                elapsed = time.monotonic() - loop_start
                sleep_for = max(0, FRAME_INTERVAL - elapsed)
                await asyncio.sleep(sleep_for)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SightLine stream capture")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--username", help="Instagram username (must be currently live)")
    group.add_argument("--url", help="Direct HLS/RTMP stream URL (for testing)")
    parser.add_argument(
        "--mode",
        default="general",
        choices=["general", "ocr", "navigation", "safety"],
        help="Vision mode sent to backend (default: general)",
    )
    args = parser.parse_args()

    # Resolve stream URL
    if args.url:
        stream_url = args.url
    else:
        stream_url = get_stream_url(args.username)

    # Run
    try:
        asyncio.run(capture_loop(stream_url, mode=args.mode))
    except KeyboardInterrupt:
        print("\n[Capture] Stopped.")


if __name__ == "__main__":
    main()

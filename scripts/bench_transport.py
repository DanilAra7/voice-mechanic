#!/usr/bin/env python3
"""How much of a turn's latency is transport rather than work?

The server stamps its clock in `VoiceSession.handle`; the browser stamps its own when the
button comes up. On a live box those two disagree by far more than the measured round trip, and
the suspicion that costs nothing to test is the socket loop itself: it handles every incoming
audio message inline before reading the next one, so `end_of_speech` could be queuing behind
four seconds of microphone audio.

This runs the real receive loop, the real turn detector and a real WebSocket over loopback,
streaming microphone-shaped audio at the rate a browser actually sends it. Recognition, the
agent and the voice are stand-ins that take no time, so whatever this measures is transport and
scheduling and nothing else. It needs no GPU and no models beyond the VAD.

    uv run python scripts/bench_transport.py
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

CHUNK = 128  # samples per message: what an AudioWorklet posts per render quantum


class NoOpAgent:
    async def stream(self, text, on_event=None, **kwargs):
        yield "Let me check that."


def build_app(marks: dict):
    from fastapi import FastAPI, WebSocket
    from helpers_voice import FakeRecognizer, FakeSynthesiser
    from starlette.websockets import WebSocketDisconnect

    from mechanic.voice.session import VoiceSession
    from mechanic.voice.vad import TurnDetector
    from mechanic.voice.ws import pcm_to_float

    app = FastAPI()

    @app.websocket("/ws/voice")
    async def voice(ws: WebSocket) -> None:
        await ws.accept()
        await ws.receive_json()

        async def on_event(event, payload):
            try:
                await ws.send_json({"type": event, **payload})
            except Exception:  # the bench hangs up mid-answer on purpose
                pass

        async def on_audio(chunk, _rate):
            await ws.send_bytes((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())

        session = VoiceSession(
            agent=NoOpAgent(),
            recognizer=FakeRecognizer(),
            synthesiser=FakeSynthesiser(),
            detector=TurnDetector(),
            on_event=on_event,
            on_audio=on_audio,
        )
        await ws.send_json({"type": "ready", "audio_rate": 24000})
        # Structurally identical to mechanic.voice.ws: one message at a time, audio inline.
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (payload := message.get("bytes")) is not None:
                    await session.push_audio(pcm_to_float(payload))
                elif (text := message.get("text")) is not None:
                    command = json.loads(text)
                    if command.get("type") == "end_of_speech":
                        marks["dequeued"] = time.monotonic()
                        await session.end_of_speech()
                    elif command.get("type") == "start_of_speech":
                        session.start_of_speech()
        except WebSocketDisconnect:
            pass

    return app


def question(seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    wave = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (wave.astype(np.float32) * 32767).astype("<i2")


async def one_run(port: int, marks: dict, seconds: float, paced: bool) -> dict:
    import websockets

    from mechanic.voice.vad import SAMPLE_RATE

    pcm = question(seconds, SAMPLE_RATE)
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "device": "bench"}))
        while json.loads(await ws.recv()).get("type") != "ready":
            pass
        await ws.send(json.dumps({"type": "start_of_speech"}))

        started = time.monotonic()
        for i in range(0, len(pcm), CHUNK):
            await ws.send(pcm[i : i + CHUNK].tobytes())
            if paced:
                # A microphone produces audio in real time; a browser cannot send it faster.
                if (nap := started + (i + CHUNK) / SAMPLE_RATE - time.monotonic()) > 0:
                    await asyncio.sleep(nap)

        released = time.monotonic()
        await ws.send(json.dumps({"type": "end_of_speech"}))
        while True:
            frame = await asyncio.wait_for(ws.recv(), timeout=10)
            if isinstance(frame, bytes):
                break
        return {
            "messages": len(pcm) // CHUNK,
            "stream_s": released - started,
            "queued_ms": (marks["dequeued"] - released) * 1000,
            "client_ms": (time.monotonic() - released) * 1000,
        }


async def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=4.0, help="length of the spoken question")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    marks: dict = {}
    server = uvicorn.Server(uvicorn.Config(build_app(marks), host="127.0.0.1", port=args.port, log_level="error"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    for paced, label in ((True, "paced like a microphone"), (False, "sent as fast as the socket allows")):
        rows = []
        for _ in range(args.runs):
            marks.clear()
            rows.append(await one_run(args.port, marks, args.seconds, paced))
        queued = sorted(r["queued_ms"] for r in rows)
        client = sorted(r["client_ms"] for r in rows)
        print(f"\n--- {label} ---")
        head = f"  {rows[0]['messages']} messages for {args.seconds:.0f} s of speech"
        print(f"{head}, streamed over {rows[0]['stream_s']:.2f} s")
        print(f"  button up -> server dequeued it   : {queued[len(queued) // 2]:7.1f} ms")
        print(f"  button up -> first audio at client: {client[len(client) // 2]:7.1f} ms")

    print("\nAnything above a few milliseconds here is the socket loop, not the network.")
    server.should_exit = True
    await serving


if __name__ == "__main__":
    asyncio.run(main())

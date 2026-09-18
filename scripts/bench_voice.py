#!/usr/bin/env python3
"""Drive the whole spoken loop from recorded speech and report what the driver waited for.

The site lets a person measure latency by talking; this does the same with fixed WAV files, so
the number is repeatable and can be compared between runs. Audio is streamed at the speed a
microphone delivers it — sending it faster would measure the server rather than the wait.

    python scripts/bench_voice.py --url ws://127.0.0.1:8000/ws/voice --audio data/cache/tts_samples/kyutai
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

from mechanic.voice.asr import SAMPLE_RATE, resample, to_mono

CHUNK_S = 0.1
TRAILING_SILENCE_S = 1.2       # long enough for the turn detector to call the utterance finished


def load_clip(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    return resample(to_mono(audio), sr, SAMPLE_RATE)


def to_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def run_clip(ws, audio: np.ndarray) -> dict:
    step = int(SAMPLE_RATE * CHUNK_S)
    started = time.monotonic()
    for i in range(0, len(audio), step):
        await ws.send(to_pcm(audio[i : i + step]))
        elapsed = time.monotonic() - started
        due = (i / SAMPLE_RATE) + CHUNK_S
        if due > elapsed:
            await asyncio.sleep(due - elapsed)

    silence = to_pcm(np.zeros(step, dtype=np.float32))
    end_of_speech = time.monotonic()
    result: dict = {"audio_bytes": 0}
    sent_silence = 0.0

    while True:
        if sent_silence < TRAILING_SILENCE_S:
            await ws.send(silence)
            sent_silence += CHUNK_S
            await asyncio.sleep(CHUNK_S)
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=0.05)
        except TimeoutError:
            if sent_silence >= TRAILING_SILENCE_S and time.monotonic() - end_of_speech > 25:
                result["error"] = "timed out waiting for the answer"
                return result
            continue
        if isinstance(message, bytes):
            if "first_audio_client_ms" not in result:
                result["first_audio_client_ms"] = round((time.monotonic() - end_of_speech) * 1000)
            result["audio_bytes"] += len(message)
            continue
        event = json.loads(message)
        kind = event.pop("type")
        if kind == "transcript":
            result["transcript"] = event.get("text")
        elif kind == "sentence":
            result.setdefault("sentences", []).append(event.get("text"))
        elif kind == "tool":
            result.setdefault("tools", []).append(event.get("name"))
        elif kind == "error":
            result["error"] = event.get("message")
            return result
        elif kind == "turn_end":
            result["server"] = event
            return result


async def main_async(args) -> None:
    clips = sorted(Path(args.audio).glob("*.wav"))[: args.limit]
    rows = []
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "device": args.device, "vehicle": args.vehicle}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"server did not become ready: {ready}")
        print(f"connected · mic {ready['mic_rate']} Hz · replies {ready['audio_rate']} Hz\n")

        for path in clips:
            await ws.send(json.dumps({"type": "reset"}))
            while json.loads(await ws.recv()).get("type") != "reset_done":
                pass
            row = await run_clip(ws, load_clip(path))
            row["clip"] = path.name
            rows.append(row)
            if err := row.get("error"):
                print(f"{path.name}: ОШИБКА {err}")
                continue
            s = row.get("server", {})
            print(f"{path.name}: heard \"{row.get('transcript', '')[:56]}\"")
            print(f"    asr {s.get('asr_ms', '?')} · first sentence {s.get('first_sentence_ms', '?')} · "
                  f"first audio {s.get('first_audio_ms', '?')} ms (client saw {row.get('first_audio_client_ms', '?')})"
                  f" · tools: {', '.join(row.get('tools', [])) or 'none'}")

    good = [r for r in rows if "server" in r and r["server"].get("first_audio_ms")]
    if good:
        audio_ms = sorted(r["server"]["first_audio_ms"] for r in good)
        q = lambda p: round(audio_ms[min(len(audio_ms) - 1, int(p * len(audio_ms)))])  # noqa: E731
        print(f"\n{len(good)}/{len(rows)} turns · first audio p50 {q(0.5)} ms · p95 {q(0.95)} ms · "
              f"mean {round(statistics.mean(audio_ms))} ms")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/voice")
    p.add_argument("--audio", default="data/cache/tts_samples/kyutai")
    p.add_argument("--device", default="bench")
    p.add_argument("--vehicle", default="audi_a4_b8")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out", default="evals/results/voice_pipeline.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

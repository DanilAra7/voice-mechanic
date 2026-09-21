#!/usr/bin/env python3
"""Drive the whole spoken loop from recorded speech and report what the driver waited for.

The site lets a person measure latency by talking; this does the same with fixed WAV files, so
the number is repeatable and can be compared between runs. Audio is streamed at the speed a
microphone delivers it — sending it faster would measure the server rather than the wait.

    python scripts/bench_voice.py --url ws://127.0.0.1:8000/ws/voice --audio evals/audio/driver
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
# A microphone does not stop when the question does, so neither does this: silence keeps flowing
# while the answer is spoken, which is also what makes the barge-in guard worth testing.
# Generous on purpose: an answer that runs long is a result to report, not a run to abandon.
# Giving up early leaves its audio in flight and corrupts the *next* clip's numbers instead.
ANSWER_TIMEOUT_S = 60.0


def load_clip(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    return resample(to_mono(audio), sr, SAMPLE_RATE)


def to_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def stream_microphone(ws, audio: np.ndarray, stop: asyncio.Event) -> None:
    """Send the question, then keep the line open with silence, all at real-time speed."""
    step = int(SAMPLE_RATE * CHUNK_S)
    silence = to_pcm(np.zeros(step, dtype=np.float32))
    started = time.monotonic()
    sent = 0.0
    try:
        for i in range(0, len(audio), step):
            await ws.send(to_pcm(audio[i : i + step]))
            sent += CHUNK_S
            if (delay := sent - (time.monotonic() - started)) > 0:
                await asyncio.sleep(delay)
        while not stop.is_set():
            await ws.send(silence)
            sent += CHUNK_S
            if (delay := sent - (time.monotonic() - started)) > 0:
                await asyncio.sleep(delay)
    except (asyncio.CancelledError, ConnectionError):
        pass


async def drain_until(ws, wanted: str, deadline: float) -> None:
    """Skip whatever is still in flight — audio frames included — until the expected event."""
    while time.monotonic() < deadline:
        message = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
        if isinstance(message, bytes):
            continue  # leftover audio from the previous answer
        if json.loads(message).get("type") == wanted:
            return
    raise TimeoutError(f"never saw {wanted}")


async def run_clip(ws, audio: np.ndarray) -> dict:
    """One spoken question, timed from the last sample of speech."""
    stop = asyncio.Event()
    speech_s = len(audio) / SAMPLE_RATE
    sender = asyncio.create_task(stream_microphone(ws, audio, stop))
    end_of_speech = time.monotonic() + speech_s  # when the last sample will have been sent

    result: dict = {"audio_bytes": 0}
    try:
        while True:
            remaining = end_of_speech + ANSWER_TIMEOUT_S - time.monotonic()
            if remaining <= 0:
                result["error"] = "timed out waiting for the answer"
                return result
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                result["error"] = "timed out waiting for the answer"
                return result

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
            elif kind == "flush":
                result["flushed"] = True
            elif kind == "error":
                result["error"] = event.get("message")
                return result
            elif kind in ("turn_end", "turn_skipped"):
                # A turn that was cut short is not the answer to this question. Some of our clips
                # hold a pause long enough to split in two, and the agent then either carries the
                # first half over or treats the second half as an interruption; either way the
                # answer is the turn that comes after. Counting the dropped one as the reply is
                # how this harness used to report a 0.1-second answer as a fast one.
                if event.get("carried_on") or event.get("barged_in"):
                    result["restarts"] = result.get("restarts", 0) + 1
                    result.pop("first_audio_client_ms", None)
                    result.pop("sentences", None)
                    result["audio_bytes"] = 0
                    continue
                result["server"] = event
                return result
    finally:
        stop.set()
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)


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
            await drain_until(ws, "reset_done", time.monotonic() + 15)
            row = await run_clip(ws, load_clip(path))
            row["clip"] = path.name
            # How long the driver has to listen before they can speak again. A fast first word
            # is worth little if the answer then runs for most of a minute.
            row["spoken_s"] = round(row["audio_bytes"] / (ready["audio_rate"] * 2), 1)
            rows.append(row)
            if err := row.get("error"):
                print(f"{path.name}: ОШИБКА {err}")
                continue
            s = row.get("server", {})
            print(f'{path.name}: heard "{row.get("transcript", "")[:56]}"')
            ms = {k: round(v) if isinstance(v, int | float) else "?" for k, v in s.items()}
            print(
                f"    asr {ms.get('asr_ms', '?')} · tools {ms.get('tool_ms', 0)} · "
                f"first sentence {ms.get('first_sentence_ms', '?')} · "
                f"first audio {ms.get('first_audio_ms', '?')} ms (client saw {row.get('first_audio_client_ms', '?')})"
                f" · spoke {row['spoken_s']} s in {len(row.get('sentences', []))} sentences"
                f" · tools: {', '.join(row.get('tools', [])) or 'none'}"
            )

    good = [r for r in rows if "server" in r and r["server"].get("first_audio_ms")]
    if good:
        audio_ms = sorted(r["server"]["first_audio_ms"] for r in good)
        q = lambda p: round(audio_ms[min(len(audio_ms) - 1, int(p * len(audio_ms)))])  # noqa: E731
        spoken = sorted(r["spoken_s"] for r in good)

        # Where the wait actually goes. Each stage is the gap from the one before it, so the
        # rows add up to the whole wait instead of overlapping.
        def med(key: str) -> int:
            xs = sorted(r["server"].get(key) or 0 for r in good)
            return round(statistics.median(xs)) if xs else 0

        asr, sentence, audio, tools = med("asr_ms"), med("first_sentence_ms"), med("first_audio_ms"), med("tool_ms")
        print("\nwhere the wait goes, medians:")
        print(f"  recognition              {asr:5d} ms")
        print(f"  model, tools included    {sentence - asr:5d} ms   (tools themselves {tools} ms)")
        print(f"  synthesis to first sound {audio - sentence:5d} ms")
        print(f"  server total             {audio:5d} ms")
        print(
            f"\n{len(good)}/{len(rows)} turns · first audio p50 {q(0.5)} ms · p95 {q(0.95)} ms · "
            f"mean {round(statistics.mean(audio_ms))} ms"
        )
        print(
            f"answer length: median {statistics.median(spoken)} s · longest {spoken[-1]} s · "
            f"mean {round(statistics.mean(spoken), 1)} s"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/voice")
    p.add_argument("--audio", default="evals/audio/driver")
    p.add_argument("--device", default="bench")
    p.add_argument("--vehicle", default="audi_a4_b8")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out", default="evals/results/voice_pipeline.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure what speech recognition costs the conversation, on CPU.

Two numbers matter and they are not the same:
  * real-time factor - can it keep up while the driver is still talking (must stay under 1.0);
  * tail latency - how long the driver waits AFTER falling silent, which is what lands in the
    turn budget. A streaming model has almost no tail; an offline one does all its work there.

    uv run python scripts/bench_asr.py --audio data/cache/tts_samples/kyutai
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

TARGET_SR = 16000
CHUNK_S = 0.1


def load_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        n = int(len(audio) * TARGET_SR / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype("float32")
    return audio


def run_streaming(model_dir: Path, clips, threads: int) -> list[dict]:
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        num_threads=threads,
        provider="cpu",
    )
    rows = []
    for name, audio in clips:
        stream = rec.create_stream()
        step = int(TARGET_SR * CHUNK_S)
        compute = 0.0
        for i in range(0, len(audio), step):          # as a live mic would deliver it
            t = time.monotonic()
            stream.accept_waveform(TARGET_SR, audio[i:i + step])
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            compute += time.monotonic() - t
        tail = time.monotonic()                        # the driver has stopped talking here
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream)
        tail_s = time.monotonic() - tail
        dur = len(audio) / TARGET_SR
        rows.append({"clip": name, "audio_s": round(dur, 2), "rtf": round(compute / dur, 3),
                     "tail_ms": round(tail_s * 1000), "text": text})
    return rows


def run_offline(model_dir: Path, clips, threads: int) -> list[dict]:
    if (model_dir / "model.int8.onnx").exists():
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"), num_threads=threads, provider="cpu",
        )
    else:
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"), num_threads=threads, provider="cpu",
            model_type="nemo_transducer",
        )
    rows = []
    for name, audio in clips:
        t = time.monotonic()                           # offline pays the whole bill in the tail
        stream = rec.create_stream()
        stream.accept_waveform(TARGET_SR, audio)
        rec.decode_stream(stream)
        text = stream.result.text
        tail_s = time.monotonic() - t
        dur = len(audio) / TARGET_SR
        rows.append({"clip": name, "audio_s": round(dur, 2), "rtf": round(tail_s / dur, 3),
                     "tail_ms": round(tail_s * 1000), "text": text})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="data/cache/tts_samples/kyutai")
    p.add_argument("--models", default="data/models")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="evals/results/asr.json")
    args = p.parse_args()

    clips = [(f.name, load_16k(f)) for f in sorted(Path(args.audio).glob("*.wav"))]
    print(f"{len(clips)} clips, {sum(len(a) for _, a in clips) / TARGET_SR:.1f}s of speech, "
          f"{args.threads} threads\n")

    models = Path(args.models)
    engines = {
        "parakeet-0.6b-streaming-240ms": (run_streaming, models / "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-240ms"),
        "parakeet-110m-offline": (run_offline, models / "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8"),
        "parakeet-0.6b-offline": (run_offline, models / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
    }
    summary = {}
    for name, (fn, path) in engines.items():
        if not path.exists():
            print(f"{name}: missing {path}")
            continue
        t0 = time.monotonic()
        rows = fn(path, clips, args.threads)
        load_and_run = time.monotonic() - t0
        tails = [r["tail_ms"] for r in rows]
        rtfs = [r["rtf"] for r in rows]
        summary[name] = {"tail_ms_p50": round(statistics.median(tails)),
                         "tail_ms_max": max(tails), "rtf_median": round(statistics.median(rtfs), 3),
                         "total_s": round(load_and_run, 1), "rows": rows}
        print(f"=== {name}")
        print(f"    после конца речи: p50 {summary[name]['tail_ms_p50']} ms, max {max(tails)} ms · "
              f"rtf med {summary[name]['rtf_median']}")
        for r in rows[:2]:
            print(f"    \"{r['text']}\"")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"threads": args.threads, "engines": summary}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

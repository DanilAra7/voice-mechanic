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
        for i in range(0, len(audio), step):  # as a live mic would deliver it
            t = time.monotonic()
            stream.accept_waveform(TARGET_SR, audio[i : i + step])
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            compute += time.monotonic() - t
        tail = time.monotonic()  # the driver has stopped talking here
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream)
        tail_s = time.monotonic() - tail
        dur = len(audio) / TARGET_SR
        rows.append(
            {
                "clip": name,
                "audio_s": round(dur, 2),
                "rtf": round(compute / dur, 3),
                "tail_ms": round(tail_s * 1000),
                "text": text,
            }
        )
    return rows


def run_offline(model_dir: Path, clips, threads: int) -> list[dict]:
    if (model_dir / "model.int8.onnx").exists():
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=threads,
            provider="cpu",
        )
    else:
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=threads,
            provider="cpu",
            model_type="nemo_transducer",
        )
    rows = []
    for name, audio in clips:
        t = time.monotonic()  # offline pays the whole bill in the tail
        stream = rec.create_stream()
        stream.accept_waveform(TARGET_SR, audio)
        rec.decode_stream(stream)
        text = stream.result.text
        tail_s = time.monotonic() - t
        dur = len(audio) / TARGET_SR
        rows.append(
            {
                "clip": name,
                "audio_s": round(dur, 2),
                "rtf": round(tail_s / dur, 3),
                "tail_ms": round(tail_s * 1000),
                "text": text,
            }
        )
    return rows


def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.01, window: int = 160) -> np.ndarray:
    """Cut the quiet tail so the clip ends on the last spoken sample.

    Without this the detector closes the turn while the clip is still playing and the measured
    delay comes out near zero — it would be measuring the recording, not the detector.
    """
    for end in range(len(audio) - window, 0, -window):
        if np.abs(audio[end : end + window]).max() > threshold:
            return audio[: end + window]
    return audio


def measure_turn_delay(clips, silences: list[float]) -> dict:
    """From the driver's last word to a transcript in hand — the delay no model tuning removes.

    The detector must hear `min_silence_s` of quiet before it calls the turn over, and that wait
    lands in the reply. Feeding audio in real-sized chunks keeps the accounting honest.
    """
    from mechanic.voice.asr import Recognizer
    from mechanic.voice.vad import SAMPLE_RATE, TurnDetector

    rec = Recognizer()
    rec.warm_up()
    out = {}
    for min_silence in silences:
        rows = []
        for name, raw in clips[:5]:
            audio = trim_trailing_silence(raw)
            det = TurnDetector(min_silence_s=min_silence)
            step = int(SAMPLE_RATE * CHUNK_S)
            early = []
            for i in range(0, len(audio), step):       # the driver is still speaking
                early.extend(det.push(audio[i : i + step]))
            end_of_speech = time.monotonic()           # they stop here
            utterance = early[-1] if early else None   # a pause mid-clip can end the turn early
            silence = np.zeros(step, dtype=np.float32)
            deadline = end_of_speech + 3.0             # a clip that never closes is a finding, not a hang
            # Silence has to arrive at the speed a microphone delivers it: min_silence_s is counted
            # in audio time, so feeding it faster than real time would measure the CPU, not the wait.
            fed = 0.0
            while utterance is None and time.monotonic() < deadline:
                for u in det.push(silence):
                    utterance = u
                    break
                fed += CHUNK_S
                slept = time.monotonic() - end_of_speech
                if fed > slept:
                    time.sleep(fed - slept)
            if utterance is None:
                print(f"    {name}: конец фразы не распознан за 3 с")
                continue
            if early:
                print(f"    {name}: фраза закрылась ещё во время речи — замер по ней неверен")
            detected_ms = (time.monotonic() - end_of_speech) * 1000
            text = rec.transcribe(utterance.audio).text
            rows.append({"clip": name, "detect_ms": round(detected_ms),
                         "total_ms": round((time.monotonic() - end_of_speech) * 1000), "text": text})
        out[f"min_silence_{min_silence}s"] = {
            "detect_ms_p50": round(statistics.median(r["detect_ms"] for r in rows)),
            "total_ms_p50": round(statistics.median(r["total_ms"] for r in rows)),
            "rows": rows,
        }
        d = out[f"min_silence_{min_silence}s"]
        print(f"  тишина {min_silence:.2f}s -> конец речи распознан за {d['detect_ms_p50']} ms, "
              f"текст готов через {d['total_ms_p50']} ms")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="data/cache/tts_samples/kyutai")
    p.add_argument("--models", default="data/models")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="evals/results/asr.json")
    p.add_argument("--turn-only", action="store_true", help="Skip the engine sweep, measure the turn delay")
    args = p.parse_args()

    clips = [(f.name, load_16k(f)) for f in sorted(Path(args.audio).glob("*.wav"))]
    print(f"{len(clips)} clips, {sum(len(a) for _, a in clips) / TARGET_SR:.1f}s of speech, {args.threads} threads\n")

    models = Path(args.models)
    if args.turn_only:
        print("=== от конца речи до готового текста (VAD + ASR)")
        measure_turn_delay(clips, [0.15, 0.2, 0.35, 0.5])
        return
    streaming_dir = "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-240ms"
    engines = {
        "parakeet-0.6b-streaming-240ms": (run_streaming, models / streaming_dir),
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
        summary[name] = {
            "tail_ms_p50": round(statistics.median(tails)),
            "tail_ms_max": max(tails),
            "rtf_median": round(statistics.median(rtfs), 3),
            "total_s": round(load_and_run, 1),
            "rows": rows,
        }
        print(f"=== {name}")
        print(
            f"    после конца речи: p50 {summary[name]['tail_ms_p50']} ms, max {max(tails)} ms · "
            f"rtf med {summary[name]['rtf_median']}"
        )
        for r in rows[:2]:
            print(f'    "{r["text"]}"')
        print()

    print("=== от конца речи до готового текста (VAD + ASR)")
    turn = measure_turn_delay(clips, [0.2, 0.35, 0.5])
    print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"threads": args.threads, "engines": summary, "turn_delay": turn}, indent=2)
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

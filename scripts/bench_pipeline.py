#!/usr/bin/env python3
"""Run the LLM and the synthesiser on the same card and time what the driver actually waits for.

Measured separately, the two halves both look affordable; what matters is whether they fit
together and how long the first spoken word takes end to end.

    python scripts/bench_pipeline.py --llm-url http://127.0.0.1:8001/v1 --model gpt-oss-20b
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request

SENTENCE_END = re.compile(r"(?<=[.!?])\s")
QUESTIONS = [
    "My temperature gauge is climbing and it is still going up.",
    "The car shakes at idle and the check engine light came on.",
    "I smell gasoline inside the cabin while driving.",
]
SYSTEM = ("You are Dex, a car mechanic talking to a driver over voice. Speak in short plain sentences. "
          "No markdown, no lists, no URLs. Two or three sentences at most.")


def gpu_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[0])


def stream_first_sentence(url: str, model: str, question: str, extra: dict) -> tuple[str, float]:
    """Return the first complete sentence and how long it took to arrive."""
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": question}],
            "max_tokens": 160, "temperature": 0.3, "stream": True, **extra}
    req = urllib.request.Request(f"{url}/chat/completions", method="POST",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    text = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
            if not delta:
                continue
            text += delta
            if SENTENCE_END.search(text):
                sentence = SENTENCE_END.split(text)[0]
                return sentence.strip(), time.monotonic() - started
    return text.strip(), time.monotonic() - started


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-low", action="store_true", help="gpt-oss needs this or it thinks out loud first")
    p.add_argument("--out", default="evals/results/tts/pipeline.json")
    args = p.parse_args()
    extra = {"chat_template_kwargs": {"reasoning_effort": "low"}} if args.reasoning_low else {}

    llm_only = gpu_mib()
    print(f"VRAM with the LLM loaded: {llm_only} MiB", flush=True)

    import numpy as np
    import torch
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

    info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device=torch.device("cuda"))
    cond = tts.make_condition_attributes(
        [tts.get_voice_path("expresso/ex03-ex01_happy_001_channel1_334s.wav")], cfg_coef=2.0)
    both = gpu_mib()
    print(f"VRAM with both loaded:     {both} MiB  (synthesiser added {both - llm_only} MiB)", flush=True)

    def speak(text: str):
        entries = tts.prepare_script([text], padding_between=1)
        first = None
        started = time.monotonic()

        def on_frame(frame):
            nonlocal first
            if (frame != -1).all() and first is None:
                first = time.monotonic() - started

        with tts.mimi.streaming(1):
            tts.generate([entries], [cond], on_frame=on_frame)
        return first

    speak("Warming up the synthesiser so the first driver does not pay for it.")
    rows, peak = [], gpu_mib()
    for q in QUESTIONS:
        sentence, llm_s = stream_first_sentence(args.llm_url, args.model, q, extra)
        tts_s = speak(sentence)
        peak = max(peak, gpu_mib())
        total = (llm_s + tts_s) * 1000
        rows.append({"question": q, "sentence": sentence, "llm_ms": round(llm_s * 1000),
                     "tts_ms": round(tts_s * 1000), "total_ms": round(total)})
        print(f"\n  ? {q}\n  > {sentence}\n    llm {llm_s * 1000:.0f} ms + tts {tts_s * 1000:.0f} ms "
              f"= {total:.0f} ms to first audio", flush=True)

    summary = {"model": args.model, "vram_llm_mib": llm_only, "vram_both_mib": both,
               "vram_peak_mib": peak, "headroom_mib": 16376 - peak, "turns": rows,
               "first_audio_ms_median": sorted(r["total_ms"] for r in rows)[len(rows) // 2]}
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\npeak {peak} MiB of 16376 · headroom {summary['headroom_mib']} MiB · "
          f"median first audio {summary['first_audio_ms_median']} ms")


if __name__ == "__main__":
    main()

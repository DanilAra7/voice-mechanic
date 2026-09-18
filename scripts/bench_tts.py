#!/usr/bin/env python3
"""Benchmark a TTS candidate the way the voice agent will actually use it.

The pipeline hands the synthesiser one sentence at a time, so the number that matters is how
long the driver waits for the FIRST sentence to become audible — not throughput on a paragraph.
Real-time factor matters second: below 1.0 the synthesiser keeps up with the conversation.

    python scripts/bench_tts.py --engine chatterbox --out-dir data/cache/tts_samples
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

# Lines the mechanic actually says: short, spoken, with the numbers and codes that trip up TTS.
LINES = [
    "Stop driving and pull over as soon as it is safe.",
    "Your coolant temperature is one hundred twenty four degrees Celsius.",
    "That code is P zero one seven one, which means the mixture is too lean.",
    "Let me check what other mechanics say about this.",
    "Short term fuel trim is sitting at seventeen percent at idle.",
    "It drops back to about four percent once you are cruising.",
    "That pattern usually points to a vacuum leak after the mass airflow sensor.",
    "Check the P C V hose and the intake boot for cracks.",
    "I would not drive it far until that is sorted.",
    "Do you want me to walk you through checking it yourself?",
]


def gpu_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def load_chatterbox(device: str):
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)

    def synth(text: str):
        wav = model.generate(text)
        return wav.squeeze(0).detach().cpu().numpy(), model.sr

    return synth


def load_vibevoice(device: str):
    import torch
    from transformers import AutoProcessor, VibeVoiceForConditionalGeneration

    repo = "microsoft/VibeVoice-Realtime-0.5B"
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    ).eval()

    def synth(text: str):
        inputs = processor(text=[f"Speaker 0: {text}"], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, tokenizer=processor.tokenizer, max_new_tokens=None)
        audio = out.speech_outputs[0].float().cpu().numpy().squeeze()
        return audio, 24000

    return synth


def load_kyutai(device: str):
    """Kyutai streams audio frames, so we can time the first frame instead of the whole sentence."""
    import numpy as np
    import torch
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

    info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device=torch.device(device))
    voice = tts.get_voice_path("expresso/ex03-ex01_happy_001_channel1_334s.wav")
    cond = tts.make_condition_attributes([voice], cfg_coef=2.0)

    def synth(text: str):
        entries = tts.prepare_script([text], padding_between=1)
        pcms, first = [], None
        started = time.monotonic()

        def on_frame(frame):
            nonlocal first
            if (frame != -1).all():
                pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                if first is None:
                    first = time.monotonic() - started
                pcms.append(np.clip(pcm[0, 0], -1, 1))

        with tts.mimi.streaming(1):
            tts.generate([entries], [cond], on_frame=on_frame)
        audio = np.concatenate(pcms, axis=-1) if pcms else np.zeros(1, dtype="float32")
        return audio, tts.mimi.sample_rate, first

    return synth


LOADERS = {"chatterbox": load_chatterbox, "vibevoice": load_vibevoice, "kyutai": load_kyutai}


def call(synth, text):
    """Adapters return (audio, sr) or (audio, sr, seconds_to_first_frame)."""
    out = synth(text)
    return out if len(out) == 3 else (*out, None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=sorted(LOADERS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="data/cache/tts_samples")
    p.add_argument("--results", default="evals/results/tts")
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results).mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf

    baseline = gpu_mib()
    t0 = time.monotonic()
    synth = LOADERS[args.engine](args.device)
    load_s = round(time.monotonic() - t0, 1)
    after_load = gpu_mib()
    print(f"[{args.engine}] loaded in {load_s}s, VRAM {after_load - baseline} MiB", flush=True)

    rows, peak = [], after_load
    for i, line in enumerate(LINES):
        t = time.monotonic()
        audio, sr, first_frame_s = call(synth, line)
        synth_s = time.monotonic() - t
        peak = max(peak, gpu_mib())
        audio = np.asarray(audio, dtype="float32")
        dur_s = len(audio) / sr
        sf.write(out_dir / f"{i:02d}.wav", audio, sr)
        rows.append(
            {
                "line": line,
                "synth_s": round(synth_s, 3),
                "audio_s": round(dur_s, 3),
                "first_audio_s": round(first_frame_s, 3) if first_frame_s else None,
                "rtf": round(synth_s / dur_s, 3) if dur_s else None,
            }
        )
        ttfa = f"{first_frame_s * 1000:5.0f} ms to first audio · " if first_frame_s else ""
        print(
            f"  {i:02d} {ttfa}{synth_s * 1000:6.0f} ms total for {dur_s:5.2f}s audio (rtf {synth_s / dur_s:.2f})",
            flush=True,
        )

    warm = rows[1:] or rows  # the first call pays CUDA warm-up; warm it at boot, not on a driver
    ttfas = [r["first_audio_s"] for r in warm if r["first_audio_s"]]
    first = rows[0]["synth_s"] * 1000
    rtfs = [r["rtf"] for r in warm if r["rtf"]]
    summary = {
        "engine": args.engine,
        "load_seconds": load_s,
        "vram_after_load_mib": after_load - baseline,
        "vram_peak_mib": peak - baseline,
        "cold_first_sentence_ms": round(first),
        "sentence_ms_p50": round(statistics.median(r["synth_s"] for r in warm) * 1000),
        "first_audio_ms_p50": round(statistics.median(ttfas) * 1000) if ttfas else None,
        "rtf_median": round(statistics.median(rtfs), 3),
        "rtf_worst": round(max(rtfs), 3),
        "lines": rows,
    }
    Path(args.results, f"{args.engine}.json").write_text(json.dumps(summary, indent=2))
    ttfa = f"first audio p50 {summary['first_audio_ms_p50']} ms · " if summary["first_audio_ms_p50"] else ""
    print(
        f"\n[{args.engine}] {ttfa}median sentence {summary['sentence_ms_p50']} ms · "
        f"cold start {summary['cold_first_sentence_ms']} ms · rtf med {summary['rtf_median']} · "
        f"VRAM peak {summary['vram_peak_mib']} MiB"
    )
    print(f"[{args.engine}] samples in {out_dir}")


if __name__ == "__main__":
    main()

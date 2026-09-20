#!/usr/bin/env python3
"""Compare voice samples and sampling temperatures for the mechanic's delivery.

The voice we shipped on day 3 is an actor performing "happy". It sells a burnt smell with the
same lift as a clean bill of health, which is wrong for the job, and enthusiasm costs time:
excited speech stretches vowels and leaves gaps. This sweeps the same speaker acted in calmer
moods and a couple of temperatures, and reports the two things that can be counted — how many
seconds of audio the driver has to sit through, and how fast it was produced. The part that
cannot be counted is why it writes every take to disk: somebody has to listen.

Run with the language model stopped; loading two models at once will not fit on the card.

    python scripts/bench_voice_style.py --out-dir data/cache/voice_style
"""

import argparse
import json
import statistics
import time
from pathlib import Path

# The same lines bench_tts.py uses, so numbers stay comparable across the project's history.
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

# Same speaker as the one in production (ex03) wherever possible, so what changes is the mood and
# not the person. The last one is a different speaker's neutral read, kept as a control.
VOICES = {
    "happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "calm": "expresso/ex03-ex01_calm_001_channel1_1143s.wav",
    "narration": "expresso/ex03-ex02_narration_001_channel1_674s.wav",
    "enunciated": "expresso/ex03-ex01_enunciated_001_channel1_388s.wav",
    "default_other": "expresso/ex01-ex02_default_001_channel1_168s.wav",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voices", default=",".join(VOICES))
    p.add_argument("--temps", default="0.6,0.4")
    p.add_argument("--out-dir", default="data/cache/voice_style")
    p.add_argument("--results", default="evals/results/voice_style.json")
    args = p.parse_args()

    import soundfile as sf
    import torch

    from mechanic.voice.tts import Synthesiser

    names = [n.strip() for n in args.voices.split(",") if n.strip()]
    temps = [float(t) for t in args.temps.split(",") if t.strip()]
    out_root = Path(args.out_dir)
    rows = []

    for temp in temps:
        synth = Synthesiser(temp=temp)
        synth.warm_up()  # the first sentence of a fresh model pays CUDA warm-up
        for name in names:
            synth.voice = VOICES[name]
            synth._cond = synth._tts.make_condition_attributes(
                [synth._tts.get_voice_path(synth.voice)], cfg_coef=synth.cfg_coef
            )
            out_dir = out_root / f"{name}_t{temp}"
            out_dir.mkdir(parents=True, exist_ok=True)
            takes = []
            for i, line in enumerate(LINES):
                started = time.monotonic()
                speech = synth.say(line, use_cache=False)
                sf.write(out_dir / f"{i:02d}.wav", speech.audio, speech.sample_rate)
                takes.append(
                    {
                        "line": line,
                        "audio_s": round(speech.duration_s, 3),
                        "synth_s": round(time.monotonic() - started, 3),
                        "first_frame_ms": round(speech.first_frame_ms) if speech.first_frame_ms else None,
                        # Seconds of speech per character of text: the pace, independent of length.
                        "s_per_char": round(speech.duration_s / len(line), 4),
                    }
                )
            audio_s = sum(t["audio_s"] for t in takes)
            row = {
                "voice": name,
                "sample": VOICES[name],
                "temp": temp,
                "audio_s_total": round(audio_s, 2),
                "audio_s_median": round(statistics.median(t["audio_s"] for t in takes), 2),
                "s_per_char_median": round(statistics.median(t["s_per_char"] for t in takes), 4),
                "first_frame_ms_p50": round(statistics.median(t["first_frame_ms"] for t in takes)),
                "rtf_median": round(statistics.median(t["synth_s"] / t["audio_s"] for t in takes), 3),
                "takes": takes,
            }
            rows.append(row)
            print(
                f"{name:14s} t={temp}  {row['audio_s_total']:6.2f}s of speech  "
                f"median {row['audio_s_median']:5.2f}s  {row['s_per_char_median']:.4f} s/char  "
                f"first frame {row['first_frame_ms_p50']:4d} ms  rtf {row['rtf_median']:.2f}",
                flush=True,
            )
        del synth
        torch.cuda.empty_cache()

    base = next((r for r in rows if r["voice"] == "happy" and r["temp"] == temps[0]), rows[0])
    for row in rows:
        row["vs_baseline_pct"] = round((row["audio_s_total"] / base["audio_s_total"] - 1) * 100, 1)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({"baseline": base["voice"], "rows": rows}, indent=2))

    print(f"\nagainst {base['voice']} t={base['temp']} ({base['audio_s_total']}s):")
    for row in sorted(rows, key=lambda r: r["audio_s_total"]):
        print(f"  {row['voice']:14s} t={row['temp']}  {row['vs_baseline_pct']:+5.1f}% speech time")
    print(f"\nsamples in {out_root}, numbers in {args.results}")


if __name__ == "__main__":
    main()

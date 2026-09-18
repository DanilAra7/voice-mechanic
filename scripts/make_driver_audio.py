#!/usr/bin/env python3
"""Speak the benchmark questions, so latency is measured against something a driver would say.

Needs the synthesiser, so it runs on the rented GPU:
    uv run python scripts/make_driver_audio.py
"""

from pathlib import Path

import soundfile as sf

from mechanic.data.common import ROOT
from mechanic.voice.tts import Synthesiser

QUESTIONS = [
    "My temperature gauge is climbing and it is still going up.",
    "The car shakes at idle and the check engine light came on.",
    "I am getting a code P zero one seven one, what does that mean?",
    "Is this a common problem on my car?",
    "How do I check the coolant level?",
    "My battery light keeps flickering when I slow down.",
]


def main() -> None:
    out = ROOT / "evals" / "audio" / "driver"
    out.mkdir(parents=True, exist_ok=True)
    tts = Synthesiser()
    for i, line in enumerate(QUESTIONS):
        speech = tts.say(line)
        sf.write(out / f"{i:02d}.wav", speech.audio, speech.sample_rate)
        print(f"{i:02d} {speech.duration_s:4.1f}s  {line}", flush=True)
    print(f"\nwrote {len(QUESTIONS)} clips to {out}")


if __name__ == "__main__":
    main()

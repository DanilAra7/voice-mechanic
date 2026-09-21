#!/usr/bin/env python3
"""Put the driver in a running car and see whether the recogniser still hears them.

Every accuracy number on this project was measured in a quiet room, which is the one place a
driver never is. This mixes the real recordings with engine and road noise at a chosen
signal-to-noise ratio, writes them beside their transcripts, and `bench_wer.py` scores them the
same way it scores the clean ones.

The noise is synthetic and says so: low-frequency rumble for the engine, broadband hiss for the
road, and a slow wander in level so it is not a constant the recogniser can simply subtract. It
is not a recording of a real garage, so read the result as "how it degrades", not as a promise
about any particular workshop. What it does do is cost nothing and stay reproducible.

    python scripts/make_noisy_audio.py --snr 20 --snr 10
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

RNG_SEED = 7


def engine_and_road(samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Rumble plus hiss plus a slow swell, which is what a cabin sounds like at speed."""
    white = rng.standard_normal(samples).astype("float32")
    # A one-pole low pass repeated a few times: cheap pink-ish rumble under about 200 Hz.
    rumble = white.copy()
    alpha = np.exp(-2 * np.pi * 120 / sample_rate)
    for _ in range(3):
        out = np.empty_like(rumble)
        acc = 0.0
        for i, x in enumerate(rumble):
            acc = alpha * acc + (1 - alpha) * x
            out[i] = acc
        rumble = out
    rumble /= np.abs(rumble).max() or 1.0
    hiss = white / (np.abs(white).max() or 1.0)
    noise = 0.8 * rumble + 0.2 * hiss
    # Engine load and passing traffic: the level is not constant, so the recogniser cannot
    # estimate it once and subtract it.
    t = np.arange(samples) / sample_rate
    noise *= 1.0 + 0.3 * np.sin(2 * np.pi * 0.11 * t)
    return noise.astype("float32")


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(speech**2)) or 1e-12
    noise_power = float(np.mean(noise**2)) or 1e-12
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    return np.clip(speech + noise * scale, -1.0, 1.0).astype("float32")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="evals/audio/real")
    p.add_argument("--snr", type=float, action="append", help="signal-to-noise ratio in dB; repeatable")
    args = p.parse_args()

    source = Path(args.source)
    clips = sorted(source.glob("*.wav"))
    if not clips:
        raise SystemExit(f"no recordings in {source}")

    for snr in args.snr or [20.0, 10.0]:
        out_dir = source.parent / f"{source.name}_snr{int(snr)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(RNG_SEED)
        for clip in clips:
            audio, sample_rate = sf.read(clip, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            noisy = mix_at_snr(audio, engine_and_road(len(audio), sample_rate, rng), snr)
            sf.write(out_dir / clip.name, noisy, sample_rate)
            reference = clip.with_suffix(".txt")
            if reference.exists():
                shutil.copy(reference, out_dir / reference.name)
        print(f"{len(clips)} clips at {snr:.0f} dB SNR -> {out_dir}")


if __name__ == "__main__":
    main()

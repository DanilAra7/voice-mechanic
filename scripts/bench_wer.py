#!/usr/bin/env python3
"""How often recognition gets the words wrong — the one ASR number we never had.

Speed was measured on day 4 and accuracy was not, so "the transcript looked clean" was doing
the work of a number. It was not clean: a run on 2026-09-20 produced "the car shakes IT idle"
and "IT'S I am getting a code P0171". Neither broke the answer, but neither was measured.

Every clip needs a `.txt` beside it holding what was actually said:

    evals/audio/driver/00.wav
    evals/audio/driver/00.txt

    uv run python scripts/bench_wer.py --audio evals/audio/driver
    uv run python scripts/bench_wer.py --audio evals/audio/real     # real voices, real room

Both sides are normalised the same way before comparing: lower case, no punctuation, contractions
written out, and digits spoken as words joined up, so "p zero one seven one" and "P0171" count as
the same thing — the agent's tools accept either, and "it's" against "it is" is a disagreement
about spelling rather than about what was said. Both figures are printed, because expanding
contractions flatters the score and you should be able to see by how much.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import soundfile as sf

from mechanic.voice.asr import SAMPLE_RATE, Recognizer, resample, to_mono
from mechanic.voice.vocabulary import repair

CONTRACTIONS = {
    "it's": "it is", "i'm": "i am", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "won't": "will not", "isn't": "is not",
    "i've": "i have", "you're": "you are", "there's": "there is", "that's": "that is",
    "i'd": "i would", "i'll": "i will", "we're": "we are", "wasn't": "was not",
    "haven't": "have not", "hasn't": "has not", "wouldn't": "would not", "let's": "let us",
}
DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def normalise(text: str, expand: bool = True) -> list[str]:
    text = text.lower().replace("\u2019", "'")
    if expand:
        for short, long in CONTRACTIONS.items():
            text = re.sub(rf"\b{re.escape(short)}\b", long, text)
    words = re.sub(r"[^\w\s]", " ", text).split()
    out: list[str] = []
    for word in words:
        digit = DIGITS.get(word)
        # A run of spoken digits after a letter is a trouble code being read aloud.
        if digit and out and re.fullmatch(r"[a-z]\d*", out[-1]):
            out[-1] += digit
        elif digit and out and re.fullmatch(r"\d+", out[-1]):
            out[-1] += digit
        else:
            out.append(digit or word)
    return out


def edits(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance, the plain two-row version."""
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", default="evals/audio/driver")
    p.add_argument("--threads", type=int, default=None, help="override the recogniser's own choice")
    p.add_argument("--out", default="evals/results/wer.json")
    args = p.parse_args()

    clips = sorted(Path(args.audio).glob("*.wav"))
    if not clips:
        raise SystemExit(f"no .wav files in {args.audio}")
    recognizer = Recognizer(threads=args.threads)
    print(f"{len(clips)} clips · {recognizer.threads} threads\n")

    rows, total_edits, total_words = [], 0, 0
    char_edits, char_count = 0, 0
    strict_edits, strict_words = 0, 0
    for path in clips:
        truth_path = path.with_suffix(".txt")
        if not truth_path.exists():
            print(f"{path.name}: skipped, no {truth_path.name}")
            continue
        audio, sr = sf.read(path, dtype="float32")
        transcript = recognizer.transcribe(resample(to_mono(audio), sr, SAMPLE_RATE))
        reference = normalise(truth_path.read_text(encoding="utf-8"))
        # Scored through the same repair the live session applies, or the number would flatter
        # a pipeline the driver never talks to.
        heard = normalise(repair(transcript.text))
        wrong = edits(reference, heard)
        total_edits += wrong
        total_words += len(reference)
        char_edits += edits(list(" ".join(reference)), list(" ".join(heard)))
        char_count += len(" ".join(reference))
        strict_ref = normalise(truth_path.read_text(encoding="utf-8"), expand=False)
        strict_edits += edits(strict_ref, normalise(transcript.text, expand=False))
        strict_words += len(strict_ref)
        rows.append(
            {
                "clip": path.name,
                "reference": " ".join(reference),
                "heard": " ".join(heard),
                "word_errors": wrong,
                "words": len(reference),
                "decode_ms": round(transcript.decode_ms),
            }
        )
        mark = "  " if not wrong else "->"
        print(f"{path.name}: {wrong}/{len(reference)} wrong · {round(transcript.decode_ms)} ms")
        if wrong:
            print(f"   said  {' '.join(reference)}")
            print(f"{mark} heard {' '.join(heard)}")

    if not total_words:
        raise SystemExit("nothing to score: every clip is missing its .txt")
    wer = 100 * total_edits / total_words
    cer = 100 * char_edits / char_count
    strict = 100 * strict_edits / strict_words
    clean = sum(1 for r in rows if not r["word_errors"])
    print(f"\nWER {wer:.1f}%  ·  CER {cer:.1f}%  ·  {clean}/{len(rows)} clips word-perfect")
    print(f"WER {strict:.1f}% counting a contraction as a mistake")
    print(f"decode median {statistics.median(r['decode_ms'] for r in rows):.0f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"wer": round(wer, 2), "wer_strict": round(strict, 2), "cer": round(cer, 2), "clips": rows}, indent=2
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

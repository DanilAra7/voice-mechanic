# Spoken driver questions

The fixed input the end-to-end latency figures are measured against, so a run today can be
compared with a run last week. Synthesised with the same Kyutai voice the agent speaks in —
convenient and repeatable, but also optimistic: a real driver in a garage brings an accent,
background noise and a cheaper microphone. Treat these numbers as a floor.

Regenerate (on a machine with the synthesiser): `uv run python scripts/make_driver_audio.py`
Measure: `uv run python scripts/bench_voice.py --audio evals/audio/driver`

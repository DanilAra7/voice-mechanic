# Runbook: renting a GPU and running the benchmarks

Written on day 3 to choose the **language model + voice** pair that fits **16 GB of VRAM** with
the smallest latency, and kept since as the runbook for every rented box. Where the plan and
what happened differ, both are here: the plan is what we expected to be true.

## 1. Renting on Vast.ai

- Filter on **GPU RAM = 16 GB** (RTX 4060 Ti 16GB / RTX 4080 / RTX 5070 Ti / A4000), a `Direct`
  connection, and enough disk for the models (35 GB is enough; 60 GB is comfortable).
- A card with exactly 16 GB is itself the proof of the budget: nothing larger physically fits.
- **Filter on `inet_down_cost` as well.** Hourly prices differ by cents; bandwidth prices differ
  fifteenfold, and 95% of one session's bill was downloading models. It is in the output of
  `vastai search offers --raw`.
- **Sort by `cpu_cores_effective`, not `cpu_cores`.** The latter is the host's; ours is the
  share. The first instance was picked on "8+ cores" and had seven effective ones.
- Image: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`. **Ubuntu 24.04 is not optional** — the
  prebuilt llama.cpp binaries need glibc 2.38, and the `pytorch/pytorch` image ships 2.35 with no
  compiler to build from source.
- Always pass `--cancel-unavail`. Without it, an offer that has been taken in the meantime
  silently creates a **stopped** instance and returns `success: False`.

## 2. Bringing the box up

One script does the whole machine — packages, llama.cpp, the models, the ASR weights, and a
dead-man switch that stops the instance after a few hours so a forgotten box cannot drain the
budget:

```bash
ssh -p <port> root@<host> 'bash -s' < scripts/gpu_up.sh
```

The dead-man switch uses the container's own `CONTAINER_API_KEY` from `/proc/1/environ`, so the
account key never lands on a rented machine.

Then the code and the prebuilt index:

```bash
git archive --format=tar HEAD | gzip -c | ssh … 'cd /workspace/mechanic && tar xzf -'
tar czf - data/processed data/index | ssh … 'cd /workspace/mechanic && tar xzf -'
```

`data/index/dense.npy` is about 86 MB and takes minutes to rebuild, so it travels rather than
being recomputed. Rebuilding it on the box, if needed:

```bash
uv run python -m mechanic.knowledge.search build --dense-only    # 56,207 passages
```

fastembed runs on the CPU through onnxruntime, so this wants vCPUs rather than the GPU — and it
wants an explicit `--threads`, because onnxruntime counts the **host's** cores inside a
container and will start 98 threads on a 6.7-core share.

Finally, everything that makes a demo: model, API behind a key, a car on the ramp, the public
address, and a warm-up so the first person to speak does not pay for it:

```bash
MECHANIC_ACCESS_KEY=… MECHANIC_NGROK_DOMAIN=… ssh … 'bash -s' < scripts/demo_up.sh
```

At the end of a session: `vastai stop instance <id>`, and write the state into
`docs/PROGRESS.md`. **Stop, do not destroy** — see DECISIONS.md for the arithmetic.

## 3. The language models that were compared

All four ran against the same scenarios through `mechanic.evals.run`, recording peak VRAM, time
to first token, tokens per second, tool accuracy, content accuracy, and p50/p95 of the first
sentence. The table of results is in NOTES.md.

| # | Model | Engine |
|---|---|---|
| 1 | gpt-oss-20b (MXFP4) | llama.cpp |
| 2 | Qwen3-30B-A3B-Instruct-2507, Q2 and Q3 | llama.cpp |
| 3 | Qwen3-30B-A3B-Instruct-2507, Q4 with experts on the CPU | llama.cpp |
| 4 | Qwen3-14B Q5 | llama.cpp |

vLLM was in the original plan and was not used: llama.cpp's prebuilt CUDA binaries were running
within minutes, and the quantisations that mattered were GGUF.

## 4. The voices that were compared

The same ten mechanic's lines through each, measuring **time to the first audio chunk** (which is
what latency means here), real-time factor, VRAM, and quality by ear from the files written to
`data/cache/tts_samples/`.

Candidates: Chatterbox and Chatterbox Turbo, Kyutai TTS 1.6B, Orpheus 3B quantised,
VibeVoice-Realtime-0.5B, Qwen3-TTS.

## 5. The rule for choosing

Take the most capable language model for which:

- the whole stack — model, voice, buffers — is at or under 16 GB;
- the first sentence arrives in about 500 ms of text-only latency;
- tool accuracy is at least 90% on the scenarios.

If nothing passes, drop the context to 8k and try again.

That is what happened: 8k context, `reasoning_effort: low`, gpt-oss-20b. The model that scored
better on the scenarios, Qwen3-30B-A3B Q2, left no room for any voice at all — and on this
budget the voice chooses the model.

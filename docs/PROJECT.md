# Project: the voice mechanic

## The brief, as given

- A voice agent on an **open model**. No LLM APIs.
- Runs on a server with a GPU limit. The project's own constraint: **the whole stack ≤ 16 GB of
  VRAM**.
- **Minimise latency.**
- A genuine **agent with several tools**, not a task tracker with a voice.
- **English** only.
- The **smartest model that fits**. Quantisation is allowed.
- The deliverable is a **website** where a person talks to the agent and **measures the latency
  themselves**.
- Deadline: **2026-09-24** (one week from 2026-09-17).

## Constraints we started with

- No server of our own, so Vast.ai by the hour. A card with exactly 16 GB, which enforces the
  budget physically rather than by good intentions.
- Development machine: MacBook Air M4, 16 GB unified memory, macOS 26.6. Has `uv`, `node 24`,
  `git`. No `brew`, no `docker`, no `ollama`.
- The user's real car is an **Audi A4** with an Android phone, Torque Pro and an OBD adapter, but
  the demo uses the **simulator only** — a demo that needs a particular car in a particular
  driveway is not a demo.
- The person judging the work has never seen the project. Everything has to make sense without a
  guided tour.

## Architecture, as built

```
Browser ── microphone ──► WebSocket ──► Silero VAD ──► Parakeet ASR (CPU)
                                                            │
                                                            ▼
                                            safety rules (regex, no model)
                                                            │
                                                            ▼
    speaker ◄── audio frames ◄── Kyutai TTS ◄── gpt-oss-20b + 8 tools ──► SQLite sensors
                                   (GPU)              (GPU)              hybrid search index
```

The browser is served by the same FastAPI process as the API, because a browser hands over a
microphone only on a secure origin and one origin is one fewer thing to get wrong.

Everything resident peaks at **15,819 MiB of 16,376**.

The original draft had Pipecat for orchestration and a Cloudflare Tunnel for the public address.
Both were dropped after measurement; see DECISIONS.md for why, and NOTES.md for what the
Cloudflare tunnel actually did on this host.

## Components, and what they became

| Part | Candidates considered | Chosen |
|---|---|---|
| Language model | gpt-oss-20b MXFP4 · Qwen3-30B-A3B · Qwen3-14B AWQ | **gpt-oss-20b**, the only one a streaming voice fits beside |
| Speech synthesis | Chatterbox · Kyutai TTS 1.6B · Orpheus 3B · VibeVoice-Realtime | **Kyutai**, the only one that streams |
| Speech recognition | Parakeet TDT 0.6B via sherpa-onnx on the CPU | **as planned**, offline rather than streaming |
| Turn detection | Silero VAD, Pipecat Smart-Turn | **Silero + push-to-talk** |
| Orchestration | Pipecat, or our own | **our own**, over WebSocket |
| Search | BM25 + embeddings on the CPU, filtered by car | **as planned**, fused with RRF |

## The agent's tools

| Tool | Where its data comes from |
|---|---|
| `get_vehicle` / `set_vehicle` | session state |
| `read_live_data` / `get_sensor_trend` | the simulator, speaking the Torque web-upload protocol into SQLite |
| `lookup_dtc` | a local OBD-II code database |
| `search_how_to` | carcarekiosk.com |
| `search_owner_reports` | startmycar.com |
| `search_forum` | the Motor Vehicle Maintenance & Repair Stack Exchange dump |

While a slow tool runs the agent says a fixed filler line ("Let me check that"), synthesised at
boot so it costs nothing. Safety rules run in code before the model sees the question.

## The five cars

| id | Car | Engine |
|---|---|---|
| `audi_a4_b8` | Audi A4 B8, 2009–2016 (the user's own car) | 2.0 TFSI I4 turbo |
| `honda_accord_9` | Honda Accord 9th gen, 2013–2017 | 2.4 I4 |
| `ford_f150_13` | Ford F-150 13th gen, 2015–2020 | 2.7 EcoBoost V6 |
| `honda_civic_10` | Honda Civic 10th gen, 2016–2021 | 2.0 I4 |
| `toyota_corolla_11` | Toyota Corolla 11th gen, 2014–2019 | 1.8 I4 |

Chosen by how much real data exists for each, not by preference. See DECISIONS.md and the data
sources table in NOTES.md.

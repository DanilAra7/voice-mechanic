# Voice Mechanic

A self-hosted voice agent that diagnoses car problems: it listens, reads the car's live OBD-II
data, looks things up, and answers out loud. No LLM APIs — every model runs on one GPU with a
hard budget of **16 GB of VRAM**, and the whole system is optimised for how fast it starts
talking back.

> **Status: work in progress** (day 2 of 7). The agent, its tools and the knowledge base are
> built and tested; the speech pipeline, the web UI and the latency benchmarks land next.
> This README is rewritten at the end with the measured numbers.

## What it does

Ask it "my temperature gauge is climbing, what's wrong?" and it will read the live coolant
temperature, notice the trend, pull the trouble code, check what other owners of *your* model
report, and tell you whether to keep driving.

Eight tools:

| Tool | What it does |
|---|---|
| `get_vehicle` / `set_vehicle` | which car this conversation is about |
| `read_live_data` | current sensor values + active trouble codes, with normal ranges attached |
| `get_sensor_trend` | how one sensor moved over the last minutes (rising vs steady) |
| `lookup_dtc` | 9,533 OBD-II codes with causes ranked by likelihood |
| `search_how_to` | step-by-step maintenance for this exact generation |
| `search_owner_reports` | problems other owners of this model reported |
| `search_forum` | mechanics Q&A threads for diagnosis reasoning |

## Cars

Five generations are supported end to end: Audi A4 B8 (2009–2016), Honda Accord 9th (2013–2017),
Ford F-150 13th (2015–2020), Honda Civic 10th (2016–2021), Toyota Corolla 11th (2014–2019).

## Live car data, without a car

The car's sensors arrive over the **Torque Pro web-upload protocol** (`GET /torque?k5=…&kc=…`,
answered with `OK!`). A simulator speaks that same protocol, so the demo needs no vehicle: pick a
car, inject a fault (overheating, misfire, vacuum leak, dirty MAF, stuck thermostat, failing
alternator) and watch the agent work it out from the data. Plugging in a real phone running
Torque Pro needs no code change — just point its Webserver URL at `/torque`.

## Knowledge base

| Source | Content | Licence |
|---|---|---|
| Motor Vehicle Maintenance & Repair Stack Exchange (official data dump) | 20,996 Q&A threads | CC BY-SA |
| carcarekiosk.com | 238 maintenance guides for the five generations | scraped politely, linked back |
| startmycar.com | 5,430 English owner problem reports | scraped politely, linked back |
| [OBDex](https://github.com/foerbsnavi/OBDex) | 9,533 trouble codes with causes and symptoms | CC0 |

Search is hybrid: BM25 for exact jargon ("P0171", "PCV"), dense vectors for paraphrased
symptoms, fused with Reciprocal Rank Fusion and boosted toward the driver's own car. A query
takes about 7 ms over 56,207 passages.

## Running it

```bash
uv sync
uv run pytest -q                                          # 57 tests, no model needed
uv run uvicorn mechanic.server:app --port 8000            # backend + Torque receiver
uv run python -m mechanic.torque.simulator --vehicle audi_a4_b8 --fault vacuum_leak
```

Building the knowledge base (downloads ~100 MB, scrapes at 1 request/second):

```bash
uv run python -m mechanic.data.stackexchange
uv run python -m mechanic.data.carcarekiosk
uv run python -m mechanic.data.startmycar
uv run python -m mechanic.knowledge.search build
```

Evaluating a model against the 32 scenarios:

```bash
uv run python -m mechanic.evals.run --model <name> --base-url http://127.0.0.1:8001/v1
```

## Repo layout

```
src/mechanic/torque/     Torque protocol: receiver, SQLite store, vehicle simulator
src/mechanic/knowledge/  trouble-code database and hybrid search index
src/mechanic/agent/      tools, system prompt, streaming agent loop
src/mechanic/evals/      scenario runner (tool choice, answer content, latency)
src/mechanic/data/       knowledge-base builders (dump converter + two scrapers)
docs/                    design decisions, progress log, GPU runbook, engineering notes
```

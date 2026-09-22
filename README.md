# Voice Mechanic

A self-hosted voice agent that diagnoses car faults. You talk to it, it reads your car's live
OBD-II sensors, looks things up, and answers out loud.

No LLM APIs. Every model — speech recognition, the language model, speech synthesis — runs on
one rented GPU inside a hard budget of **16 GB of VRAM**, and the whole system is built around
one number: how long you wait before it starts talking back.

**From the end of your sentence to the first sound of the answer: 849 ms.**

---

## Table of contents

- [Try it](#try-it)
- [What it actually does](#what-it-actually-does)
- [How it is put together](#how-it-is-put-together)
- [Latency: where the wait goes](#latency-where-the-wait-goes)
- [Quality: what was measured and how](#quality-what-was-measured-and-how)
- [What it cannot do](#what-it-cannot-do)
- [Running it yourself](#running-it-yourself)
- [Repository layout](#repository-layout)
- [Data sources and licences](#data-sources-and-licences)

---

## Try it

The demo runs on a rented GPU that is not left switched on, so the link works when the box is
up. When it is, open it in Chrome, **allow the microphone**, and **hold the button while you
speak**.

```
https://cruncher-wieldable-pork.ngrok-free.dev/app/?k=<key>
```

Two things to know before you start:

1. The free tunnel shows its own warning page first. Click **Visit Site** once.
2. **Start a car in the Garage panel before you ask about a symptom.** With no car running the
   agent has no sensors to read and will keep telling you to check your adapter — which is
   correct and sounds stupid. The page warns you when this is the case.

### A demo that shows the point in ninety seconds

1. In **Garage**, pick *Audi A4 B8*, fault **vacuum leak**, mode **idle**, speed ×10. Press
   *Start the car*. Watch the short-term fuel trim climb past 15%.
2. Hold the button and say: *"My car idles rough and there's a hissing noise, can you check the
   data?"*
3. It reads the live sensors, sees the lean mixture and the P0171 code, and tells you it is most
   likely a vacuum leak after the mass airflow sensor.

**Nobody told it what was wrong.** The fault you injected is deliberately hidden from the agent
— it is in the browser and in the simulator, never in the prompt or in any tool result. The
*Reveal* button shows you the answer so you can mark its work.

Then try the other half:

- Say *"I can smell petrol inside the car"* — the warning comes back in **233 ms**, before the
  language model has produced a single token. Safety rules are code, not a request in a prompt.
- Say *"my OBD adapter is broken, don't use it"* — it stops reading sensors and works from what
  you tell it. What the driver says about their own hardware outranks the sensor.

---

## What it actually does

It is an agent, not a chatbot with a voice. Every answer that needs a fact goes and gets it.

| Tool | What it returns |
|---|---|
| `read_live_data` | current sensor values and active trouble codes, each labelled normal / too lean / OVERHEATING against that engine's healthy range, with the direction it has been moving |
| `get_sensor_trend` | how one sensor moved over the last minutes, judged against the band it belongs in |
| `lookup_dtc` | 9,533 OBD-II codes with causes ranked by likelihood; understands codes spoken aloud ("P zero one seven one") |
| `search_how_to` | maintenance steps for this exact model generation |
| `search_owner_reports` | what other owners of this model complain about |
| `search_forum` | mechanics' Q&A threads, for reasoning about symptoms |
| `get_vehicle` / `set_vehicle` | which car this conversation is about |

Five car generations are supported end to end: **Audi A4 B8** (2009–2016), **Honda Accord 9th**
(2013–2017), **Ford F-150 13th** (2015–2020), **Honda Civic 10th** (2016–2021), **Toyota Corolla
11th** (2014–2019). They were chosen by how much real data exists for them, not by taste.

### Live car data, without a car

Sensor data arrives over the **Torque Pro web-upload protocol** — the same `GET /torque?k5=…&kc=…`
that the Android app sends, answered with `OK!`. A simulator speaks that protocol over real HTTP,
so the demo needs no vehicle: pick a car, inject a fault, and the readings drift the way the
physics says they should.

Six faults: overheating, misfire, vacuum leak, dirty MAF, stuck thermostat, failing alternator.
They develop over time rather than switching on — a vacuum leak shows 16.4% short-term fuel trim
at idle and 6.5% in town, which is exactly the pattern a mechanic uses to tell it from a bad
sensor.

**A real phone works with no code change.** Point Torque Pro's *Webserver URL* at `/torque` and
the agent reads your actual car.

---

## How it is put together

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

| Part | Choice | Why this one |
|---|---|---|
| Language model | **gpt-oss-20b**, MXFP4, 8k context, `reasoning_effort: low` | The only model that leaves room for a streaming voice beside it. Qwen3-30B-A3B scored better on scenarios (71.9% vs 65.6%) and no voice fits next to it. |
| Speech synthesis | **Kyutai TTS 1.6B** | It emits audio frames while still generating: first sound at 350 ms against 1,186 ms for a non-streaming model of the same quality. On a 16 GB card, the voice dictates the model. |
| Speech recognition | **Parakeet TDT 0.6B int8**, offline, on the **CPU** | 147 ms median, and it leaves the GPU entirely to the other two. The streaming build of the same model runs at 1.69× real time — it cannot keep up with speech. |
| Turn detection | **Silero VAD v5** + push-to-talk | Releasing a button is an exact end-of-turn. Hands-free pays roughly 0.4 s waiting for the silence to prove itself. |
| Transport | **WebSocket**, hand-written | Tunnels do not carry UDP, so not WebRTC. Pipecat was tried and dropped: the timings are part of what is being delivered and needed exact control. |
| Search | **BM25 + dense vectors**, fused with RRF, boosted toward the driver's car | Jargon like "P0171" needs exact matching; "it hesitates when I accelerate" needs meaning. 56k passages, ~6 ms a query. |

Everything is resident at once and peaks at **15,819 MiB of the card's 16,376**. The margin is
577 MiB, and it is not a comfortable one.

---

## Latency: where the wait goes

The site lets you measure it yourself: the panel times from **the moment you release the button**
to **the first audio sample your browser plays**, on your clock, not ours. Every turn is also
recorded server-side, so the numbers survive the tab being closed.

The panel cuts that wait into pieces **that do not overlap**, showing both the answer you just
heard and the median of your session:

| Row | What it is |
|---|---|
| `silence` | hands-free only: waiting for the pause after your sentence to prove it was a pause |
| `recognition` | Parakeet turning what you said into words, on the CPU |
| `model` | gpt-oss-20b deciding what to answer |
| `tools` | sensors, trouble codes and search — **only the part that ran before you heard anything** |
| `speech` | Kyutai turning the first sentence into sound |
| `turn hold` | the first sound held back until the turn is certain — see below |
| `network` | the wire both ways, plus the audio pipeline in your own tab |

Each piece is timed where it happens rather than subtracted from the one before it. That sounds
pedantic and is not: when a lookup will be slow the agent says *"Let me check that"* first and
keeps searching **while it talks**, so the first sentence lands before the tool has finished. Any
breakdown built by subtraction reports negative model time on exactly those turns. The panel
instead tells you how much tool time you never waited for.

### Holding the button

This is the mode the headline number is measured in. The clock starts the instant you let go.

| Stage | Median | What it is |
|---|---|---|
| Silence detection | **0 ms** | releasing the button is an exact end of turn; nothing to wait out |
| Recognition | **307 ms** | Parakeet on the CPU |
| Language model, tools included | **485 ms** | to the first sentence worth speaking; the tools themselves are **0 ms** at the median, 149 ms at worst |
| Synthesis to first sound | **57 ms** | near zero because the opening lines are synthesised at boot and replayed from cache |
| Turn hold | **0 ms** | the button already said the turn was over |
| **Server total** | **849 ms** | p95 1,733 ms, 15/15 turns completed |
| Network, Kyiv → Frankfurt | **60–71 ms** | round trip, measured in the browser |
| In the browser | **the rest** | see the honest note below |

End to end in a real browser in another country: **p50 about 1.5 s**.

### Hands-free

With no button, the end of your turn has to be *detected*, and that is the most expensive stage
in the system. It also happens **before every clock above starts** — the server's zero is the
moment the detector speaks up, not the moment you stopped talking.

- **Silence detection: 366–378 ms** of audio time, measured on three clips
  (`min_silence_s = 0.35` plus the detector's own windowing). Day 3 measured 507 ms wall-clock
  from the last sample, feeding the stream in real time; the difference is chunking and
  smoothing, and the honest range is **roughly 0.4 s**.
- **Turn hold: usually 0 ms.** The design is *work early, speak on confirmation* — recognition,
  the model and the voice all start on the short 0.35 s threshold, while the audio is held until
  a longer 0.55 s timer agrees the turn really ended. Because the work takes ~849 ms and the
  timer only 550 ms, the timer has almost always expired by the time there is anything to play.
  It bites only when the answer is faster than 550 ms — which is exactly the **safety** path, so
  a 233 ms warning is heard at about 1.0 s hands-free instead of 233 ms on the button.

So hands-free costs **roughly 0.4 s more**, putting it near **1.25 s** server-side and **1.9 s**
in a browser across a border. Why a threshold that long: ordinary spoken questions contain
pauses of 0.6–0.9 s. At 0.35 s alone, *"I am getting a code P0171, what does that mean?"* splits
into two turns and the agent answers the first half.

### The part that is still not attributed

The server says 849 ms; the browser says about 1,500 ms. The round trip through the tunnel
measures 60–71 ms, so **roughly 600 ms is unexplained**, and calling it "the network" would be a
guess. Ruled out by measurement rather than by argument:

- **The server falling behind the microphone.** The socket loop runs the detector on every
  incoming chunk before reading the next one, so a backlog would delay the button release behind
  queued audio. Measured: 0.006 ms per chunk, **0.003× real time**. No backlog.
- **Microphone buffering.** The browser posts every 128 samples — 8 ms.
- **Playback scheduling.** The browser stamps its clock when the first audio frame *arrives*, not
  when it is audible, so the 20 ms output buffer is not in the number.
- **A sample-rate mismatch.** The capture context is pinned to 16 kHz explicitly.

- **The socket loop being slow to notice the button.** `scripts/bench_transport.py` runs the real
  receive loop and the real detector over a real WebSocket, streaming audio at the rate a browser
  sends it: **0.4 ms** to dequeue the button release, **7.2 ms** to the first audio frame back.
  Remove the pacing and the same bench shows 41 ms, so it can see a backlog when one exists.

**The likeliest explanation is that the 600 ms was never one number.** 849 and 1,500 are medians
over *different sets of turns*: the server's stages were filed from a field that is empty on the
first turn of every session, so first turns dropped out of the stage medians while staying in the
client median — and the first turn after a start costs about 1,880 ms while the synthesiser
compiles its kernels. That is enough to manufacture most of the gap. The round trip is the other
half of the doubt: 60–71 ms was sampled by a timer that fires *between* turns, never during one.

Both are now instrumented instead of argued about: the browser reports the bytes still unsent
when the button came up and a round trip measured at that exact moment, and client and server
figures are finally recorded against the same turn. One real conversation settles it.

The answer itself has a **median length of 6.9 seconds**, down from 24 s before the length cap —
the wait before the first word is only half of what being kept waiting feels like.

Two things carry most of the win:

- **Fixed phrases are synthesised at startup.** "Let me check that" is the first thing you hear
  on most turns, and it costs nothing at conversation time. First sound went 1,496 → 1,031 ms.
- **Safety answers skip the model entirely.** 233 ms, by regular expression.

---

## Quality: what was measured and how

Everything below is reproducible from this repository. The numbers are not flattering
everywhere, and the places where the measurement itself is weak are marked as such.

### Recognition

| Condition | Word error rate | Clips word-perfect |
|---|---|---|
| Quiet room, real human voice | **2.2%** | 12 of 15 |
| 20 dB SNR (engine and road noise) | **3.9%** | 11 of 15 |
| 10 dB SNR | **4.4%** | 10 of 15 |

The noise is synthetic — low-frequency rumble, broadband hiss, a slow swell — and generated by
`scripts/make_noisy_audio.py`. Read it as "how it degrades", not as a promise about a particular
workshop.

Ordinary English was never the problem; the trade's own words were. `fuel trims` came back as
`field dreams`, and `the OBD adapter` as `the ad app server` — which is how a driver came to tell
the agent his adapter was broken and be asked about his readings three more times. The proper
cure is contextual biasing, which needs a `bpe.model` that `parakeet-tdt-0.6b-v2` does not ship.
What is here instead is a short table of mishearings actually observed, and it is labelled a
stopgap in the source.

### The agent, over 76 scenarios

Each scenario runs against a simulated car with a known fault and checks the tools called, the
content of the answer, and whether the text is speakable at all.

| | |
|---|---|
| Scenarios passed whole | **60 / 76 = 78.9%** |
| Correct tools chosen | **91.5%** |
| Voice-safe text (no markdown, URLs, invented codes) | **100%** |
| Tests | **147**, green |

By category: `owner_reports` and `safety` **100%**, `how_to` 88.9%, `conversation` 85.7%,
`multi_turn` 83.3%, `live_data` 71.4%, `dtc` 60%, `vehicle` 60%, **`forum` 57.1%** — the weakest,
and the one that resisted five attempts.

### Safety

The scenarios that matter most are the ones where being helpful is the wrong answer.

**11 of 11 pass by substance.** Two of them are counted as failures by the automatic check
because it wants the agent to repeat the word "airbag" back, and the agent says the right thing
without parroting. The scenario file was deliberately **not** edited after seeing that result.

Safety is not a request in a prompt. A symptom that can hurt someone is matched in code before
the model is consulted, and the refusal to drive is spoken first — 233 ms for "I smell petrol
inside the cabin", 331 ms for "my brake pedal goes almost to the floor". Warning lights for
systems that only matter in a crash get a mandated closing line appended after the answer.

### Truthfulness — and why this number is soft

The obvious metric is a trap. "Content accuracy" was measured for days by checking whether
expected phrases appeared in the answer. Reading all seventy passing answers by hand against the
fault the simulator was running showed what that hides:

- *"The voltage has dipped … but it has been steady"* scored a hit on the word **down**, on a
  truck with a failing alternator.
- A flat denial of carbon buildup on a direct-injection engine scored a hit on **carbon**.
- *"That is not a normal, everyday odour"* scored a hit on **normal**, in the one scenario whose
  whole point is not frightening a driver over a normal smell.
- Two how-to answers would have had a driver draining oil through the filler cap and
  disconnecting a battery before opening the hood.

Real correctness on that build was **58/76 = 76.3%**, against a reported 92.7%. The hand-read
verdicts are in `evals/content_gold.yaml`, one line of reasoning each, so they can be argued with.

An LLM judge now scores every answer against the injected fault (`mechanic.evals.judge`). On the
current build it says **80.3% contain no falsehood**. That figure comes with its own measurement:
the judge was scored against the hand-read set and agrees **74% exactly, 81% on correct-versus-not**.
Reading its verdicts shows what it gets wrong — it marks a correct refusal to discuss the weather
as a failure, and penalises asking the driver a clarifying question.

**So: about 80% of answers contain nothing false, measured with an instrument that is itself
noisy, and the noise is larger than the differences between recent builds.** That is the honest
statement. A single confident number here would be a worse one.

### Search

Measured against 390 relevance judgements nobody on this project made: when a question on
Stack Exchange is closed as a duplicate of another, a human has said those two are the same
problem.

| | R@1 | R@5 | R@10 | MRR@10 | Query |
|---|---|---|---|---|---|
| BM25 only | 14.9% | 30.0% | 39.5% | 0.219 | 1.5 ms |
| Dense only | 20.0% | 43.3% | 52.6% | 0.298 | 5.6 ms |
| **Both, fused** | **23.3%** | 42.3% | 51.8% | **0.309** | 6.2 ms |

R@5 of 42% is a **floor, not a verdict**: only the one thread a moderator linked counts as
correct, so another thread answering the question perfectly scores zero.

More useful is where the blame falls when a scenario fails. Across the failures: wrong tool
chosen 6, did not look anything up 4, generation 1, **retrieval 0**. The search has never been
the reason an answer was wrong, which is why no effort went into improving it.

---

## What it cannot do

Written down because a demo that hides these is worth less than one that names them.

**Measured and not good enough**

- The `forum` category answers 57% of the time. It is one weak category out of nine.
- About one answer in five contains something false, and the instrument measuring that is noisy.
- Two how-to answers now refuse to give steps at all — grounding trades usefulness for honesty,
  and this is the bill.

**Not tested at all**

- **A thirty-minute session.** Never run. This is the risk most likely to spoil a live demo.
- Long conversations. The context is 8k with no summarisation; a long enough conversation will
  overflow it.
- Two drivers at once. One card, one conversation.
- Voices other than the one the 2.2% word error rate was measured on.
- Cars outside the five. A driver with something else is told so.

**Known and deliberate**

- The GPU has 577 MiB of headroom. Restarting the API over a live model has produced
  `CUBLAS_STATUS_ALLOC_FAILED` once, mid-conversation.
- A cold start costs minutes, nearly all of it loading models. Bring the demo up before anyone
  is waiting — `scripts/demo_up.sh` does this and warms the models on a throwaway conversation.
- The public URL is protected by a key in the link and nothing else. There are no accounts and
  no personal data; it keeps out passers-by and is not pretending to be authentication.
- It is a demo, not a garage. The page says so above the conversation.

---

## Running it yourself

### Locally, without a GPU

Everything except the GPU half runs on a laptop: the tools, the search, the simulator, the
scenario runner and all 147 tests.

```bash
uv sync
uv run pytest -q                                            # 147 tests, no models needed
uv run uvicorn mechanic.server:app --port 8000              # backend, Torque receiver, web UI
```

Then open <http://127.0.0.1:8000/app/> and use the Garage panel. Voice needs the GPU half.

Build the knowledge base (downloads about 100 MB, scrapes at 1 request per second):

```bash
uv run python -m mechanic.data.stackexchange
uv run python -m mechanic.data.carcarekiosk
uv run python -m mechanic.data.startmycar
uv run python -m mechanic.knowledge.dtc
uv run python -m mechanic.knowledge.search build
```

Ask the search something:

```bash
uv run python -m mechanic.knowledge.search query "high fuel trim at idle" --vehicle audi_a4_b8
```

### On a rented GPU

Tested on a **RTX A4000, 16 GB**, image `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`, 35 GB disk.
Ubuntu 24.04 is not optional: the prebuilt llama.cpp binaries need glibc 2.38.

```bash
# once per machine: llama.cpp, models, ASR weights, run scripts
ssh -p <port> root@<host> 'bash -s' < scripts/gpu_up.sh

# the code and the prebuilt index
git archive --format=tar HEAD | gzip -c | ssh … 'cd /workspace/mechanic && tar xzf -'
tar czf - data/processed data/index | ssh … 'cd /workspace/mechanic && tar xzf -'

# everything else: model, API, a car on the ramp, public URL, warm-up
MECHANIC_ACCESS_KEY=… MECHANIC_NGROK_DOMAIN=… ssh … 'bash -s' < scripts/demo_up.sh
```

`scripts/demo_up.sh` takes about ten minutes, nearly all of it loading models, and warms them on
a throwaway conversation so the first person to talk does not pay for it.

### Reproducing the measurements

```bash
# 76 scenarios against any OpenAI-compatible server
uv run python -m mechanic.evals.run --model gpt-oss-20b --base-url http://127.0.0.1:8001/v1

# judge those answers against the fault the simulator was running
uv run python -m mechanic.evals.judge evals/results/<run>.json --model gpt-oss-20b

# the whole spoken loop, from recorded speech, with the stage breakdown
uv run python scripts/bench_voice.py --url ws://127.0.0.1:8000/ws/voice --audio evals/audio/real

# word error rate, clean and under noise
uv run python scripts/make_noisy_audio.py --snr 20 --snr 10
uv run python scripts/bench_wer.py --audio evals/audio/real

# search, against Stack Exchange duplicate pairs
uv run python -m mechanic.evals.retrieval
```

### Environment variables

| Variable | Effect |
|---|---|
| `MECHANIC_ACCESS_KEY` | gates everything except `/api/health` and `/torque` behind a key in the link |
| `MECHANIC_LLM_BASE_URL`, `MECHANIC_LLM_MODEL` | where the language model lives |
| `MECHANIC_VOICE`, `MECHANIC_VOICE_TEMP` | the synthesiser's voice sample and sampling temperature |
| `MECHANIC_PROMPT`, `MECHANIC_REASONING` | prompt variant and reasoning effort, for A/B runs |
| `MECHANIC_ASR_THREADS` | override the recogniser's cgroup-derived thread count |
| `MECHANIC_LATENCY_LOG` | where per-turn client measurements are appended |

---

## Repository layout

```
src/mechanic/
  voice/          WebSocket transport, VAD, ASR, TTS, turn orchestration, vocabulary repair
  agent/          8 tools, system prompt, streaming loop, safety rules
  knowledge/      trouble-code database, hybrid search index
  torque/         Torque protocol receiver, SQLite store, vehicle simulator
  evals/          scenario runner, LLM judge, retrieval evaluation
  data/           knowledge-base builders: dump converter and two polite scrapers
  server.py       FastAPI: web UI, Torque receiver, Garage API, latency summary
web/              the site: conversation, tool cards, latency panel, Garage
evals/            76 scenarios, 15 real voice recordings, hand-read verdicts, results
scripts/          benchmarks, GPU bring-up, demo bring-up, publishing
docs/             decisions with their reasons, progress log, engineering notes
tests/            147 tests
```

Four documents carry the reasoning, and they are worth more than this README if you want to know
why something is the way it is:

- **[docs/DECISIONS.md](docs/DECISIONS.md)** — every choice, why it was made, what was rejected.
- **[docs/NOTES.md](docs/NOTES.md)** — the engineering log: measurements, dead ends, traps.
- **[docs/PROGRESS.md](docs/PROGRESS.md)** — current state and what remains.
- **[docs/PROJECT.md](docs/PROJECT.md)** — the brief and the constraints.

---

## Data sources and licences

| Source | Content | Licence and access |
|---|---|---|
| Motor Vehicle Maintenance & Repair Stack Exchange | 20,996 Q&A threads from the official data dump | CC BY-SA; the site's robots.txt disallows crawling, so the dump is used and every answer links back |
| carcarekiosk.com | 238 maintenance guides for the five generations | scraped at 1 request/second, robots.txt respected, linked back |
| startmycar.com | 5,430 English owner problem reports | scraped at 1 request/second, robots.txt respected, linked back |
| [OBDex](https://github.com/foerbsnavi/OBDex) | 9,533 trouble codes with causes and symptoms | CC0 data, MIT code |

Stack Exchange content is CC BY-SA and every passage carries a link to its thread.

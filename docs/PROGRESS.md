# Progress

## Now

**2026-09-22. The work is being presented to the reviewer today.**

### Bringing the demo up

Instance **51763823 is stopped, not destroyed** — everything is installed on it. The ngrok domain
is reserved, so the address does not change between sessions:

```
https://cruncher-wieldable-pork.ngrok-free.dev/app/?k=<MECHANIC_ACCESS_KEY>
```

```bash
vastai start instance 51763823            # about 3 minutes
vastai ssh-url 51763823                   # the port changes on every start

MECHANIC_ACCESS_KEY=… MECHANIC_NGROK_DOMAIN=cruncher-wieldable-pork.ngrok-free.dev \
  ssh -p <port> root@<host> 'bash -s' < scripts/demo_up.sh
```

`demo_up.sh` starts the model, the API behind the key, a car on the ramp, the public address, and
**warms the models on a throwaway conversation**. About ten minutes, nearly all of it loading
models. Do it **before** anyone opens the tab: the first sentence of a cold process takes minutes
and is the first thing they would hear.

Two things that have gone wrong in front of people:

- The free ngrok tier shows its own page first. One click on **Visit Site**.
- **Start a car in the Garage first.** A conversation with no car running is "check your adapter"
  on every question, which reads as the agent being stupid. There is now a warning above the
  conversation, but it is still the first thing to check.

### Measured on the shipping build

| | |
|---|---|
| Safety | **11/11 by substance** (9/11 by the literal check — see NOTES) |
| Tools | **91.5%** |
| Scenarios passed whole | 60/76 = 78.9% |
| Truthfulness (LLM judge) | **80.3%** free of falsehood; the judge agrees with a hand-read set 74% exactly |
| Word error rate | **2.2%** quiet, **4.4%** at 10 dB SNR |
| **What a person waits** | **1,537 ms** p50 on live browser turns, 847 ms best — recognition 113, model 582, **synthesis 692**, network 67 |
| Where the 600 ms went | **synthesis of a first sentence that is not in the cache**, plus two medians over different sets of turns. Not the network: 67 ms at the release, 0 bytes queued |
| Stage breakdown | measured, not derived: recognition / model / tools-before-sound / speech / turn hold / network, and they sum to the wait exactly |
| Answer length | 6.9 s on real voice |
| Tests | **147** green |

By category: `owner_reports` and `safety` 100%, `how_to` 88.9, `conversation` 85.7,
`multi_turn` 83.3, `live_data` 71.4, `dtc` 60, `vehicle` 60, **`forum` 57.1**.

### Not done, and said out loud

- Truthfulness is about 80%, and the instrument measuring it is noisier than the differences
  between recent builds.
- Two how-to answers refuse to give steps at all — grounding trades usefulness for honesty.
- The `forum` category answers 57% of the time.
- **A thirty-minute session has never been run.** The last untested gate, and the one most likely
  to spoil a live demo.
- Five cars, one voice, synthetic noise.
- Long conversations will overflow the 8k context; there is no summarisation.

### What to do next, in order

1. **Run the thirty-minute session.** One GPU hour, about $0.12. After it the claim "it will not
   fall over" is a measurement rather than a hope.
2. **Fix the judge's rubric** — stop penalising a clarifying question and a correct refusal, and
   give it the real state of the car — then re-measure. Roughly forty minutes, and afterwards
   "N% false" is a number instead of an opinion.
3. The `forum` category. The next untried idea is merging `search_forum` and
   `search_owner_reports` into one tool with a parameter, since the model does not reliably tell
   them apart even with rewritten descriptions.
4. **Synthesis of the first sentence, 692 ms and the largest stage there is.** Found 2026-09-22
   on live turns; it vanishes to 1 ms whenever the opening line is one primed at boot. The
   cheapest idea is to widen what gets primed; the honest one is that Kyutai's first frame costs
   350 ms standalone and twice that while llama-server is generating on the same card. Either
   way this, not the network, is where a second of the wait lives.
5. VRAM headroom is 577 MiB. Reduce llama-server's batch or context and re-measure.

## Ship gates, set 2026-09-21

The four things that break a product for a real person: it can harm them, it can lie to them, it
can mishear them, and it can be unbearable to wait for. Everything else is diagnostics for us.

| Gate | Baseline | Now |
|---|---|---|
| Safety | 11/11, zero false answers in that category | **met by substance** |
| Outright false answers | ≤5% | ~20% by the judge, ~11% by hand |
| Strictly correct answers | ≥85% | 76.3% when last hand-read |
| Tools | ≥90%, and zero "answered without looking" | **91.5%** |
| Word error rate | ≤5% quiet, ≤10% in noise | **2.2% / 4.4%** |
| Latency in the browser | p50 ≤1.0 s, p95 ≤1.5 s | p50 ~1.5 s, accepted by the user |
| Answer length | median ≤8 s | **6.9 s** |
| Thirty-minute session | zero failures | **not run** |

Deliberately **not** gated: search quality. Across eleven scenario failures, retrieval caused
zero of them. A threshold on a non-binding constraint is a day spent making a number prettier
while the product stays the same.

## Where a driver would still be let down

Written after the first run on live speech, and kept because a demo that hides these is worth
less than one that names them.

**Measured and short of the mark**

- `forum` at 57%: one weak category out of nine.
- "Content accuracy" by keyword is not diagnostic correctness. Reading seventy passing answers by
  hand found eight outright false and four half-invented. The verdicts are in
  `evals/content_gold.yaml`.
- Two how-to answers now decline to give steps.
- The vehicle boost in search is still unmeasured: only 6 of 294 target threads in the duplicate
  set are tagged with one of our five cars.

**Never tested**

- A thirty-minute session, and long conversations against an 8k context.
- Two drivers at once. One card, one conversation, contention unmeasured.
- Voices other than the one the word error rate was measured on.
- Cars outside the five.
- The adapter dropping mid-sentence. There is "no data"; there is no "the data stopped halfway".
- Hands-free end to end. Under a bonnet both hands are busy, so it is the only honest mode, and
  it costs roughly 0.4 s more than push-to-talk — the silence the detector has to wait out, now
  measured per turn (366-378 ms of audio time on three clips) and carried into `detection_ms`.
  The browser never measures hands-free at all: `client_ms` is stamped on button release, and
  there is no button, so the panel falls back to the server's own figure.

**Closed since that list was written**: the public address, the access key, the disclaimer, the
cold-start warm-up, noise measurement, and the safety scenarios.

## Deliberately considered and deferred

**Putting a sensor summary in the context, the way the car is.** It would remove a whole tool
round from questions like "why is the temperature climbing". Deferred on purpose: the brief asks
for an agent with tools and `read_live_data` is the most-used one, and the scenarios explicitly
require it — changing the implementation and the measurement together to make the number go up
is fitting the test. Worth revisiting when there is an independent measure of answer quality.

**Calibrating the confirmation threshold on varied speech.** It is 0.9 s (0.35 from the detector
plus 0.55) set from a single measured clip. Different speakers pause differently; somewhere this
cuts a question short and somewhere it waits for nothing.

## Browser state

No state may depend on the browser pane being alive. Dev servers start by name from
`.claude/launch.json` (`preview_start backend`, FastAPI on :8000).

| When | URL | What was done |
|---|---|---|
| 2026-09-17 | /docs | the backend's Swagger: `POST /api/sim/{device}` starts a simulator, `GET /api/sensors/{device}` returns readings and codes |
| 2026-09-20 | /app/ | Garage: broke an A4 with a vacuum leak at ×5, watched the short-term trim rise and P0171 appear |
| 2026-09-20 | :8010/app/ over an SSH tunnel | the whole pipeline: injected a misfire, the agent named the coil or plug from the sensors alone |
| 2026-09-21 | the public ngrok address | real conversations, latency recorded per turn |

## Vast.ai instances

| When | Instance | GPU | $/h | State |
|---|---|---|---|---|
| 2026-09-18 | 51427537 | RTX A4000 16 GB | 0.129 | destroyed — Ubuntu 22.04 image, llama.cpp would not start |
| 2026-09-18 | 51432831 | RTX A4000 16 GB | 0.151 | destroyed after day 3 |
| 2026-09-18 | 51469409 | RTX A4000 16 GB | 0.098 | destroyed — measuring latency changes |
| 2026-09-19 | 51478762 | RTX A4000 16 GB | 0.096 | destroyed — first working end-to-end run |
| 2026-09-20 | **51763823** | RTX A4000 16 GB | 0.120 | **stopped, everything installed** |

> Before finishing a session, check the instance is stopped.

### Money

- **$3.86 of $5 spent**, $1.14 left. No card is attached, so overspending is impossible.
- **95% of the bill was traffic, not GPU.** One session: downloads $0.844, GPU hours $0.037,
  disk $0.006. Of 21.6 GB downloaded, 7 GB was Python and CUDA wheels.
- A stopped instance costs $0.019/h for its disk. Reinstalling from scratch costs $0.85 on a host
  charging $39/TB, so stopping pays for itself after 45 idle hours.
- The dead-man switch is armed automatically by `gpu_up.sh` using the container's own key.

## Key parameters

- Language model: gpt-oss-20b MXFP4, 8k context, `reasoning_effort: low` **set on the server**.
- Voice: Kyutai 1.6B, `n_q=32`, sample `expresso/ex03-ex01_calm_001_channel1_1143s.wav` since
  2026-09-22 — the same actor reading calmly instead of the day-3 "happy" take. Costs 7.9% of
  speech time, about half a second on a typical answer; time to the first sound is unchanged.
  Do not lower `MECHANIC_VOICE_TEMP` to flatten it: that stretches the delivery instead. See NOTES.
- Recognition: parakeet-tdt-0.6b-v2 int8 **offline**, threads `min(8, cgroup budget)` —
  `os.cpu_count()` lies inside a container.
- Answer length capped by `MAX_SPOKEN_CHARS = 140` in `agent/loop.py`; synthesis speaks about
  13.5 characters a second, and the cut lands between sentences.
- Fixed phrases are synthesised at boot by `Synthesiser.prime`.
- Everything resident peaks at about 15.8 of 16.4 GB.
- `.env` is read by `src/mechanic/config.py`; the environment wins over the file.
- Repository: `git@github.com:DanilAra7/voice-mechanic.git`. Code reaches the instance through
  `git archive`, so no deploy key is needed.

## The plan, by day

- [x] **Day 1** — repository skeleton, Torque simulator, choice of cars and forum, all three data builders
- [x] **Day 2** — trouble-code database, tools, agent loop, scenarios and runner, search index
- [x] **Day 3 (GPU)** — benchmarked four language models and five voices; chose gpt-oss-20b 8k + Kyutai
- [x] **Day 4** — the WebSocket pipeline: VAD → ASR → model → voice, streaming and barge-in
- [x] **Day 5** — the site: conversation, tool cards, latency panel, Garage, benchmark mode
- [x] **Day 6 (GPU)** — latency work, prompt work, the 76-scenario suite, real voice recordings
- [x] **Day 7** — hand-read accuracy, the LLM judge, grounding, safety rules, noise, the public address, README

## Log

- **2026-09-17** — subject, stack and plan chosen. Day 1: cars and forum picked by measuring how much data exists for each; the Torque simulator works end to end; the Stack Exchange dump is processed. Day 2: trouble-code database, search, tools, agent loop, evaluation harness.
- **2026-09-18** — day 3 on a rented box. llama.cpp b11037 sees the A4000. The first offer was taken while we were creating it and silently produced a stopped instance; recreated with `--cancel-unavail`. An A4000 was chosen over a 4060 Ti for memory bandwidth (448 against 288 GB/s). Noted that `cpu_cores` in an offer is the host's, not ours.
- **2026-09-19** — first working end-to-end spoken run.
- **2026-09-20** — the Garage; fixed phrases primed at boot; answer length capped; ASR threads from the cgroup budget; 15 real voice recordings made and the whole pipeline measured on them; two real safety holes found and closed.
- **2026-09-21** — all seventy passing answers read by hand against the injected fault; the LLM judge and the pooled retrieval judge written; grounding moved to where the passages arrive; trend direction made band-relative; safety rules made deterministic; noise measurement; the public address, the access key and the disclaimer; the vocabulary repair; a dropped-connection bug that looked like a flaky tunnel and was an `AttributeError`.
- **2026-09-22** — README and documentation written in English for a reader who knows nothing about the project. Shipped the calm voice sample, paying 7.9% of speech time for delivery that suits the job. Replaced the latency panel's cumulative milestones with five stages timed where they happen, which add up to the wait exactly and finally give the confirmation hold and the tool time a row each.

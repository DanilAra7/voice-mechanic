# Engineering notes

Measurements, dead ends and traps. Everything here was paid for once; the point of writing it
down is not to pay twice. Figures are measured unless marked `(?)`.

## Environment

- Laptop: MacBook Air M4, 16 GB, macOS 26.6. Passively cooled, so long local workloads throttle.
- Available: `uv`, `node v24`, `git`, system `python3 3.9.6` (unused — the project is on 3.12
  through uv). Not available: `brew`, `docker`, `ollama`. 7z archives unpack with
  `uvx --from py7zr py7zr x <file> <dir>`.
- zsh: bare `echo ===` breaks on equals-expansion; use quotes.

## Commands

```bash
uv run pytest -q
uv run ruff check --fix src tests && uv run ruff format src tests
uv run uvicorn mechanic.server:app --port 8000 --reload
uv run python -m mechanic.torque.simulator --vehicle audi_a4_b8 --mode idle --fault vacuum_leak --time-scale 20
uv run python -m mechanic.data.stackexchange       # ~3 s
uv run python -m mechanic.data.carcarekiosk        # ~5 min cold, seconds from cache
uv run python -m mechanic.data.startmycar          # slow: hundreds of requests at 1/s
uv run python -m mechanic.knowledge.dtc            # rebuild the code cache
uv run python -m mechanic.knowledge.search build   # the index (~56k passages)
uv run python -m mechanic.knowledge.search query "high fuel trim at idle" --vehicle audi_a4_b8
uv run python -m mechanic.evals.run --model <name> --base-url http://127.0.0.1:8001/v1

# start a simulator through the API:
curl -X POST localhost:8000/api/sim/demo -H 'content-type: application/json' \
  -d '{"vehicle":"audi_a4_b8","mode":"idle","fault":"vacuum_leak","time_scale":20}'
curl localhost:8000/api/sensors/demo
```

## Data sources (measured 2026-09-17)

| Source | What | Volume | Access |
|---|---|---|---|
| mechanics.stackexchange.com | the `stackexchange_20251231` dump from archive.org, 73 MB | 28,380 questions, 40,675 answers → 20,996 threads with an accepted or upvoted answer, ~39 MB of text | CC BY-SA. The site's robots.txt disallows everything and sets `ai-train=no`, so the dump is the only legitimate path |
| carcarekiosk.com | `/videos/<Make>/<Model>/<genYear>` → about 40 pages per generation (81 for the F-150); the text is the video description, which is the steps | ~240 pages across five cars | robots.txt allows it; there is a sitemap |
| startmycar.com | `/us/<make>/<model>/problems[/pageN]`, 30 cards a page; solved reports carry a `solucionado` class; answers live on the detail page | accord 2282, f-150 2025, corolla 1000, civic 708, a4 338 across all generations, much of it Spanish | robots.txt allows it; no sitemap |

## Traps and non-obvious things

- A browser hands over a microphone only over HTTPS, or on localhost.
- Tunnels do not proxy UDP, so WebRTC is out and the transport is a WebSocket.
- Torque web upload, confirmed: GET, keys `k<hex pid>` with no leading zeros (`k5`, `kc`,
  `kff1005`), metadata in `userFullName<pid>` and friends, answered with `OK!`. The unit arrives
  as `\xC2\xB0C`, replaced with `°`.
- In the simulator `dt = interval_s * time_scale`; with a small interval a fault needs a large
  time scale or it never develops.
- carcarekiosk's generation page, e.g. `Audi/A4_Quattro/2009`, covers the whole generation.
- Fake-LLM tests: the `AsyncOpenAI` client is replaced by an object with a scripted chunk list;
  the loop mutates `messages`, so the fake must copy it.
- `pytest` does not see `tests` as a package — shared helpers live in `tests/conftest.py` and are
  imported as `from conftest import …`.
- **bm25s**: a query needs `bm25s.tokenize(q, return_ids=False)`, a list of strings. Passing ids
  from a fresh tokeniser points them at a *different* vocabulary and the results are garbage
  (hit 2026-09-17). Indexing is the opposite and uses ids. `tests/test_search.py` guards this.
- Secrets: `.env.example` is in git as documentation, `.env` is ignored. `load_env()` uses
  `os.environ.setdefault`, so on a rented machine the container's environment beats the file.
- The M4 has no fan: fastembed takes about 490% CPU and the laptop throttles. Locally, pass
  `--threads 4`; better, compute embeddings on the rented GPU with `--dense-only`.
- `invalid value encountered in matmul` warnings on macOS are false — Accelerate setting FPU
  flags. Checked against float64: the difference is 1e-7. Do not fix.

## Renting a GPU: traps (2026-09-18)

- **`cpu_cores` in a Vast offer is the HOST's, not ours.** Our share is `cpu_cores_effective`,
  and memory likewise: `cpu_ram` is the host's, ours is roughly `cpu_ram * gpu_frac`. The first
  instance was chosen on `cpu_cores>=8` and had seven effective cores.
- **fastembed/onnxruntime starts threads by the number of VISIBLE cores.** In a container `nproc`
  shows the host's 56 while the cgroup gives 6.7 (`/sys/fs/cgroup/cpu.max` → `672000 100000`).
  The result was 98 threads on 6.7 cores and eight minutes without a single passage. Always pass
  `--threads N` from the quota, plus `OMP_NUM_THREADS`.
- **Prebuilt llama.cpp binaries (b11037) need glibc 2.38+**, so Ubuntu 24.04. The
  `pytorch/pytorch` image is 22.04 with glibc 2.35 and the binaries refuse to start
  (`GLIBC_2.38 not found`); it has no nvcc or gcc either, so building from source is not an
  escape. The working image is `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`.
- llama.cpp needs TWO release archives: `llama-<build>-bin-ubuntu-cuda-12.8-x64.tar.gz` and the
  matching `cudart-…`, plus `apt install libgomp1`. Everything unpacks flat into one directory
  and `LD_LIBRARY_PATH` points at it.
- **`vastai create instance` without `--cancel-unavail`, on a machine that has just been taken,
  silently creates a stopped instance** and returns `success: False`. A later `start` answers
  "Required resources are currently unavailable, state change queued".
- `vastai destroy instance` asks for confirmation; in a script, `yes y | vastai destroy …`.
- **`pkill -f <pattern>` over ssh kills your own session** when the pattern matches the command
  line running it. This cost four attempts in one afternoon. Kill by PID, or match the executable
  name exactly with `pgrep -x`.
- An instance has its own `CONTAINER_API_KEY` in `/proc/1/environ`, next to `CONTAINER_ID`, and
  can stop itself with it. The account key never needs to go on a rented machine.
- Vast's variables are visible in PID 1's environment but not in an ssh session — read them with
  `tr '\0' '\n' < /proc/1/environ`.
- **`setsid --fork` is the reliable way to detach over ssh.** `setsid nohup … &` intermittently
  does not survive the session closing.

## Dependencies: two traps (2026-09-18)

- **`sherpa-onnx` is useless without `sherpa-onnx-core`.** The native libraries are a separate
  package that `uv sync` will not pull on its own, and the import fails on
  `Library not loaded: @rpath/libonnxruntime.dylib`. Both lines are in `pyproject.toml`.
- **`moshi` pins `numpy<2.3` while `fastembed` on Python 3.14 wants `numpy>=2.3`.** Solved by
  `requires-python = ">=3.12,<3.14"`; on 3.12 both live together on numpy 2.2.6. `moshi` itself
  is in the `gpu` extra so a laptop does not pull CUDA wheels.

## Models: verified names and sizes (2026-09-18)

| Repository | File | Size |
|---|---|---|
| `ggml-org/gpt-oss-20b-GGUF` | `gpt-oss-20b-MXFP4.gguf` | 11.3 GB |
| `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | `…-UD-Q3_K_XL.gguf` | 12.9 GB |
| `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | `…-UD-Q4_K_XL.gguf` | 16.5 GB |

## Benchmarks

### Language models: VRAM (2026-09-18, RTX A4000, 16,376 MiB)

Peak sampled by `nvidia-smi` twice a second during load and generation. Baseline 16 MiB.

| Candidate | Context | Peak VRAM | Left for a voice | Load |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 | 16k | **11,688 MiB** | ~4.6 GB | 16.0 s |
| Qwen3-30B-A3B UD-Q3_K_XL | 16k | **14,886 MiB** | ~1.4 GB | 18.3 s |
| Qwen3-30B-A3B UD-Q3_K_XL | 8k | **14,110 MiB** | ~2.2 GB | 11.5 s |

Cutting the context from 16k to 8k saves only **776 MiB** on the A3B: its KV cache is small
because only 3B of 30B parameters are active, and the weights dominate. So "use a smaller
context" does not rescue anything, and the choice between gpt-oss-20b and Qwen3-30B is really
the choice between "there is room for an expressive voice" and "smarter, with a tiny voice".

### gpt-oss-20b: reasoning effort decides whether the candidate is usable (2026-09-18)

llama-server accepts `chat_template_kwargs: {"reasoning_effort": …}` in the request body. Timed
to the first **content** token, since reasoning goes to `delta.reasoning_content` and is never
spoken. `max_tokens: 250`.

| reasoning_effort | to the first word | to the first token | reasoning tokens | answer |
|---|---|---|---|---|
| **low** | **1,195 ms** | 781 ms | 7 | fine |
| medium | 3,774 ms | 118 ms | 223 | fine |
| high | — | 484 ms | 247 | **none**: the whole token budget went on thinking |
| default | 3,213 ms | 573 ms | 178 | fine |

Without `reasoning_effort: low` this model is unusable for voice; with it, 1.2 s to the first
word. Note the trap in the other direction: at `high` it never reaches an answer, so a benchmark
that only scored the text would have called the candidate broken.

### Language models: the deciding run (2026-09-18, 32 scenarios)

After the "car in the prefix" change. 16k context for all.

| Candidate | Peak VRAM | For a voice | tok/s | Scenarios | Tools | Content | p50 first sentence | p95 |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b MXFP4, reasoning low | 11,700 | 4.6 GB | 116.6 | 65.6% | 84.8% | 75.8% | 881 ms | 1,650 ms |
| Qwen3-30B-A3B Q2_K_XL | 12,960 | 3.3 GB | 120.6 | 71.9% | 84.8% | 75.8% | 630 ms | 1,749 ms |
| Qwen3-30B-A3B Q3_K_XL | 14,908 | 1.4 GB | 106.3 | 71.9% | 78.8% | 81.8% | 698 ms | 1,633 ms |
| Qwen3-14B Q5_K_XL | 12,488 | 3.8 GB | **35.2** | 75.0% | 90.9% | 81.8% | **2,027 ms** | 3,363 ms |

**The dense model loses on physics, not on intelligence.** Qwen3-14B is the best at choosing
tools (90.9%) and runs at 35 tok/s against 105–120 for the others: every token reads all 9.8 GB
of weights, where a 30B-A3B mixture reads about 3B. At 448 GB/s those are exactly the numbers.
Sparsity is the only way to get low latency inside 16 GB.

Effect of putting the car in the system prefix, on tool accuracy:
gpt-oss 72.7 → 84.8 · Qwen Q3 72.7 → 78.8 · Qwen Q2 69.7 → 84.8 · Qwen3-14B 78.8 → 90.9.
The first sentence got slower for all of them: the agent stopped deflecting with a question and
started searching.

### What the benchmark exposed in the agent itself

- **Fuel smell: all four failed.** The prompt demanded "stop driving" first; every model started
  diagnosing instead. This does not separate candidates — it is a guardrail problem, and it
  became one.
- The first search query costs **~800 ms**, a one-off load of the embedding model. Warm it at
  startup or the first driver pays for it.
- Synonym checks were too narrow: "you should not drive the car" was scored as a failure.

### Voices (2026-09-18)

Ten mechanic's lines including codes and numbers. "To first sound" for a streaming engine is the
first audio frame; for a non-streaming one it is unavoidably the whole sentence. The first call
of either pays 6.0–6.6 s of CUDA warm-up.

| Engine | To first sound | Whole sentence | RTF | Peak VRAM | Streams |
|---|---|---|---|---|---|
| **Kyutai TTS 1.6B** | **350 ms** (345–351, very stable) | 1,560 ms | 0.42 | 4,522 MiB | yes |
| Chatterbox 0.1.7 | 1,186 ms (= the whole sentence) | 1,186 ms | 0.41 | 3,556 MiB | no |

Rejected without measurement: **Qwen3-TTS** — no open weights, API only, which breaks the brief.
**VibeVoice-Realtime-0.5B** — claims ~300 ms, but the PyPI package contains no streaming class.
**Orpheus 3B** — leaves no room beside the language model.

### The two together on one card — this is what decided the choice

Separate peaks must not be added: `gpt-oss 16k (11,700) + Kyutai (4,522)` left 154 MiB on paper
and produced an **out-of-memory** with 39 MiB free in practice. Simultaneous spikes eat the
margin.

| | |
|---|---|
| gpt-oss-20b MXFP4, **8k** context, reasoning low | 11,472 MiB |
| + Kyutai TTS 1.6B | +4,077 → 15,549 MiB |
| Peak under load | **15,875 of 16,376 MiB, 501 MiB spare** |
| Model to first sentence | 297–465 ms |
| Voice to first sound | ~355 ms |
| **To first sound, together** | **655 ms median** |

The 8k context is not cosmetic here: dense gpt-oss has a much larger KV cache than the sparse
Qwen, and cutting it saved about 900 MiB — exactly what was missing. The same trick on Qwen saved
776 MiB and rescued nothing.

**655 ms is a floor**: a shortened prompt, no tool calls, no recognition.

### Why Qwen3-30B-A3B Q2 lost despite scoring better

Q2 takes 12,960 MiB and leaves 3,416. Kyutai (4,522) does not fit at all; Chatterbox (3,556)
misses by 140 MiB. Even at 8k context about 4,192 remain — enough for Chatterbox, which means
1,186 ms to first sound instead of 350. On 16 GB **the voice chooses the model**, not the other
way round, and measuring them separately was a methodological mistake: the interim recommendation
survived two hours before the voice benchmark overturned it.

### Recognition on the CPU (2026-09-18, M4, 4 threads, sherpa-onnx)

Two different things were measured: real-time factor — whether it keeps up while the driver is
still speaking — and the **tail**, how long it needs *after* they stop. Only the tail is in the
latency budget. Audio was ten of our own synthesised samples (38.7 s), so this is optimistic; a
real microphone in a garage is worse.

| Model | Tail p50 | RTF | Transcript |
|---|---|---|---|
| parakeet-unified-0.6b int8 **streaming**, 240 ms | 434 ms | **1.69** | "124 degrees" |
| parakeet_tdt_ctc **110m** int8 offline | 34 ms | 0.009 | invented a leading "and" |
| **parakeet-tdt-0.6b-v2 int8 offline** | **104 ms** | 0.027 | clean |

**Streaming recognition is a trap here.** An RTF of 1.69 means the model cannot keep up and falls
further behind as the sentence goes on: by the end of a four-second question it owes about 2.6 s
beyond the measured tail. One offline pass over the same sentence takes 104 ms. Running a heavy
model once is cheaper than running it in pieces.

For the budget: **recognition is not the problem** — 104 ms against roughly 1,500 ms for a single
pass of the language model. What needs optimising is the number of model round trips.

### Day 4: what the latency and safety changes did (2026-09-18)

| | day 3 (16k) | + "announce before calling" | **reverted + filler counted** | + "how-to only from the tool" |
|---|---|---|---|---|
| Scenarios | 65.6% | 68.8% | **81.2%** | 81.2% |
| Tools | 84.8% | 75.8% | **87.9%** | 87.9% |
| Content | 75.8% | 90.9% | 90.9% | 87.9% |
| Safety | **0%** | 100% | **100%** | 100% |
| p50 first sentence | 881 ms | 848 ms | **447 ms** | 436 ms |
| p95 | 1,650 ms | 2,783 ms | **857 ms** | 819 ms |
| how_to | 25% | 25% | 25% | **75%** |

Three separate conclusions, easy to confuse with one another:

1. **"Say a line before calling a tool" is a harmful instruction.** The model announced the call
   and stopped there — the spoken sentence replaced the action. Nine points of tool accuracy for
   thirty milliseconds. Reverted.
2. **Half the latency win was a measurement error.** `first_sentence_ms` was set for the model's
   own sentences and for the guardrail, but not for the filler line — which the driver hears. We
   had been overstating latency in exactly the cases the filler exists for. Fixing it dropped p50
   from 881 to 447 ms with no change in behaviour.
3. **The safety guardrail works: 0% → 100%.** A deterministic check ahead of the model, rather
   than hope invested in a prompt.

**We hit the resolution of the scenario set.** With 32 scenarios, two to four per category, one
scenario is 3 points overall and 25–50 points inside its category. The how-to change moved its
own category from 25% to 75% and the overall number did not shift. Tuning a prompt against
numbers that noisy is self-deception; the set had to grow first.

### End-of-utterance detection is the most expensive stretch (2026-09-18, Silero VAD v5)

Measured honestly: the silence after speech is fed **in real time**, because
`min_silence_duration` is counted in audio time. Feeding it faster gives a beautiful 2 ms, which
measures the CPU rather than the wait.

| `min_silence_s` | End of speech detected | Text ready (VAD + ASR) |
|---|---|---|
| 0.15 | 302 ms | 412 ms |
| 0.20 | 405 ms | 509 ms |
| **0.35 (default)** | 507 ms | **616 ms** |
| 0.50 | 704 ms | 820 ms |

The detector costs roughly `min_silence_s + 150–200 ms`: Silero works in windows and smooths,
and our own chunking adds 100 ms.

That made the budget **616 ms of waiting + 447 ms of model + 355 ms of voice ≈ 1,418 ms** — the
voice-activity detector costing more than the language model. The earlier estimate of 900 ms
simply had no end-of-utterance wait in it at all.

This is what push-to-talk removes: releasing a button is an exact end of turn.

### The silence threshold cuts a person off mid-question (2026-09-18)

The clip "I am getting a code P zero one seven one, what does that mean?" (5.12 s):

| `min_silence_s` | Split |
|---|---|
| 0.35 (ours) | **two utterances**: "I am getting a code P0171" \| "What does that mean?" |
| 0.6 | still two |
| 0.9 | one |

Ordinary speech contains a **0.6–0.9 s** pause inside a question. The agent answered the first
half, and the second half arrived as an interruption — and the interruption logic fired
*correctly*.

**What must not be done:** deciding the turn is over by asking whether the text is a complete
sentence. "I am getting a code P0171" is a complete sentence and an incomplete question, and a
model will confidently say it is finished. What separates them is intonation, which is not in the
text.

**What is done instead: work early, speak on confirmation.** Recognition, the model and the voice
all start on the short threshold, but the audio is held until the long one confirms. If the
driver carries on, the turn is cancelled quietly and the unspoken fragment is glued onto the next
utterance. Latency becomes the maximum of "confirm" and "work", not their sum.

Three rules had to be separated explicitly, each fixing what the previous one broke:

1. An interruption requires **sustained** speech (≥0.25 s), or echo triggers it.
2. It does not count inside a 0.8 s window after the answer starts, **and duration inside that
   window does not accumulate** — the first version of the window merely postponed the same bug.
3. **Only a confirmed turn can be interrupted.** Before confirmation no sound has been played, so
   there is nothing to interrupt: speech at that moment is the rest of the question.

### The pipeline running end to end (2026-09-18)

Recorded questions fed into the WebSocket **in real time**, timed from the last sample of speech.

| | ms |
|---|---|
| Recognition | 128–252 |
| First sentence from the model | 510–1,130 |
| **First sound at the client, p50** | **1,377** |
| p95 | 1,878 |

The first turn after a start costs about 1,880 ms — a one-off compilation of torch kernels in the
synthesiser. Warming at startup is mandatory.

**Four bugs that no component benchmark could have shown:**

1. **Out-of-memory on the second turn.** The "501 MiB spare" from day 3 was not spare: model
   11,469 + voice 4,485, plus fragmentation. Fixed on the llama.cpp side with
   `--cache-type-k q8_0 --cache-type-v q8_0 --ubatch-size 128` (−200 MiB) and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. **The last audio frames were lost.** The synthesiser hands frames over through
   `call_soon_threadsafe`; the loop exited on `work.done() and queue.empty()` while callbacks were
   still pending, and a sentence arrived with no sound.
3. **The agent interrupted itself.** Interruption fired on "the detector hears speech", which
   stays true for a while after the question ends — the tail in the detector's buffer. Four turns
   out of six died while reporting themselves as successful.
4. **A turn could end in silence** when the model produced only reasoning. For a voice agent that
   is indistinguishable from a dropped connection; it now apologises out loud.

The instability in that run was in the *test client*: it sent and received in one coroutine and
called `json.loads` on any message during the handshake. After separating them, **p95 fell from
1,878 to 1,442 ms** — the measuring rig was adding almost half a second of spread. The lesson is
to fix the rig before the thing being measured.

## The Garage panel (2026-09-20)

A panel in `web/`: car, driving mode, fault and how fast it develops (×1/×5/×20), live readings
every 1.5 s, and trouble codes. It talks to the API that already existed; the backend needed only
one addition, a `{"type":"vehicle"}` command on the socket so the car can change without
reconnecting.

Verified in the browser (Audi A4 B8, vacuum leak, ×5):

- P0171 appears after about nine real seconds (the simulator's threshold is `fault_time_s > 45`).
- **Short-term fuel trim: 6.5% in town against 16.4% at idle** — the signature of a vacuum leak,
  visible in the panel. A reading turns red exactly where the agent's own tool calls it abnormal,
  so the panel never knows more than the agent.

Traps:

- `[hidden]` is overridden by any class that sets `display`, because the user-agent style has zero
  specificity. `[hidden]{display:none !important}` was needed or "hidden" blocks are visible.
- A global `mouseup` for the hold-to-talk button caught clicks anywhere on the page, so any Garage
  button put the status into "thinking…".
- Sensor names come from phone uploads, so they are written into the DOM as text, never as markup.
- The car keeps running on the server after a page reload; the front end adopts it through
  `GET /api/sim/{device}` instead of showing an empty garage.

## Day 6: what the rental actually costs, and where the latency really was (2026-09-20)

### 95% of the bill was not the GPU

One session on instance 51763823 (RTX A4000, Estonia, $0.106/h) cost **$0.934**:

| Line | $ |
|---|---|
| **Download traffic** | **0.844** |
| GPU hours | 0.037 |
| Disk | 0.006 |
| Upload traffic (our code and index) | 0.004 |

The cause: this machine charges **$0.0390625 per GB, i.e. $39/TB**, where the previous ones
charged $2.67/TB. `inet_down_cost` is right there in `vastai search offers` and was not looked
at. The hourly difference between a cheap and an expensive offer is cents; the traffic difference
is dollars.

```bash
vastai search offers 'gpu_ram>=15.5 gpu_ram<=17 cpu_cores_effective>=6 inet_down_cost<=0.005 …'
```

21.56 GB was downloaded — reconciled against `/sys/class/net/eth0/statistics/rx_bytes`, which
matched the bill to the cent — not the 13 GB previously assumed:

| What | GB |
|---|---|
| gpt-oss-20b MXFP4 and the ASR model | 12.5 |
| **`uv sync --extra gpu`: torch and CUDA wheels** | **7.1** |
| llama.cpp and cudart | 1.9 |
| apt and the rest | ~0.5 |

Seven gigabytes of Python dependencies were not in any estimate.

**Consequence: a paused instance is now stopped, not destroyed.** Stopped it pays for disk
($0.4/GB/month → 35 GB ≈ $0.019/h); bringing it back costs $0.85 of traffic. Break-even is about
45 idle hours. This reverses the day-4 rule, which was written assuming $2.67/TB.

### The agent talked for half a minute and nobody was measuring it

The first end-to-end benchmark run hit its timeout: asked about temperature, the agent answered
with **47 seconds of speech**. There was no measurement of answer length at all — only time to
first sound, which multiplying words does not hurt.

Synthesis speaks **13.5 characters a second** over twelve answers, so text length *is* listening
time.

| | baseline | 320-char cap | **140-char cap** |
|---|---|---|---|
| Median answer | 24.0 s | 16.6 s | **12.4 s** |
| Longest | 36.6 s | 27.7 s | **16.3 s** |

The prompt asked for two or three sentences before and after; the model agrees and says six. Only
a hard cap in code works. It cuts **between sentences**, so the driver always hears a finished
thought.

### Phrases synthesised in advance: the cheapest win in the project

The agent says a handful of lines **word for word**: fillers before a tool call, the safety
warning, the apology for an empty turn. They are also, usually, the **first** thing the driver
hears. Synthesising them at boot and keeping the frames in memory:

| | before | after |
|---|---|---|
| First sound, p50 | 1,496 ms | **1,031 ms** |
| First sound, p95 | 2,134 ms | **1,479 ms** |

On four turns out of six `first_audio_ms` now equals `first_sentence_ms` **plus one millisecond**
— synthesis left the critical path entirely. Nothing about the answer changes: same voice, same
words.

The win (~650 ms) is larger than the standalone voice benchmark (350 ms) because on a live
pipeline the synthesiser shares the card with a model that is still generating.

### Recognition: four threads on fifteen cores

`Recognizer` hard-coded four threads. It now takes `min(8, cgroup budget)`, read from
`/sys/fs/cgroup/cpu.max` — `os.cpu_count()` answers 64 on that machine while our share is 15.36.
Median recognition **345 → 265 ms**.

### The day, end to end

| | morning | evening |
|---|---|---|
| First sound, p50 | 1,772 ms | **1,031 ms** |
| First sound, p95 | 2,217 ms | **1,479 ms** |
| Median answer length | 24.0 s | **12.4 s** |
| Turns completed | 6/6 | 6/6 |

Scenarios on the same 32: **27/32 = 84.4%** (was 81.2), tools **90.9%** (was 87.9), content
**93.9%** (was 87.9), safety 100%, voice-format clean 100%. **Latency fell and quality rose** — a
short answer has fewer opportunities to be wrong.

### Two defects in the measuring rig, found on the way

1. **An abandoned turn counted as an answer.** One clip contains a 0.6–0.9 s pause inside the
   question; the rig returned on the first `turn_end` and printed "answered in 0.1 seconds". A
   turn marked `carried_on` or `barged_in` is no longer counted.
2. **`reset` did not shut the agent up.** It cleared the history while the answer went on playing
   and landed in the next question's measurement.

### The Garage on a live GPU: the whole thing verified (2026-09-20)

Over an SSH tunnel, `/app/` was opened from the rented machine, a misfire was injected at ×20,
and the question was asked by voice. The agent was never told the fault:

> "The codes show a random misfire and a specific misfire on cylinder one. That usually points to
> a spark-related issue — ignition coil, spark plug, or a vacuum leak on that cylinder."

First sound 795 ms. The simulator was running on the same machine, speaking the Torque protocol
exactly as a phone would.

## Closing the measurement gaps (2026-09-20, evening)

### Search quality: numbers at last

No labels were written; existing ones were used. On mechanics.stackexchange, a question closed as
a duplicate of another is a human judgement that these are the same problem. The dump has **390**
such pairs whose target thread is in our index.

| | R@1 | R@5 | R@10 | MRR@10 | ms |
|---|---|---|---|---|---|
| BM25 only | 14.9 | 30.0 | 39.5 | 0.219 | 1.5 |
| Dense only | 20.0 | **43.3** | **52.6** | 0.298 | 5.6 |
| **RRF fusion (what the agent uses)** | **23.3** | 42.3 | 51.8 | **0.309** | 6.2 |
| Fusion without the solved-thread bonus | 23.1 | 41.5 | 52.3 | 0.308 | 6.2 |

- **Fusion wins on first place and MRR, not on recall.** Its R@5 is slightly below pure vectors:
  RRF does not find more, it sorts better.
- **The dense half carries considerably more than BM25** (43.3 against 30.0).
- **This is a floor, not a verdict.** Only the thread a moderator linked counts.
- **The vehicle boost is still unmeasured**: these questions carry no car, and of 294 target
  threads only 6 are tagged with one of our five.

### Blame: the search was never the problem

`TurnResult.blame` looks for the missing phrase inside what the tools returned.

| Whose fault | Failures out of 25 |
|---|---|
| Tool choice | 15 |
| Did not look anything up | 6 |
| Generation (it was found and ignored) | 4 |
| **Search returned rubbish** | **0** |

The component treated as the main risk caused none of the failures. Tuning `RRF_K` and
`VEHICLE_BOOST` would have been wasted work.

### 76 scenarios instead of 32

| | 32 | 76 |
|---|---|---|
| Passed | 84.4% | **67.1%** (51/76) |
| Tools | 90.9% | 81.7% |
| Content | 93.9% | 76.8% |

The drop is not the agent getting worse; it is the set finally seeing the blind spots. In the new
categories the agent systematically answers from memory instead of calling `search_*`.

### The brevity wording was not the culprit (A/B on the same 76 scenarios)

| | terse (current) | original |
|---|---|---|
| Passed | 51/76 (67.1%) | 51/76 (67.1%) |
| Tools | 81.7% | 82.9% |
| Content | 76.8% | 79.3% |
| First sentence p50 | **566 ms** | 729 ms |
| Whole turn p50 | **1,506 ms** | 1,785 ms |

**The hypothesis was wrong.** Identical to the scenario; the differences in tools and content
(1.2 and 2.5 points) are smaller than one scenario is worth (1.3 points). The terse wording is
**163 ms faster** to the first sentence, so it stays.

### Live speech: 15 recordings of a real voice

Recorded through `web/record.html`, which writes 16 kHz WAV and **uploads each clip to the
server** — a browser download silently lost all fifteen takes on the first attempt.

**Word error rate 3.3%** (4.4% counting contractions as mistakes), 11 of 15 clips word-perfect,
median decode 147 ms. On synthesised speech it is 3.5%: **a real voice costs nothing**.

The one substantive error was **"fuel trims" → "field dreams"**, a core term of the domain.

End to end on those recordings: **15/15 turns, first sound p50 800 ms, p95 2,231 ms**. Median
answer length 15.4 s — still long.

### The safety rule that did not survive a microphone

Spoken aloud: "I **smelled** gasoline inside the cabin while driving". Recognised in the past
tense. The rule knew `smell|smells|smelling` and stayed silent; the agent calmly discussed P0171
with somebody sitting in petrol fumes. In `scenarios.yaml` the same sentence is written in the
present tense and had always passed.

Fixed: every verb carries its endings, the phrasings from the recordings were added
(`brakes went soft`, `steering went heavy`), and refuelling is excused — but only when the smell
is not in the cabin. **Rules written against text we type ourselves must be tested on speech.**

### No P0171 bias, but three generation defects

The same question ("the temperature is climbing") against three states of the car:

| In the car | Codes in the data | What the agent said |
|---|---|---|
| vacuum_leak | P0171 | P0171 |
| misfire_cyl1 | P0300, P0301 | "random misfires — codes P0300 and P0301" |
| healthy | none | named no code |

The code comes from the data. But three other things surfaced:

1. **It drags an unrelated code in and invents a connection.** With a misfire, asked about
   temperature: "codes P0300 and P0301… that could be pulling heat from the engine". Invented.
2. **It contradicts its own tool.** 95 °C against a normal range of 85–105 → "this is a sign the
   engine is overheating".
3. **It repeats the driver's claim as a reading.** "The temperature is climbing" on a flat line →
   "ninety-five degrees and still climbing".

The cure was the one that worked for safety: move the judgement into code. `read_live_data` now
returns a `status` per reading ("normal", "TOO LEAN", "OVERHEATING"), a `verdict` over the whole
set, and a `trend` per sensor. The thresholds and their wording live in one place, `Band`.

### The card ran out of memory mid-session

The socket opened, the agent said nothing, and the browser showed `CUBLAS_STATUS_ALLOC_FAILED`.
15,973 of 16,376 MiB were in use. The cause was restarting the API over a live llama-server: the
old process does not return memory instantly and the margin is 577 MiB. **This is a risk for a
demo**; the fix is to take batch or context away from llama-server.

Also: `tar` from a Mac carries AppleDouble `._00.wav` files, which crash soundfile. Set
`COPYFILE_DISABLE=1` before tar and clean up with `find . -name "._*" -delete`.

### What actually fixed quality, step by step (2026-09-20, all runs on the same 76 scenarios)

| | Passed | Tools | Content | 1st sentence p50 | Turn p50 |
|---|---|---|---|---|---|
| Start of session | 51/76 (67.1%) | 81.7% | 76.8% | 566 ms | 1,506 ms |
| + "look it up before asking" | 60/76 (78.9%) | 90.2% | 86.6% | 587 ms | 2,060 ms |
| + safety fixes | 60/76 (78.9%) | 89.0% | 86.6% | 622 ms | 2,147 ms |
| **+ tool descriptions rewritten** | **65/76 (85.5%)** | **92.7%** | **92.7%** | 608 ms | 1,944 ms |

**The two changes that did almost all of it:**

1. **"If a tool could answer it, call the tool first."** The diagnosis came from reading answers,
   not from percentages: the agent was not answering from memory, it was **asking a question back
   instead of working** — "How do I change the wiper blades?" → "Which model year do you have?",
   with the car written in its own system prefix. how_to 44 → 100%, owner_reports 57 → 100%.
2. **Tool descriptions written the way a person asks.** "Search a mechanics Q&A forum for
   diagnosis discussions" became a list of the phrasings a driver uses ("why would…", "what
   causes…", "how do mechanics…") plus an explicit boundary against the neighbouring tool. Tools
   89 → 92.7%, content 86.6 → 92.7%.

**Two hypotheses that proved false**, written down so they are not tested again: the brevity
wording (identical, 51/76 both ways), and `reasoning_effort: medium` (33/76 and twice as slow).

### Final end-to-end measurement on live speech

| | start of session | end |
|---|---|---|
| Turns completed | 15/15 | 15/15 |
| First sound p50 | 800 ms | **812 ms** |
| First sound p95 | 2,231 ms | **1,238 ms** |
| Median answer length | 15.4 s | 14.6 s |

p95 nearly halved at the same p50: the outliers are gone. The guardrail on live speech:
"I smelled gasoline inside the cabin" → first sentence in **233 ms**, "my brake pedal goes almost
to the floor" → **331 ms**, with no model involved at all.

## The voice: calmer means longer (measured 2026-09-20)

The user noticed the voice was too emotional and guessed that toning it down might also save
time. Measured by `scripts/bench_voice_style.py` on the same ten lines. **There is no saving;
there is a bill.**

| Sample | Total speech | Against the current one | s/char |
|---|---|---|---|
| `expresso/ex03-ex01_happy_…` (current) | 40.5 s | — | 0.0648 |
| `expresso/ex03-ex02_narration_…` | 42.2 s | +4.2% | 0.0676 |
| `expresso/ex03-ex01_calm_…` | 43.7 s | +7.9% | 0.0726 |
| `expresso/ex01-ex02_default_…` (different speaker) | 47.4 s | +17.0% | 0.0767 |
| `expresso/ex03-ex01_enunciated_…` | 49.9 s | +23.3% | 0.0771 |

- **First sound does not depend on the sample:** 616–622 ms for all five, RTF 0.50–0.53. That is
  the codec's frame rate, not a property of the voice.
- **Temperature does not flatten the delivery, it stretches it.** `temp 0.4` instead of 0.6:
  happy +16.8%, calm +15.2%, enunciated +39.9%. Less variety means longer pauses and drawn-out
  vowels. Leave it alone; change the sample instead.
- `MECHANIC_VOICE` and `MECHANIC_VOICE_TEMP` make both settable without a rebuild.

**Shipped the calm sample on 2026-09-22.** The bill was known before it was paid: 7.9% more
speech time, roughly half a second on a 6.9-second answer, and nothing at all on time to first
sound. Taken because a mechanic telling you your brakes are gone should not sound pleased about
it, and because the extra half second lands after the driver already has their answer, where a
wait costs least. `narration` at +4.2% is the fallback if it ever matters more than the delivery.

## The latency panel was showing milestones, not stages (2026-09-22)

Four rows called "recognised / first sentence / first audio / network", each a stamp on one
timeline from the same zero. Three of them nest inside each other, so the bars grew to the right
and overlapped, the model and the tools never appeared as numbers of their own, and the only way
to read a stage off the panel was to subtract two rows by eye.

Worse, the obvious fix is wrong. Deriving the model's share as
`first_sentence_ms - asr_ms - tool_ms` **goes negative on any turn with a slow tool**, because
the agent speaks a filler before it looks: the first sentence — and the first sound — happen
before the tool has finished. That is exactly the turn a breakdown is worth having.

So the stages are now timed where they happen, in `TurnTimings`:

| Field | Measured | Note |
|---|---|---|
| `asr_ms` | already was | recognition, wall clock |
| `tts_ms` | new | the synthesiser of the first sentence, to its first frame |
| `hold_ms` | new | the first frame held back until the turn is confirmed (`CONFIRM_EXTRA_S`) |
| `tool_ms_to_audio` | new | of `tool_ms`, the part that ran before the driver heard anything |
| `model_ms` | new | the remainder |

Recognition, synthesis and the hold are disjoint wall-clock intervals inside `first_audio_ms`, so
their remainder is real. The tools report their own durations, which are measured around the call
and can overrun that remainder by a hair, so the tool share is capped at what is left — which
makes the five pieces add up to `first_audio_ms` exactly, asserted in
`test_the_stages_of_a_filler_turn_are_slices_not_milestones`.

Two things this made visible that were hidden before:

- **The turn hold is 550 ms of deliberate silence**, and it used to be charged to the model. Zero
  in push-to-talk (the button declares the turn over); the whole 550 ms hands-free. Part of the
  "hands-free costs about 880 ms" figure, now with its own row rather than folded into the wait.
- **Tool time the driver never waits for.** On a filler turn the search runs while the agent is
  already talking. The panel says so: "a further 612 ms of tool time ran while the agent was
  already talking".

`latency_summary()` and `GET /api/latency` report the same stages as medians, each over the
turns that carried it, with `stages_from_turns` saying how many that was. Medians of disjoint
stages need not sum to the median total, so the totals are measured in their own right rather
than summed.

### The most expensive stage happened before any clock started

Every figure in a turn is stamped against `zero`, set in `handle()`. In hands-free `handle()` is
called when the **detector** yields an utterance, which is already a third of a second after the
driver's last word — so the server was honestly reporting a wait that begins after the most
expensive stage of the turn, and hands-free looked almost as fast as the button.

`Utterance` now carries `trailing_s`: audio consumed between the utterance's last sample and the
newest sample the detector had when it finally committed. Measured on the three Kyutai clips:

| Clip | Speech | Silence waited out |
|---|---|---|
| 00 | 3.68 s | **378 ms** |
| 01 | 3.52 s | **374 ms** |
| 02 | 5.04 s | **366 ms** |

That is `min_silence_s = 0.35` plus the detector's own windowing, and it is audio time, so at a
real-time stream it is wall time too. Day 3 measured 507 ms for the same thing feeding in real
time through 100 ms chunks; production uses 8 ms chunks. **Roughly 0.4 s** is the honest figure.
`flush()` leaves it at zero, which is the whole point of push-to-talk.

**And the confirmation hold is almost never paid.** The design is *work early, speak on
confirmation*: everything starts on the 0.35 s threshold and only the audio waits for the 0.55 s
timer. Since the work takes ~849 ms and the timer 550 ms, the timer has already expired by the
time there is a frame to hold. It bites only when the answer beats 550 ms — the safety path, at
233 ms, which hands-free therefore delivers at about 1.0 s. Worth knowing before claiming the
guardrail is instant in both modes.

### The browser's number was filed against the previous answer

`record_client_latency` read `session.last_timings`, which was assigned at `turn_end`. The
browser sends its figure the instant it **hears** the first frame — mid-turn. So every row in
`client_latency.jsonl` paired this turn's client wait with the previous turn's server stages.
Medians over many turns survive it; a single row explaining one slow turn does not, and that is
what the file is for. `last_timings` is now published where `first_audio_ms` is set.

### Four explanations for the missing 600 ms, killed by measurement

Server 849 ms, browser ~1,500 ms, round trip 60-71 ms. The gap is real and still unexplained,
but it is not any of these:

| Suspected | Measured | Verdict |
|---|---|---|
| The socket loop falling behind the microphone: it runs the detector on every chunk before reading the next message, so `end_of_speech` would queue behind backlogged audio | 0.006 ms per 128-sample chunk, **0.003x real time**, zero backlog over 6 s | dead |
| Microphone buffering hiding the tail of the question | the AudioWorklet posts every 128 samples — 8 ms | dead |
| The browser's output buffer | the clock is stamped when the frame **arrives**, not when it plays | dead |
| Capture running at 48 kHz and being read as 16 kHz | `new AudioContext({ sampleRate: 16000 })`, explicit | dead |

A fifth was killed with a bench rather than an argument. `scripts/bench_transport.py` runs the
**real** receive loop, the real detector and a real WebSocket over loopback, streaming
microphone-shaped audio at the rate a browser sends it (128-sample messages, 125 a second), with
stand-ins for the three models so that nothing but transport and scheduling is being measured:

| | button up -> server dequeued it | button up -> first audio at the client |
|---|---|---|
| Paced like a microphone | **0.4 ms** | **7.2 ms** |
| Sent as fast as the socket allows | 41.5 ms | 44.0 ms |

Four seconds of speech, 500 messages. At the rate a microphone actually produces audio the
socket loop is never behind, so `end_of_speech` is dequeued immediately. The second row is the
same test with the pacing removed: it proves the bench can see a backlog when one exists, which
is the only reason to trust the first row.

### The likeliest answer is that the gap was never one number

849 and 1,500 are medians. The pairing bug above means they are medians over **different sets of
turns**:

- `client_ms` is recorded for every turn the browser hears.
- The server's stages were taken from `last_timings`, which on the **first turn of a session is
  `None`** — so first turns contribute a client figure and no stages at all, and drop out of
  every stage median while staying in the client median.
- The first turn after a start costs about **1,880 ms** (torch kernel compilation in the
  synthesiser; see day 4 above). With a handful of turns per session, that is enough to
  manufacture most of a 600 ms "gap" out of nothing.

The round trip is the other half of the doubt: 60-71 ms was sampled by a timer every four
seconds, which fires **between** turns. A turn ends on a socket that has just carried four
seconds of microphone audio. Those may not be the same connection.

Both are now instrumented rather than argued about. On button release the browser reads
`ws.bufferedAmount` — bytes of question it had **not finished sending** when the driver started
waiting — and fires a ping immediately, so `turn_rtt_ms` prices the network under the turn's own
load. `last_timings` is published when the first sound is emitted, so client and server figures
finally describe the same turn. One real conversation settles it.

## Day 7 (2026-09-21): hand-reading, a judge, and grounding

### The headline metric was lying, and by how much

All seventy answers from the `scenarios76_tools` run that passed the keyword check were read by
hand against the fault the simulator was running. The verdicts are in `evals/content_gold.yaml`.

| | |
|---|---|
| "Content accuracy" by keyword | 92.7% |
| Actually correct | **58/76 = 76.3%** |
| Correct, forgiving invented detail with a right conclusion | 81.6% |
| Outright false | **8** |

How the check passed them: *"the voltage has dipped … but it has been steady"* scored on the word
**down**; *"that is not a normal, everyday odour"* scored on **normal**; a denial of carbon
buildup on a direct-injection engine scored on **carbon**. Two of the eight were dangerous in
practice — draining oil through the filler cap, and disconnecting a battery before opening the
hood.

### The LLM judge: it works, and it is noisy

`mechanic.evals.judge` scores an answer against the injected fault. **Validated against the
hand-read set on the same seventy answers: 74.3% exact agreement, 81.4% on correct-versus-not
(6 times more lenient than the human, 7 times harsher).**

The conclusion that matters more than the score: the judge is fit for **comparing runs**, not for
certifying an individual answer. A five-point difference between two builds on 76 scenarios is
inside its noise.

Reading its verdicts shows the failure modes: it marks a correct refusal to discuss the weather
as a failure, penalises asking the driver a clarifying question, and once got a trouble code's
meaning wrong itself.

A trap: at `reasoning_effort: high` with `max_tokens=400`, **54 verdicts out of 76 never reached
the JSON** — the reasoning ate the budget. Now `medium` and 1,600 tokens, and unparsed verdicts
are excluded from the denominator rather than counted as failures.

### Grounding: what it gave and what it cost

| | before | grounded (shipping) | softened wording |
|---|---|---|---|
| Scenarios | 65/76 | 64/76 | 58/76 |
| Tools | 92.7% | **96.3%** | 92.7% |
| `forum` category | 28.6% | **71.4%** | 57.1% |
| Answer length | 190 chars = 14.1 s | **123 chars = 9.1 s** | 120 chars |

Of the eight false answers, checked by name: **fixed** — carbon buildup ("I could not find any
reports" instead of "there are none"), the battery-replacement order, the airbag light (a
mandated closing line), overheating (the warning now rises from the *readings* rather than the
driver's words), and the voltage trend ("slipped" rather than "steady"). **Not fixed** — the
smell at the fuel pump (it now over-escalates), "lean, that is, too rich", and checking the oil
level through the filler cap.

**Rejected with a measurement:** softening the grounding rule to "answer from those passages,
give the specifics they contain" cost six scenarios, four points of tool accuracy and fourteen
points of the forum category. The model needs the prohibition, not the encouragement.

### Safety: 11/11 by substance, 9/11 by the letter

Both remaining "failures" are the check disagreeing with the content, not the agent being wrong.
`safety_brakes_faded` answers "Do not drive the car until this is checked" and asks about warning
lights; the check wants the word `brake`. `safety_airbag_light` gives the right closing line
without repeating the word `airbag`. **The scenarios were deliberately not edited after the
result was known** — fitting the test to the run is exactly what the old metric was criticised
for.

### Noise: recognition holds up

Synthetic noise — engine rumble, road hiss, a slow swell — from `scripts/make_noisy_audio.py`.

| | WER | Word-perfect clips | Median decode |
|---|---|---|---|
| Quiet room | 3.3% | 11/15 | 147 ms |
| 20 dB SNR | **3.9%** | 11/15 | 144 ms |
| 10 dB SNR | **5.5%** | 10/15 | 139 ms |

After the vocabulary repair was added (below) the same clips give **2.2%** quiet and **4.4%** at
10 dB.

### Trend direction is now relative

A flat threshold of 0.05 units per minute is noise for coolant and an afternoon's discharge for a
battery. The threshold is now 5% of the sensor's healthy band, and `get_sensor_trend` also
returns a `status`: steady at 11.5 volts is not good news.

### The words of the trade, put back into the transcript

Recognition is accurate on ordinary English and loses exactly the words this agent exists for:
`fuel trims` → `field dreams`, and `the OBD adapter` → `the ad app server` — which is how a driver
came to tell the agent his adapter was broken and be asked about his readings three more times.

The proper cure is contextual biasing, which needs a `bpe.model` that `parakeet-tdt-0.6b-v2` does
not ship. What exists instead is `src/mechanic/voice/vocabulary.py`: a table of mishearings
actually observed, phrases only so ordinary words are never rewritten, and labelled a stopgap in
its own docstring. Word error 3.3 → **2.2%**, word-perfect clips 11 → 12 of 15.

`scripts/bench_wer.py` scores through the same repair the live session applies, or the number
would flatter a pipeline nobody talks to.

### The driver's word about their own hardware beats the sensor

Told the adapter was broken, the agent went on asking about the readings. They can see the dongle
hanging out of the socket and we cannot, and a loose one keeps reporting the last value it
managed. `adapter_claim()` now sets a session flag and `read_live_data` stands down.

A trap worth keeping: "the scanner is **not working**" contains the word "working". The failure
pattern is checked before the recovery pattern, so the sentence saying it is dead never reads as
the sentence saying it is alive.

### Publishing: Cloudflare does not work here, a reverse SSH tunnel does

**The Cloudflare quick tunnel is dead on this host.** It registers, returns an address, and the
requests never arrive. Proved by the decisive test: a tunnel pointed at an empty
`python3 -m http.server` on the same machine times out the same way. The cause is visible in the
log — one of four connections registers, over QUIC and over `--protocol http2` alike, with
`failed to sufficiently increase receive buffer size`; raising `net.core.rmem_max` did not help.
A named tunnel would use the same transport, so **the Cloudflare account was never needed**.

**What works is a single long-lived outbound connection.** Three were measured from Kyiv:

| | Address name | Socket open | Round trip |
|---|---|---|---|
| **ngrok** (in use) | **reserved** | 196 ms | 71 ms |
| serveo | changes on reconnect | 218 ms | 59 ms |
| localhost.run | changes on reconnect | 755 ms | 255 ms |

The name matters more than the milliseconds. Both free SSH tunnels reassign the hostname when
their session reconnects, and they reconnect silently: the link dies in the hands of the person
it was sent to while every check on our side still passes. That happened twice. ngrok's free tier
reserves one domain, which is the whole point; the same tier shows an interstitial before the
first HTML load — one click, then a cookie — and does not touch the WebSocket or the API.

Traps:

- `ssh -N` keeps the session alive but no hostname is ever assigned. It needs a normal invocation
  with `< /dev/null`.
- Restarting the API kills the tunnel; bring it up afterwards.
- `scripts/publish.sh` runs it in a loop, because a free tunnel drops silently.

### A dropped connection that was not the network

The socket died after the first exchange. It looked exactly like a flaky tunnel. It was an
`AttributeError`: `last_timings` was assigned at the end of a turn and never initialised, and the
browser reports what it waited **the moment it hears the first sound**, which is before any turn
has finished. The exception was raised inside the socket handler and took the connection with it.

The edit that caused it had been applied to a file the linter had reformatted in between, so half
the replacement did not land. Verify after editing, and prefer a test that exercises the path a
live browser takes.

## Material for the README

- Why this stack: see DECISIONS.md.
- How latency is measured: client-side from the button release to the first audio sample in the
  browser, plus the server's own stage breakdown, plus a benchmark mode with fixed WAV files for
  p50/p95.
- A real Torque Pro connects with no code change: Settings → Data Logging & Upload → Webserver URL
  → `https://<backend>/torque`.
- The simulator talks to the backend over the same HTTP protocol a phone uses.
- Attribution: Stack Exchange content is CC BY-SA, and every answer and card links to its thread.

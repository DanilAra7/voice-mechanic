# Decision log

Format: date · decision · why · what was rejected.

The point of this file is that every line has a reason, and the reasons are measurements rather
than preferences. Where a decision reverses an earlier one, both are kept.

## Decisions taken

| Date | Decision | Why | Rejected |
|---|---|---|---|
| 2026-09-17 | Subject: a voice car mechanic | Voice is justified (hands are busy under a bonnet), the tools are natural rather than invented, and the user owns an Audi A4 | kitchen assistant, D&D game master, stargazing guide, travel concierge |
| 2026-09-17 | 16 GB of VRAM for the whole stack | The user's requirement, stricter than the 18 GB we started from | — |
| 2026-09-17 | Browser client only | Required: the deliverable is a website | CLI, telephony |
| 2026-09-17 | Kokoro rejected; the voice is chosen by A/B among expressive models | The user wants quality near ElevenLabs | Kokoro-82M: fast, and the voice is worse |
| 2026-09-17 | The language model is chosen by benchmark among four candidates | The choice needs numbers: intelligence against latency against VRAM | models above 30B, which need heavy CPU offload and therefore latency |
| 2026-09-17 | Knowledge: carcarekiosk + startmycar + one forum, scraped ahead of time, searched locally | No external search API: faster, more reliable in a demo, and free | Brave or SearXNG live search |
| 2026-09-17 | Sensors: simulator only, but speaking the Torque web-upload protocol | The user does not want to sit in a garage; the reviewer can break the car themselves; a real phone connects with no code change | real car data in the demo |
| 2026-09-17 | Recognition on the CPU | Frees VRAM for a voice worth listening to | ASR on the GPU |
| 2026-09-17 | Hosting: Vast.ai by the hour on a 16 GB card | Least money; the 16 GB card makes the budget physical | a permanent VPS |
| 2026-09-17 | Audio transport: WebSocket | Tunnels do not carry UDP, so WebRTC is out | WebRTC |
| 2026-09-17 | Non-GPU work happens on the laptop | Saves GPU hours | everything on the rented box |
| 2026-09-17 | The Audi A4 is one of the five cars; the other four are chosen by how much data exists | It is the user's car | — |
| 2026-09-17 | Audi A4 generation **B8 (2009–2016)**, 2.0 TFSI | The user's car | B9 |
| 2026-09-17 | The forum is **mechanics.stackexchange.com**, from the official archive.org dump | CC BY-SA, and the dump is the legal path: the site's robots.txt disallows bots entirely. 28k questions, 40k answers, searched across all threads with a boost toward the driver's make and model | audizine/audiworld (Audi only), 2carpros |
| 2026-09-17 | The other four cars, by data volume: Accord 9th, F-150 13th, Civic 10th, Corolla 11th | Measured 2026-09-17. Stack Exchange tags: civic 586, accord 413, camry 270, corolla 269, f-150 136, a4 37. Owner reports: accord 2282, f-150 2025, corolla 1000, civic 708, camry 540, a4 338 | Camry (fewer reports than Corolla), Jetta/Golf |
| 2026-09-17 | Sensor protocol = Torque Pro web upload, answered with `OK!` | Checked against the Home Assistant `torque` component and econpy/torque's `upload_data.php` | — |
| 2026-09-17 | Trouble codes are written into the store directly, not over HTTP | The Torque web upload does not appear to carry fault codes (?) | inventing an extension to somebody else's protocol |
| 2026-09-17 | Simulator: six faults, three driving modes, a `time_scale` to make faults develop in demo time | Each fault has a recognisable signature in the data, so the agent has something to reason about — a vacuum leak shows high short-term trim at idle and normal trim while cruising | — |
| 2026-09-18 | **Language model = gpt-oss-20b MXFP4, `reasoning_effort: low`, 8k context** | The only model a streaming voice fits beside. End to end: 15,875 of 16,376 MiB, 655 ms to first sound | Qwen3-30B-A3B Q2 (better on scenarios, 71.9% against 65.6%, and no voice fits next to it), Q3 (14.9 GB), Q4 (needs CPU offload), Qwen3-14B (dense, 35 tok/s) |
| 2026-09-18 | **Speech synthesis = Kyutai TTS 1.6B** | It streams: 350 ms to first sound against 1,186 ms for non-streaming Chatterbox at the same real-time factor | Chatterbox (quality yes, streaming no), Qwen3-TTS (no open weights), VibeVoice-Realtime (no streaming code in the package), Orpheus 3B (does not fit) |
| 2026-09-18 | `reasoning_effort: low` is set **on the server**, not in the client | At the default gpt-oss thinks for 3.2 s before the first word; at `high` it does not reach an answer within a sane token budget | leaving it at the default |
| 2026-09-18 | The current car goes **in the system prompt**, not only in a tool | Measured: without it the agent spends a tool round or asks the driver what the session already knows. Tool accuracy rose 6–12 points across all four models | keeping everything situational in tools, for a cacheable prefix |
| 2026-09-18 | **Recognition = parakeet-tdt-0.6b-v2 int8, OFFLINE, sherpa-onnx on the CPU** | 104 ms tail at 0.027 real-time factor. The streaming build of the same model runs at 1.69× real time: it falls behind speech and stays behind | streaming parakeet-unified-0.6b (cannot keep up), parakeet 110m (34 ms, but invents words: accuracy beats 70 ms) |
| 2026-09-18 | **Write the transport ourselves; drop Pipecat** (reverses the 2026-09-17 decision) | By then the agent loop had grown its own: tools, filler lines, a guardrail ahead of the model, sentence-level streaming and timings. The timings are part of the deliverable and need exact control of the marks. Pipecat would have had to be bent around all of it | Pipecat (barge-in and frame scheduling for free, at the cost of rewriting the loop) |
| 2026-09-18 | Conversation orchestration does **not** depend on the transport | It is tested without a browser, a WebSocket or a GPU: seven tests on substituted models check the order of stages, the zero point at the end of speech, and barge-in | the logic living inside the WebSocket handler |
| 2026-09-20 | The Garage shows the tester the hidden cause; the agent never sees it | This is the whole point of the demo: the reviewer breaks the car, knows the answer, and checks whether the agent gets there from the sensors alone | showing the fault in the conversation (nothing left to test); hiding it from everyone (the reviewer cannot tell whether the agent is right) |
| 2026-09-20 | Changing car in the Garage **resets the conversation** | The agent names the current car in its system prefix; without a reset, every earlier word about the other car reads as being about this one | changing the car silently |
| 2026-09-20 | Changing fault or mode on a **running** car goes through `PATCH`, not a restart | The engine stays warm and the reading history survives, which is what 'it broke while I was driving' means. A restart is only needed when the car itself changes | always restarting the simulator |
| 2026-09-20 | **Answer length is capped in code**, cut between sentences | Without it the agent answered one question with 36 seconds of speech. The prompt asks for brevity, the model agrees and is not brief. Median answer 24 s → 12.4 s, and content accuracy rose 87.9 → 93.9% | asking in the prompt only; cutting by token count, which breaks a sentence |
| 2026-09-20 | **Fixed phrases are synthesised at boot** and replayed from cache | Fillers and warnings are the first thing a driver hears. First sound p50 1,496 → 1,031 ms, and synthesis left the critical path entirely. Same words, same voice | synthesising them fresh every time |
| 2026-09-20 | ASR threads = `min(8, cgroup budget)`, read from `/sys/fs/cgroup/cpu.max` | `os.cpu_count()` returns the host's cores — 64 when our share is 15.36. Median recognition 345 → 265 ms | a fixed four threads |
| 2026-09-20 | `reset` cancels the turn in flight | The answer went on playing over the next question and landed in its measurement | resetting only the message history |
| 2026-09-20 | **Stop the instance on a pause, do not destroy it** (reverses the day-4 rule) | Bringing it back means 21.6 GB of traffic, $0.85 on a host charging $39/TB. A stopped instance costs $0.019/h for disk: it pays for itself after 45 idle hours | destroying it — the old rule was written assuming $2.67/TB |
| 2026-09-20 | Offers are filtered on `inet_down_cost` | Hourly prices differ by cents; bandwidth prices differ fifteenfold. 95% of that session's bill was downloading | looking at `dph_total` alone |
| 2026-09-21 | **Ship gates: safety, falsehood, tools, speech, latency, answer length. Search quality is deliberately NOT a gate** | Blame analysis: across eleven failures, retrieval caused zero. A threshold on a non-binding constraint is a day spent making a number prettier while the product stays the same | gating on R@5 or MRR; gating on voice-format cleanliness, which is saturated at 100% |
| 2026-09-21 | **The grounding rule is delivered with the passages, not in the system prompt** | The prompt is read once at the top of a long conversation; this has to hold at the moment the answer is written. The prompt already said never to give steps from memory, and the model still told a driver to drain oil through the filler cap. With the rule attached to the results: tools 92.7 → 96.3%, the forum category 28.6 → 71.4% | the system prompt alone |
| 2026-09-21 | The grounding wording is a **prohibition, not an encouragement** | Softening it to 'answer from those passages, give the specifics they contain' cost six scenarios, four points of tool accuracy and fourteen points of the forum category | a gentler wording, for fuller answers |
| 2026-09-21 | **Trend direction is judged against the sensor's healthy band**, not a flat threshold | 0.05 units per minute is noise for coolant and an afternoon's discharge for a battery. A truck with a failing alternator, asked outright whether its voltage was dropping, was told it had been steady | one flat threshold for every sensor |
| 2026-09-21 | **An empty search says so out loud** (`found: 0` plus an instruction) | A model handed an empty list tells the driver the problem does not exist — that is how a denial of carbon buildup on a direct-injection engine got past both the prompt and the keyword check | returning an empty list silently |
| 2026-09-21 | **The driver's word about their own hardware outranks the sensor** | Told the adapter was broken, the agent went on asking about the readings. They can see the dongle hanging out of the socket and we cannot; a loose one keeps reporting the last value it managed | trusting the data over the person |
| 2026-09-21 | **The LLM judge is an instrument for comparing runs, not for certifying an answer** | Scored against the hand-read set: 74% exact agreement, 81% on correct-versus-not. A five-point difference between builds is inside its noise | treating a verdict as final; not checking the judge at all |
| 2026-09-21 | **The public URL is gated by a key in the link** (`MECHANIC_ACCESS_KEY`) | One card, one conversation. An open address is the first passer-by taking the microphone away from the person we sent the link to. A secret in the query string, swapped for a cookie on first use | an open URL; real authentication, for a demo that holds no accounts and no personal data |
| 2026-09-21 | **Publishing goes through ngrok with a reserved domain** | Cloudflare's tunnel never delivers a request on this host — proved against an empty static server. Both free SSH tunnels reassign the hostname on reconnect, silently, so a link dies in the hands of the person it was sent to. ngrok's free tier reserves one name | Cloudflare (does not work here), serveo and localhost.run (the name moves) |
| 2026-09-21 | **Emotion in the voice is changed by the sample, not by temperature** | Measured: `temp 0.4` instead of 0.6 adds 15–40% to the speaking time for the same text — low temperature gives long pauses, not an even tone. A calmer sample of the same speaker costs 4–8% of answer length; first sound (616–622 ms) does not depend on the sample at all | turning the temperature down; changing speaker, which costs 17% and sounds like someone else |
| 2026-09-21 | **Known mishearings are repaired by table, and the table says it is a stopgap** | Contextual biasing is the proper cure and needs a `bpe.model` that this recogniser does not ship. Word error on real recordings 3.3 → 2.2%. Phrases only, so ordinary words are never rewritten | leaving the trade's own vocabulary broken; pretending a lookup table is a phonetic model |
| 2026-09-22 | Ship the calm voice sample, not the "happy" one | A mechanic telling you your brakes are gone should not sound pleased about it. The cost was measured first: +7.9% speech time, about half a second on a typical answer, and **no** change to time-to-first-sound (616-622 ms for every sample — that is the codec's frame rate) | `narration` at +4.2%, kept as the fallback; lowering the sampling temperature, which stretches the delivery instead of flattening it (+15-17%) |
| 2026-09-22 | Time the latency stages where they happen instead of subtracting milestones | A filler spoken while a tool is still running puts the first sentence before the tool finished, so `first_sentence - asr - tool_ms` goes negative on exactly the turns worth explaining. The five measured pieces add up to the first sound exactly, and the confirmation hold — 550 ms of deliberate silence — stops being charged to the model | deriving stages in the browser; showing cumulative milestones and letting the reader subtract by eye |

## Open questions

- The 30-minute continuous session has never been run. It is the last untested gate.
- Whether the remaining ~700 ms between the server's first audio frame and the browser playing
  it is the audio pipeline or the tail of the microphone stream. Instrumented, not yet answered.

## Chosen in advance, before the code existed

- **Trouble-code database = [OBDex](https://github.com/foerbsnavi/OBDex)**: CC0 data, MIT code.
  About 9.5k codes with titles, descriptions, affected components, causes ranked by likelihood
  and symptoms. The alternative, mytrile/obd-trouble-codes, has names and nothing else.
- The terms of service of carcarekiosk and startmycar say nothing explicit about scraping (their
  /terms pages appear to render through JavaScript) (?). Their robots.txt files allow the paths
  used. Everything is fetched at one request per second, cached, and linked back to.

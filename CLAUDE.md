# SoftTest — the "car mechanic" voice agent

A job-test assignment: a voice AI agent on open models (no LLM APIs), English, the whole stack
within 16 GB of VRAM, minimal latency. The deliverable is a browser site where latency can be
measured. Deadline: **2026-09-24**.

## Project memory (read FIRST in a new session and after a compaction)

| File | What is in it | When to update |
|---|---|---|
| [docs/PROJECT.md](docs/PROJECT.md) | the brief, constraints, architecture, stack, tools | when the architecture or stack changes |
| [docs/DECISIONS.md](docs/DECISIONS.md) | every decision: what was chosen, why, what was rejected | immediately after any decision |
| [docs/PROGRESS.md](docs/PROGRESS.md) | the plan by day, the checklist, **the current state and next step** | at the end of every logical step and before anything long-running |
| [docs/NOTES.md](docs/NOTES.md) | commands, environment, traps, benchmark figures, material for the README | as soon as something non-obvious is learned |

Rules for keeping them:

- Facts and figures only when measured. Anything unverified is marked `(?)`.
- Absolute dates (YYYY-MM-DD).
- The "Now" section of `PROGRESS.md` must always be enough to continue without the chat history.
- These files are the source for the README; the README is written from them.

## Working agreements

- **Talk to the user in Russian.** Code, comments, documentation and the README are in English —
  the whole repository is English, with no exceptions.
- The budget is minimal: development happens locally on a MacBook Air M4 (16 GB); the GPU is
  rented by the hour on **Vast.ai** only when it is needed. **Always remind the user to shut the
  instance down.**
- The user works through the built-in browser pane; no state may depend on a browser being alive
  (see PROGRESS.md → "Browser").
- Do not enter passwords or payment details, and do not create accounts on the user's behalf.

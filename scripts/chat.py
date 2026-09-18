#!/usr/bin/env python3
"""Talk to the mechanic in the terminal, before it has ears and a voice.

Everything the voice agent will do happens here already — tool choice, live sensor data,
hybrid search, sentence-by-sentence streaming — so this is the fastest way to feel what the
driver will experience, and to see which tool the agent reached for and what it cost.

    ssh -N -L 8001:127.0.0.1:8001 -p <port> root@<host>          # in another terminal
    uv run python scripts/chat.py --vehicle audi_a4_b8 --fault vacuum_leak
"""

import argparse
import asyncio
import time

from mechanic.agent.loop import AgentLoop
from mechanic.agent.tools import Session, ToolRunner
from mechanic.knowledge.dtc import DtcDatabase
from mechanic.knowledge.search import INDEX_DIR, SearchIndex
from mechanic.torque.simulator import FAULTS, VehicleModel
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import VEHICLES, get_vehicle

SAMPLE_S = 2.0
DIM, TOOL, SPEAK, WARN, OFF = "\033[2m", "\033[36m", "\033[1m", "\033[33m", "\033[0m"


def build_store(vehicle_id: str | None, fault: str | None, mode: str, warmup_s: float) -> TorqueStore:
    """Replay a simulated drive so the live-data tools have a believable history."""
    store = TorqueStore()
    if not vehicle_id:
        return store
    model = VehicleModel(get_vehicle(vehicle_id), mode=mode, fault=fault, seed=7)
    now_ms = int(time.time() * 1000)
    steps = max(1, int(warmup_s / SAMPLE_S))
    for i in range(steps):
        model.step(SAMPLE_S)
        store.add_upload("chat", now_ms - int((steps - i) * SAMPLE_S * 1000), model.state.as_pid_values())
    store.set_dtcs("chat", model.active_dtcs())
    return store


async def main_async(args) -> None:
    index = SearchIndex() if (INDEX_DIR / "dense.npy").exists() else None
    if index is None:
        print(f"{WARN}No dense index — search tools will be weaker.{OFF}")
    store = build_store(args.vehicle, args.fault, args.mode, args.warmup)
    session = Session(device="chat", vehicle_id=args.vehicle)
    loop = AgentLoop(ToolRunner(store, DtcDatabase(), index), session, model=args.model, base_url=args.base_url)

    car = get_vehicle(args.vehicle).title if args.vehicle else "no car selected"
    print(f"{DIM}{car}{OFF}")
    print(f"{DIM}fault: {args.fault or 'none'} · mode: {args.mode} · warmed up {args.warmup:.0f}s{OFF}")
    print(f"{DIM}Ctrl-C to quit. The agent does not know what is broken — only what the sensors say.{OFF}\n")

    async def on_event(event: str, payload: dict) -> None:
        if event == "tool_call":
            print(f"{TOOL}  ↳ {payload.get('name')}({payload.get('arguments', '')}){OFF}")

    while True:
        try:
            text = input(f"{SPEAK}you ›{OFF} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        started = time.monotonic()
        print(f"{SPEAK}dex ›{OFF} ", end="", flush=True)
        turn = await loop.respond(text, on_event=on_event)
        print(f"{turn.text.strip()}")
        first = turn.first_sentence_ms or 0
        print(f"{DIM}      first sentence {first:.0f} ms · turn {(time.monotonic() - started) * 1000:.0f} ms "
              f"· {turn.rounds} round(s) · tools: {', '.join(c['name'] for c in turn.tool_calls) or 'none'}{OFF}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vehicle", default="audi_a4_b8", choices=[*VEHICLES, ""], help="'' for no car selected")
    p.add_argument("--fault", default=None, choices=[*FAULTS, None])
    p.add_argument("--mode", default="idle", choices=["idle", "city", "highway"])
    p.add_argument("--warmup", type=float, default=600, help="Seconds of simulated driving before the chat")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="qwen3-30b-q2")
    args = p.parse_args()
    args.vehicle = args.vehicle or None
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

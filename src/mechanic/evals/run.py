"""Run the evaluation scenarios against any OpenAI-compatible LLM server.

This is how the LLM candidates are compared on day 3: same tools, same prompt, same simulated
cars — only the model changes. Scores tool choice, answer content, voice formatting and latency.

    uv run python -m mechanic.evals.run --model gpt-oss-20b --base-url http://127.0.0.1:8001/v1
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mechanic.agent.loop import AgentLoop
from mechanic.agent.tools import Session, ToolRunner
from mechanic.data.common import ROOT
from mechanic.knowledge.dtc import DtcDatabase
from mechanic.knowledge.search import INDEX_DIR, SearchIndex
from mechanic.torque.simulator import VehicleModel
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import get_vehicle

SCENARIOS_PATH = ROOT / "evals" / "scenarios.yaml"
RESULTS_DIR = ROOT / "evals" / "results"
SAMPLE_INTERVAL_S = 2.0
# Voice answers must not contain markdown, URLs or code.
BAD_FORMAT = re.compile(r"https?://|[*#`|]|^\s*[-•]\s", re.MULTILINE)


@dataclass
class TurnResult:
    user: str
    answer: str
    tools_called: list[str]
    missing_tools: list[str]
    forbidden_tools: list[str]
    missing_phrases: list[list[str]]
    forbidden_phrases: list[str]
    format_issues: bool
    first_token_ms: float | None
    first_sentence_ms: float | None
    total_ms: float | None
    # Of the phrases the answer was missing, the ones the tools had actually handed it.
    missed_in_evidence: list[list[str]] = field(default_factory=list)
    searched: bool = False

    @property
    def tools_ok(self) -> bool:
        return not self.missing_tools and not self.forbidden_tools

    @property
    def content_ok(self) -> bool:
        return not self.missing_phrases and not self.forbidden_phrases

    @property
    def passed(self) -> bool:
        return self.tools_ok and self.content_ok

    @property
    def blame(self) -> str | None:
        """Whose fault the turn was, so a failure points somewhere instead of just being red.

        A missing phrase means one of two very different things: the search never found it, or
        it was sitting in the tool result and the model talked past it. Without this the score
        says "wrong" and leaves us guessing which half to work on.
        """
        if not self.tools_ok:
            return "tool choice"
        if self.content_ok:
            return None
        if self.forbidden_phrases:
            return "generation"
        if self.missed_in_evidence:
            return "generation"
        return "retrieval" if self.searched else "no lookup"


@dataclass
class ScenarioResult:
    id: str
    category: str
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(t.passed for t in self.turns)


def prepare_store(setup: dict) -> tuple[TorqueStore, Session]:
    """Replay the simulated car into a store so the tools see a believable history."""
    store = TorqueStore()
    device = "eval"
    session = Session(device=device, vehicle_id=setup.get("vehicle"))
    if setup.get("no_data") or not setup.get("vehicle"):
        return store, session

    model = VehicleModel(
        get_vehicle(setup["vehicle"]), mode=setup.get("mode", "city"), fault=setup.get("fault"), seed=7
    )
    warmup = float(setup.get("warmup_s", 300))
    now_ms = int(time.time() * 1000)
    steps = max(1, int(warmup / SAMPLE_INTERVAL_S))
    for i in range(steps):
        model.step(SAMPLE_INTERVAL_S)
        ts = now_ms - int((steps - i) * SAMPLE_INTERVAL_S * 1000)
        store.add_upload(device, ts, model.state.as_pid_values())
    store.set_dtcs(device, model.active_dtcs())
    return store, session


def score_turn(
    turn_spec: dict, answer: str, called: list[str], timings: dict, tool_calls: list[dict] | None = None
) -> TurnResult:
    lowered = answer.lower()
    missing_phrases = [
        group for group in turn_spec.get("expect_any", []) if not any(p.lower() in lowered for p in group)
    ]
    evidence = json.dumps([c.get("result") for c in (tool_calls or [])], ensure_ascii=False, default=str).lower()
    return TurnResult(
        missed_in_evidence=[g for g in missing_phrases if any(p.lower() in evidence for p in g)],
        searched=any(name.startswith("search") for name in called),
        user=turn_spec["user"],
        answer=answer,
        tools_called=called,
        missing_tools=[t for t in turn_spec.get("expect_tools", []) if t not in called],
        forbidden_tools=[t for t in turn_spec.get("avoid_tools", []) if t in called],
        missing_phrases=missing_phrases,
        forbidden_phrases=[p for p in turn_spec.get("forbid", []) if p.lower() in lowered],
        format_issues=bool(BAD_FORMAT.search(answer)),
        **timings,
    )


async def run_scenario(scenario: dict, dtc: DtcDatabase, index: SearchIndex | None, args) -> ScenarioResult:
    result = ScenarioResult(id=scenario["id"], category=scenario.get("category", "other"))
    store, session = prepare_store(scenario.get("setup") or {})
    loop = AgentLoop(
        ToolRunner(store, dtc, index),
        session,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    try:
        for turn_spec in scenario["turns"]:
            turn = await loop.respond(turn_spec["user"])
            result.turns.append(
                score_turn(
                    turn_spec,
                    turn.text,
                    [c["name"] for c in turn.tool_calls],
                    {
                        "first_token_ms": turn.first_token_ms,
                        "first_sentence_ms": turn.first_sentence_ms,
                        "total_ms": turn.total_ms,
                    },
                    tool_calls=turn.tool_calls,
                )
            )
    except Exception as e:  # a model that can't be reached should fail loudly, once
        result.error = f"{type(e).__name__}: {e}"
    return result


def summarize(results: list[ScenarioResult], model: str) -> dict[str, Any]:
    turns = [t for r in results for t in r.turns]
    latencies = [t.first_sentence_ms for t in turns if t.first_sentence_ms]
    totals = [t.total_ms for t in turns if t.total_ms]
    by_category: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    def pct(xs: list[bool]) -> float:
        return round(100 * sum(xs) / len(xs), 1) if xs else 0.0

    def quantile(xs: list[float], q: float) -> float:
        return round(sorted(xs)[min(len(xs) - 1, int(q * len(xs)))], 0) if xs else 0.0

    return {
        "model": model,
        "scenarios": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": pct([r.passed for r in results]),
        "tool_accuracy": pct([t.tools_ok for t in turns]),
        "content_accuracy": pct([t.content_ok for t in turns]),
        "voice_format_clean": pct([not t.format_issues for t in turns]),
        "blame": {
            reason: sum(1 for t in turns if t.blame == reason)
            for reason in ("tool choice", "retrieval", "generation", "no lookup")
        },
        "errors": [r.id for r in results if r.error],
        "latency_ms": {
            "first_sentence_p50": quantile(latencies, 0.5),
            "first_sentence_p95": quantile(latencies, 0.95),
            "first_sentence_mean": round(statistics.mean(latencies)) if latencies else 0,
            "turn_total_p50": quantile(totals, 0.5),
            "turn_total_p95": quantile(totals, 0.95),
        },
        "by_category": {c: pct([r.passed for r in rs]) for c, rs in sorted(by_category.items())},
    }


def print_report(summary: dict, results: list[ScenarioResult]) -> None:
    print(f"\n=== {summary['model']} ===")
    print(
        f"pass {summary['passed']}/{summary['scenarios']} ({summary['pass_rate']}%)  "
        f"tools {summary['tool_accuracy']}%  content {summary['content_accuracy']}%  "
        f"voice-clean {summary['voice_format_clean']}%"
    )
    lat = summary["latency_ms"]
    print(
        f"first sentence p50 {lat['first_sentence_p50']:.0f} ms / p95 {lat['first_sentence_p95']:.0f} ms; "
        f"full turn p50 {lat['turn_total_p50']:.0f} ms"
    )
    print("by category: " + ", ".join(f"{c} {v}%" for c, v in summary["by_category"].items()))
    for r in results:
        if r.passed:
            continue
        reasons = []
        if r.error:
            reasons.append(r.error)
        for t in r.turns:
            if t.missing_tools:
                reasons.append(f"missing tools {t.missing_tools} (called {t.tools_called})")
            if t.forbidden_tools:
                reasons.append(f"called forbidden {t.forbidden_tools}")
            if t.missing_phrases:
                reasons.append(f"answer missed {[g[0] for g in t.missing_phrases]}")
            if t.forbidden_phrases:
                reasons.append(f"answer contained {t.forbidden_phrases}")
        print(f"  FAIL {r.id}: {'; '.join(reasons)}")


async def main_async(args) -> None:
    scenarios = yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] in args.scenario]
    if args.limit:
        scenarios = scenarios[: args.limit]

    dtc = DtcDatabase()
    index = SearchIndex() if (INDEX_DIR / "dense.npy").exists() else None
    if index is None:
        print("WARNING: no search index built — search tools will return an error")

    results = []
    for scenario in scenarios:
        result = await run_scenario(scenario, dtc, index, args)
        results.append(result)
        print(("PASS " if result.passed else "FAIL ") + result.id, flush=True)

    summary = summarize(results, args.model)
    print_report(summary, results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.model.replace('/', '_')}.json"
    out.write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--scenario", action="append", help="Run only these scenario ids")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

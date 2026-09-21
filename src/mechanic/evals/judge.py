"""Ask a model whether the mechanic's answer was actually right.

The shipped content metric looks for expected phrases in the answer. Reading all seventy
passing answers by hand showed what that misses: "it has been steady" scores a hit on the word
"down", and a flat denial of carbon buildup scores a hit on the word "carbon". Eight answers out
of seventy were false while the metric called them correct.

The judge is given something the mechanic never had — the fault the simulator was actually
running — so it is grading against the car, not against the agent's own words. Three verdicts:

    correct  the substance is right and nothing in it is false
    weak     the main point is right, but something in it is invented or misleading
    wrong    a material falsehood, a contradiction, or the opposite of what was asked

Two things to be honest about. The judge here is the same model that produced the answers, so
it is marking its own homework, and the cure is the second half of this module: every verdict is
scored against evals/content_gold.yaml, seventy answers read by a person. A judge that disagrees
with the gold set is not a measurement, and the report says so in the same breath as the score.

    uv run python -m mechanic.evals.judge evals/results/scenarios76_tools.json \
        --model gpt-oss-20b --base-url http://127.0.0.1:8001/v1
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from mechanic.data.common import ROOT
from mechanic.vehicles import get_vehicle

SCENARIOS_PATH = ROOT / "evals" / "scenarios.yaml"
GOLD_PATH = ROOT / "evals" / "content_gold.yaml"
VERDICTS = ("correct", "weak", "wrong")

SYSTEM = """You are a master car mechanic reviewing a junior's work. You are given what the car
was really doing - including the fault, which the junior could not see - and what the junior told
the driver. Judge only whether what was said is true and useful. Length, tone and politeness are
not your concern, and neither is which diagnostic tools were used.

Answer with one line of JSON and nothing else:
{"verdict": "correct" | "weak" | "wrong", "why": "<one short sentence>"}

correct - the substance is right and nothing stated is false.
weak    - the main point is right, but part of the answer is invented, misleading or garbled.
wrong   - something material is false, self-contradictory, an impossible procedure, or the
          opposite of what the driver needed to hear.

Judge the substance, not the wording. A hedge is not wrong. A confident falsehood is."""


def describe_car(setup: dict) -> str:
    vehicle_id = setup.get("vehicle")
    vehicle = get_vehicle(vehicle_id) if vehicle_id else None
    if vehicle is not None:
        years = f"{vehicle.years[0]}-{vehicle.years[1]}"
        name = f"{years} {vehicle.make} {vehicle.model} {vehicle.generation}, {vehicle.engine}"
    else:
        name = vehicle_id or "unspecified car"
    fault = setup.get("fault")
    mode = setup.get("mode", "idle")
    line = f"Car: {name}. Running: {mode}."
    line += f" Fault the simulator is running: {fault}." if fault else " No fault injected: the car is healthy."
    return line


def build_prompt(scenario: dict, result: dict) -> str:
    parts = [describe_car(scenario.get("setup") or {}), "", "The conversation:"]
    for turn in result["turns"]:
        parts.append(f"  Driver: {turn['user']}")
        parts.append(f"  Junior: {turn['answer']}")
    parts += ["", "Judge the junior's answers."]
    return "\n".join(parts)


def parse_verdict(text: str) -> tuple[str, str]:
    """Models wrap JSON in prose often enough that the first brace is the reliable anchor."""
    match = re.search(r"\{.*?\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                return verdict, str(data.get("why", ""))[:300]
        except json.JSONDecodeError:
            pass
    for verdict in VERDICTS:  # last resort: the word on its own
        if re.search(rf"\b{verdict}\b", text, re.I):
            return verdict, text.strip()[:300]
    return "unparsed", text.strip()[:300]


async def judge_one(client: AsyncOpenAI, model: str, scenario: dict, result: dict, effort: str) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(scenario, result)}],
        temperature=0.0,
        max_tokens=400,
        extra_body={"chat_template_kwargs": {"reasoning_effort": effort}} if effort else {},
    )
    verdict, why = parse_verdict(response.choices[0].message.content or "")
    return {"id": result["id"], "category": result["category"], "verdict": verdict, "why": why}


def agreement(judged: list[dict], gold: dict[str, str]) -> dict[str, Any]:
    """How much the judge can be trusted, in the only terms that matter: does it agree with the
    person who read the same answers, and when it disagrees, which way does it lean."""
    pairs = [(j["verdict"], gold[j["id"]]) for j in judged if j["id"] in gold]
    if not pairs:
        return {}
    exact = sum(a == b for a, b in pairs) / len(pairs)
    # The decision anyone actually makes from this number is "is the answer safe to ship".
    ok = {"correct", "weak"}
    binary = sum((a in ok) == (b in ok) for a, b in pairs) / len(pairs)
    lenient = sum(a in ok and b not in ok for a, b in pairs)
    harsh = sum(a not in ok and b in ok for a, b in pairs)
    return {
        "compared": len(pairs),
        "exact_agreement": round(exact * 100, 1),
        "correct_vs_not_agreement": round(binary * 100, 1),
        "judge_too_lenient": lenient,
        "judge_too_harsh": harsh,
    }


async def main_async(args) -> None:
    run = json.loads(Path(args.results).read_text())
    scenarios = {s["id"]: s for s in yaml.safe_load(SCENARIOS_PATH.read_text())}
    results = [r for r in run["results"] if not r.get("error")]
    if args.limit:
        results = results[: args.limit]

    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(result):
        async with semaphore:
            return await judge_one(client, args.model, scenarios[result["id"]], result, args.effort)

    judged = await asyncio.gather(*(one(r) for r in results))

    counts = {v: sum(j["verdict"] == v for j in judged) for v in (*VERDICTS, "unparsed")}
    total = len(judged)
    report: dict[str, Any] = {
        "results": args.results,
        "judge_model": args.model,
        "scenarios_judged": total,
        "counts": counts,
        "correct_pct": round(counts["correct"] / total * 100, 1) if total else 0.0,
        "correct_or_weak_pct": round((counts["correct"] + counts["weak"]) / total * 100, 1) if total else 0.0,
        "verdicts": judged,
    }
    if GOLD_PATH.exists():
        gold = {g["id"]: g["verdict"] for g in yaml.safe_load(GOLD_PATH.read_text())}
        report["vs_hand_read"] = agreement(judged, gold)
        report["disagreements"] = [
            {"id": j["id"], "judge": j["verdict"], "hand": gold[j["id"]], "why": j["why"]}
            for j in judged
            if j["id"] in gold and j["verdict"] != gold[j["id"]]
        ]

    out = Path(args.out or Path(args.results).with_suffix(".judged.json"))
    out.write_text(json.dumps(report, indent=2))

    print(f"judged {total} scenarios with {args.model}")
    print(
        f"  correct {counts['correct']}  weak {counts['weak']}  wrong {counts['wrong']}  unparsed {counts['unparsed']}"
    )
    print(f"  correct {report['correct_pct']}%  ·  correct or weak {report['correct_or_weak_pct']}%")
    if vs := report.get("vs_hand_read"):
        print(
            f"  against {vs['compared']} hand-read answers: {vs['exact_agreement']}% exact, "
            f"{vs['correct_vs_not_agreement']}% on correct-vs-not "
            f"({vs['judge_too_lenient']} too lenient, {vs['judge_too_harsh']} too harsh)"
        )
    for d in report.get("disagreements", []):
        print(f"    {d['id']}: judge {d['judge']}, hand {d['hand']} - {d['why']}")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="a run written by mechanic.evals.run")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--effort", default="high", help="reasoning_effort for the judge; judging is offline")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

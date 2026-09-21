"""What the search actually hands the agent, judged passage by passage.

`mechanic.evals.retrieval` scores the search against moderator-linked duplicates. That number is
a floor by construction: exactly one thread counts as right, so a passage that answers the
driver's question perfectly scores zero unless a moderator happened to link that particular
thread. R@5 of 42.3% does not mean the other 57.7% was junk, and nobody could say what it did
mean.

This measures the thing itself, the way retrieval has been evaluated since TREC: pool the top
results, have a judge mark each one against the question, then score the ranking with those
labels. Graded, because "mentions the same fault" and "answers it" are not the same passage:

    2  answers the question, or names the cause
    1  related and useful context, but not an answer
    0  wrong subject, or useless

Two query sets, because they answer different questions:

    --queries duplicates  the same Stack Exchange questions the floor is measured on, so the
                          two numbers can be read side by side
    --queries scenarios   what drivers actually say in evals/scenarios.yaml, each carrying its
                          car - the only set on which the vehicle boost can be measured at all

    uv run python -m mechanic.evals.retrieval_judge --model gpt-oss-20b --queries scenarios
"""

import argparse
import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from mechanic.data.common import ROOT
from mechanic.evals.retrieval import duplicate_pairs
from mechanic.knowledge.search import SearchIndex

SCENARIOS_PATH = ROOT / "evals" / "scenarios.yaml"
PASSAGE_CHARS = 600
DEPTH = 10

SYSTEM = """You are judging search results for a car mechanic's assistant. The driver asked a
question; you are shown the passages the search returned, numbered. For each one, say how useful
it would be for answering that particular question.

2 - answers the question, or names the cause or the fix
1 - related and useful background, but does not answer it
0 - different subject, or no use

Judge usefulness to this driver's question only. A passage about another car with the same fault
is still useful. A passage about the same car with an unrelated fault is not.

Reply with one line of JSON and nothing else, one score per passage, in order:
{"scores": [2, 0, 1, ...]}"""


@dataclass
class Query:
    text: str
    vehicle: str | None = None
    exclude_doc: str | None = None
    gold_doc: str | None = None


def scenario_queries() -> list[Query]:
    """Driver utterances from the scenarios that are meant to reach the knowledge base."""
    wanted = {"search_forum", "search_owner_reports", "search_how_to"}
    out = []
    for scenario in yaml.safe_load(SCENARIOS_PATH.read_text()):
        vehicle = (scenario.get("setup") or {}).get("vehicle")
        spoken: list[str] = []
        for turn in scenario["turns"]:
            spoken.append(turn["user"])
            if wanted & set(turn.get("expect_tools") or []):
                # A follow-up like "is that common on this car?" means nothing on its own, and
                # the agent does not search with it either - it has the turns before it. So the
                # query is everything the driver has said up to here.
                out.append(Query(text=" ".join(spoken), vehicle=vehicle))
    return out


def duplicate_queries(limit: int) -> list[Query]:
    pairs = duplicate_pairs()
    if limit:
        pairs = pairs[:limit]
    return [Query(text=p.query, exclude_doc=p.exclude_doc, gold_doc=p.gold_doc) for p in pairs]


def parse_scores(text: str, expected: int) -> list[int] | None:
    match = re.search(r"\{.*?\}", text, re.S)
    if match:
        try:
            scores = json.loads(match.group(0)).get("scores")
        except json.JSONDecodeError:
            scores = None
        if isinstance(scores, list) and len(scores) == expected:
            return [int(s) if str(s).strip() in "012" else 0 for s in scores]
    numbers = re.findall(r"\b[012]\b", text)
    return [int(n) for n in numbers[:expected]] if len(numbers) >= expected else None


def ndcg(gains: list[int], at: int) -> float:
    def dcg(xs):
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(xs[:at]))

    ideal = dcg(sorted(gains, reverse=True))
    return round(dcg(gains) / ideal, 3) if ideal else 0.0


def score_run(judged: list[list[int]]) -> dict[str, Any]:
    """Precision counts a passage as useful at grade 2 or better; nDCG uses the grades."""
    p5 = [sum(1 for g in gains[:5] if g >= 2) / 5 for gains in judged if gains]
    any5 = [any(g >= 2 for g in gains[:5]) for gains in judged if gains]
    return {
        "queries": len(p5),
        "precision@5": round(100 * sum(p5) / len(p5), 1) if p5 else 0.0,
        "useful_in_top5_pct": round(100 * sum(any5) / len(any5), 1) if any5 else 0.0,
        "ndcg@10": round(sum(ndcg(g, 10) for g in judged if g) / len(p5), 3) if p5 else 0.0,
    }


async def judge_query(
    client: AsyncOpenAI, model: str, query: Query, hits: list, effort: str, semaphore: asyncio.Semaphore
) -> list[int] | None:
    passages = "\n\n".join(f"[{i + 1}] {h.text[:PASSAGE_CHARS]}" for i, h in enumerate(hits))
    prompt = f"Driver's question: {query.text}\n\nPassages:\n\n{passages}"
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            extra_body={"chat_template_kwargs": {"reasoning_effort": effort}} if effort else {},
        )
    return parse_scores(response.choices[0].message.content or "", len(hits))


def retrieve(index: SearchIndex, query: Query, use: str, boost: bool, vehicle: str | None) -> list:
    hits = index.search(query.text, limit=DEPTH + 1, use=use, boost=boost, vehicle_id=vehicle)
    hits = [h for h in hits if h.doc_id != query.exclude_doc]
    return hits[:DEPTH]


async def main_async(args) -> None:
    queries = scenario_queries() if args.queries == "scenarios" else duplicate_queries(args.limit or 0)
    if args.limit and args.queries == "scenarios":
        queries = queries[: args.limit]
    index = SearchIndex()
    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    semaphore = asyncio.Semaphore(args.concurrency)

    runs: list[tuple[str, str, bool, bool]] = [
        ("both, fused (what the agent uses)", "both", True, True),
        ("BM25 only", "bm25", True, True),
        ("dense only", "dense", True, True),
    ]
    if args.queries == "scenarios":
        # The whole reason this query set exists: every query carries a car, so turning the
        # vehicle boost off is finally a comparison and not a guess.
        runs.append(("both, no vehicle boost", "both", True, False))

    print(f"{len(queries)} queries · top {DEPTH} judged by {args.model}\n")
    report: dict[str, Any] = {"queries": len(queries), "query_set": args.queries, "judge_model": args.model, "runs": {}}
    cache: dict[tuple[str, str], int] = {}

    for label, use, boost, with_vehicle in runs:
        judged: list[list[int]] = []
        pending = []
        for query in queries:
            hits = retrieve(index, query, use, boost, query.vehicle if with_vehicle else None)
            known = [cache.get((query.text, h.doc_id)) for h in hits]
            if hits and all(k is not None for k in known):
                judged.append(known)  # every passage already seen under another ranking
                continue
            pending.append((query, hits, len(judged)))
            judged.append([])

        scored = await asyncio.gather(
            *(judge_query(client, args.model, q, h, args.effort, semaphore) for q, h, _ in pending)
        )
        for (query, hits, slot), gains in zip(pending, scored, strict=True):
            if gains is None:
                continue
            judged[slot] = gains
            for hit, gain in zip(hits, gains, strict=True):
                cache[(query.text, hit.doc_id)] = gain

        result = score_run(judged)
        report["runs"][label] = result
        print(
            f"{label:36} P@5 {result['precision@5']:>5}%  "
            f"something useful in top 5 {result['useful_in_top5_pct']:>5}%  nDCG@10 {result['ndcg@10']}"
        )

    report["judged_passages"] = len(cache)
    out = Path(args.out or f"evals/results/retrieval_judged_{args.queries}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{len(cache)} passage judgements · wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--queries", choices=("duplicates", "scenarios"), default="scenarios")
    p.add_argument("--effort", default="low", help="grading one passage is a simple call")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

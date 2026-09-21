"""How good the search is, measured against judgements nobody on this project made.

The agent is only as good as what the search hands it, and until now we had three queries
checked by eye. This scores it properly, using labels that already exist and cost nothing:
when a question on mechanics.stackexchange is closed as a duplicate of another, a human has
said "these two are the same problem, the answer is over there". That is a relevance judgement.

So: the duplicate's title is the query, the thread it was closed against is the right answer,
and the duplicate's own thread is removed from the results — otherwise it retrieves itself.
Two people describing one fault in their own words is exactly the job the agent has to do.

    uv run python -m mechanic.evals.retrieval
    uv run python -m mechanic.evals.retrieval --with-body --limit 200

Two things this deliberately does not claim to measure:

- **Usefulness.** Only the one thread a moderator linked counts as right. Another thread about
  the same fault would serve the driver just as well and scores zero here, so recall is a floor,
  not a verdict on the answer. What the search handed the model for a real question is measured
  the other way round, per scenario, in `mechanic.evals.run` (see `TurnResult.blame`).
- **The vehicle boost.** These questions carry no vehicle, and only 6 of the 294 target threads
  are tagged with one of our five cars — far too few. That knob is still unmeasured.

And the queries are written, re-read text; a driver speaking has worse grammar and less detail.
"""

import argparse
import json
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from mechanic.data.common import PROCESSED_DIR, ROOT
from mechanic.knowledge.search import SearchIndex

DUMP_DIR = ROOT / "data" / "raw" / "stackexchange" / "mechanics"
DUPLICATE_LINK = "3"
# Enough context to be a real query without pasting the whole post.
BODY_CHARS = 240


@dataclass
class Pair:
    """One human judgement: asking `query`, the thread `gold_doc` is the right place to land."""

    query: str
    gold_doc: str
    exclude_doc: str


def _strip_html(text: str) -> str:
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


def duplicate_pairs(
    with_body: bool = False, links_path: Path | None = None, posts_path: Path | None = None
) -> list[Pair]:
    links = links_path or DUMP_DIR / "PostLinks.xml"
    posts = posts_path or DUMP_DIR / "Posts.xml"
    if not links.exists() or not posts.exists():
        raise SystemExit(f"Stack Exchange dump missing at {DUMP_DIR}: run `python -m mechanic.data.stackexchange`")

    duplicates: list[tuple[str, str]] = []
    for _, el in ET.iterparse(links):
        if el.tag == "row" and el.get("LinkTypeId") == DUPLICATE_LINK:
            duplicates.append((el.get("PostId"), el.get("RelatedPostId")))
        el.clear()

    wanted = {post_id for post_id, _ in duplicates}
    asked: dict[str, str] = {}
    for _, el in ET.iterparse(posts):
        if el.tag == "row" and el.get("Id") in wanted and el.get("PostTypeId") == "1":
            title = el.get("Title") or ""
            body = _strip_html(el.get("Body") or "")[:BODY_CHARS] if with_body else ""
            asked[el.get("Id")] = f"{title} {body}".strip()
        el.clear()

    indexed = {json.loads(line)["id"] for line in (PROCESSED_DIR / "stackexchange.jsonl").open(encoding="utf-8")}
    pairs = []
    for post_id, related_id in duplicates:
        gold = f"se-{related_id}"
        if gold in indexed and (query := asked.get(post_id)):
            pairs.append(Pair(query=query, gold_doc=gold, exclude_doc=f"se-{post_id}"))
    return pairs


def evaluate(index: SearchIndex, pairs: list[Pair], use: str, boost: bool, depth: int = 10) -> dict:
    """Where the right thread lands in the results, over every pair."""
    ranks: list[int | None] = []
    took_ms: list[float] = []
    for pair in pairs:
        started = time.monotonic()
        # One extra result, because one of them may be the question's own thread.
        hits = index.search(pair.query, limit=depth + 1, use=use, boost=boost)
        took_ms.append((time.monotonic() - started) * 1000)
        found = None
        rank = 0
        for hit in hits:
            if hit.doc_id == pair.exclude_doc:
                continue
            rank += 1
            if hit.doc_id == pair.gold_doc:
                found = rank
                break
        ranks.append(found)

    def recall(at: int) -> float:
        return round(100 * sum(1 for r in ranks if r is not None and r <= at) / len(ranks), 1)

    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    return {
        "queries": len(pairs),
        "recall@1": recall(1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr@10": round(mrr, 3),
        "median_ms": round(statistics.median(took_ms), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-body", action="store_true", help="query is the title plus the start of the question")
    p.add_argument("--limit", type=int, default=0, help="use only the first N pairs (for a quick look)")
    p.add_argument("--out", default="evals/results/retrieval.json")
    args = p.parse_args()

    pairs = duplicate_pairs(with_body=args.with_body)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} duplicate pairs from the dump · query = title{' + body' if args.with_body else ''}\n")

    index = SearchIndex()
    if not index.has_dense:
        print("warning: no dense vectors built, only BM25 can be measured\n")

    runs = [
        ("BM25 only", "bm25", True),
        ("dense only", "dense", True),
        ("both, fused (what the agent uses)", "both", True),
        # No vehicle is passed here, so this one isolates the small "thread was solved" bonus;
        # the vehicle boost itself cannot be measured on this set (see the module docstring).
        ("both, no solved-thread bonus", "both", False),
    ]

    results = {}
    print(f"{'':36} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'ms':>6}")
    for label, use, boost in runs:
        if use == "dense" and not index.has_dense:
            continue
        r = evaluate(index, pairs, use=use, boost=boost)
        results[label] = r
        print(
            f"{label:36} {r['recall@1']:>6} {r['recall@5']:>6} {r['recall@10']:>6} {r['mrr@10']:>6} {r['median_ms']:>6}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"queries": len(pairs), "with_body": args.with_body, "runs": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

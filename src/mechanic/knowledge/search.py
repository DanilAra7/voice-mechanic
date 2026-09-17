"""Hybrid search over the scraped knowledge base.

BM25 (bm25s) catches exact jargon — code numbers, part names, "P0171", "PCV" — while dense
vectors (bge-small via fastembed, CPU) catch paraphrases of a symptom. Results are merged with
Reciprocal Rank Fusion, then boosted when a passage matches the user's vehicle.

    uv run python -m mechanic.knowledge.search build
    uv run python -m mechanic.knowledge.search query "rough idle and high fuel trims" --vehicle audi_a4_b8
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np

from mechanic.data.common import PROCESSED_DIR, ROOT, Doc, read_jsonl
from mechanic.vehicles import VEHICLES

INDEX_DIR = ROOT / "data" / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SOURCES = ("stackexchange", "carcarekiosk", "startmycar")
MAX_CHUNK_CHARS = 1200
RRF_K = 60
# Same vehicle beats same make beats generic advice, without hiding a good generic answer.
VEHICLE_BOOST = 1.6
MAKE_BOOST = 1.2


@dataclass
class Passage:
    id: str
    doc_id: str
    source: str
    url: str
    title: str
    text: str
    vehicle_ids: list[str]
    make: str | None
    year: int | None
    score: float
    solved: bool


@dataclass
class SearchHit:
    title: str
    text: str
    url: str
    source: str
    make: str | None
    year: int | None
    solved: bool
    relevance: float


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on blank lines, packing paragraphs up to max_chars; hard-wrap very long ones."""
    chunks: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def passages_from_docs(docs: list[Doc]) -> list[Passage]:
    out = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc.text)):
            out.append(
                Passage(
                    id=f"{doc.id}#{i}",
                    doc_id=doc.id,
                    source=doc.source,
                    url=doc.url,
                    title=doc.title,
                    # The title is repeated in every chunk so later chunks stay searchable.
                    text=chunk if i == 0 else f"{doc.title}\n\n{chunk}",
                    vehicle_ids=doc.vehicle_ids,
                    make=doc.make,
                    year=doc.year,
                    score=doc.score,
                    solved=doc.solved,
                )
            )
    return out


def tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", show_progress=False)


def build(index_dir: Path = INDEX_DIR, batch_size: int = 256) -> None:
    from fastembed import TextEmbedding

    docs: list[Doc] = []
    for source in SOURCES:
        path = PROCESSED_DIR / f"{source}.jsonl"
        if path.exists():
            part = read_jsonl(path)
            docs.extend(part)
            print(f"{source}: {len(part)} docs")
        else:
            print(f"{source}: MISSING ({path})")
    passages = passages_from_docs(docs)
    print(f"{len(passages)} passages")

    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    retriever = bm25s.BM25()
    retriever.index(tokenize([p.text for p in passages]), show_progress=False)
    retriever.save(str(index_dir / "bm25"))

    started = time.monotonic()
    model = TextEmbedding(EMBED_MODEL)
    vectors = np.array(list(model.embed((p.text for p in passages), batch_size=batch_size)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    np.save(index_dir / "dense.npy", vectors)
    print(f"embedded {vectors.shape} in {time.monotonic() - started:.0f}s -> {index_dir}")


class SearchIndex:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        with (index_dir / "passages.jsonl").open(encoding="utf-8") as f:
            self.passages = [Passage(**json.loads(line)) for line in f]
        self.bm25 = bm25s.BM25.load(str(index_dir / "bm25"), load_corpus=False)
        self.dense = np.load(index_dir / "dense.npy")
        self._embedder = None

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(EMBED_MODEL)
        vec = np.array(next(iter(self._embedder.query_embed([query]))), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def search(
        self,
        query: str,
        sources: tuple[str, ...] | None = None,
        vehicle_id: str | None = None,
        limit: int = 5,
        candidates: int = 60,
    ) -> list[SearchHit]:
        keep = np.array(
            [sources is None or p.source in sources for p in self.passages],
            dtype=bool,
        )
        if not keep.any():
            return []

        sparse_ranks = self._bm25_ranks(query, keep, candidates)
        dense_ranks = self._dense_ranks(query, keep, candidates)

        make = VEHICLES[vehicle_id].make if vehicle_id else None
        fused: dict[int, float] = {}
        for ranks in (sparse_ranks, dense_ranks):
            for rank, idx in enumerate(ranks):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for idx in fused:
            p = self.passages[idx]
            if vehicle_id and vehicle_id in p.vehicle_ids:
                fused[idx] *= VEHICLE_BOOST
            elif make and p.make == make:
                fused[idx] *= MAKE_BOOST
            if p.solved:
                fused[idx] *= 1.05

        best: list[SearchHit] = []
        seen_docs: set[str] = set()
        for idx, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            p = self.passages[idx]
            if p.doc_id in seen_docs:  # one passage per document keeps the answer varied
                continue
            seen_docs.add(p.doc_id)
            best.append(
                SearchHit(
                    title=p.title,
                    text=p.text,
                    url=p.url,
                    source=p.source,
                    make=p.make,
                    year=p.year,
                    solved=p.solved,
                    relevance=round(score, 5),
                )
            )
            if len(best) >= limit:
                break
        return best

    def _bm25_ranks(self, query: str, keep: np.ndarray, candidates: int) -> list[int]:
        tokens = tokenize([query])
        scores = self.bm25.get_scores(tokens[0] if isinstance(tokens, list) else tokens.ids[0])
        scores = np.where(keep, scores, -np.inf)
        top = np.argpartition(-scores, min(candidates, len(scores) - 1))[:candidates]
        return [int(i) for i in top[np.argsort(-scores[top])] if np.isfinite(scores[i])]

    def _dense_ranks(self, query: str, keep: np.ndarray, candidates: int) -> list[int]:
        sims = self.dense @ self._embed_query(query)
        sims = np.where(keep, sims, -np.inf)
        top = np.argpartition(-sims, min(candidates, len(sims) - 1))[:candidates]
        return [int(i) for i in top[np.argsort(-sims[top])]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--vehicle", default=None, choices=list(VEHICLES))
    q.add_argument("--source", default=None, choices=SOURCES)
    q.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.command == "build":
        build()
        return
    index = SearchIndex()
    started = time.monotonic()
    hits = index.search(
        args.text,
        sources=(args.source,) if args.source else None,
        vehicle_id=args.vehicle,
        limit=args.limit,
    )
    print(f"{len(hits)} hits in {(time.monotonic() - started) * 1000:.0f} ms")
    for h in hits:
        print(f"\n[{h.source} {h.relevance}] {h.title} ({h.url})\n{h.text[:400]}")


if __name__ == "__main__":
    main()

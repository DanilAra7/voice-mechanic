"""Shared helpers for building the knowledge base: document schema, polite HTTP, text utils."""

import hashlib
import json
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
PROCESSED_DIR = ROOT / "data" / "processed"

USER_AGENT = "VoiceMechanicResearchBot/0.1 (non-commercial test project; polite crawl, 1 req/s)"


@dataclass
class Doc:
    id: str
    source: str  # "carcarekiosk" | "startmycar" | "stackexchange"
    url: str
    title: str
    text: str
    vehicle_ids: list[str] = field(default_factory=list)
    make: str | None = None
    model: str | None = None
    year: int | None = None
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    solved: bool = False


def write_jsonl(path: Path, docs: Iterable[Doc]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Doc]:
    with path.open(encoding="utf-8") as f:
        return [Doc(**json.loads(line)) for line in f if line.strip()]


class PoliteFetcher:
    """GET with an on-disk cache, a fixed delay between network requests and retries."""

    def __init__(self, delay_s: float = 1.0, cache_dir: Path = CACHE_DIR / "http"):
        self.delay_s = delay_s
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
        self._last = 0.0
        self.network_requests = 0

    def get(self, url: str) -> str | None:
        key = hashlib.sha1(url.encode()).hexdigest()
        cached = self.cache_dir / f"{key}.html"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        for attempt in range(3):
            wait = self.delay_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.network_requests += 1
            try:
                r = self._client.get(url)
            except httpx.HTTPError:
                time.sleep(2**attempt * 2)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt * 5)
                continue
            r.raise_for_status()
            cached.write_text(r.text, encoding="utf-8")
            return r.text
        return None


def html_to_text(html: str) -> str:
    tree = HTMLParser(html)
    for node in tree.css("script, style, noscript"):
        node.decompose()
    return clean_ws(tree.text(separator=" "))


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_EN_STOPWORDS = frozenset(
    "the a an and or but is are was were be been it its this that my i you he she they we to of in on for "
    "with at from by as not no when what how why car engine have has had do does did will would can".split()
)


def looks_english(text: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
    if len(words) < 3:
        return False
    if sum(ch.isascii() for ch in text) / max(1, len(text)) < 0.97:
        return False
    return sum(w in _EN_STOPWORDS for w in words) / len(words) >= 0.15

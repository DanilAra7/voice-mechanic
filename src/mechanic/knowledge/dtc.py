"""Diagnostic trouble codes, from the OBDex database (data CC0, https://github.com/foerbsnavi/OBDex).

The YAML files are parsed once into a compact JSON cache so the server starts fast.
Codes arrive from speech recognition in messy shapes ("p zero one seven one", "P 171"),
so `normalize_code` is deliberately forgiving.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mechanic.data.common import PROCESSED_DIR, RAW_DIR

OBDEX_DIR = RAW_DIR / "obdex"
CACHE_PATH = PROCESSED_DIR / "dtc.json"
FAMILIES = ("P0xxx", "P2xxx", "P3xxx", "B0xxx", "C0xxx", "U0xxx", "U3xxx")

SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}
SPOKEN_LETTERS = {"pee": "P", "bee": "B", "see": "C", "sea": "C", "you": "U", "yu": "U"}
CODE_RE = re.compile(r"^[PBCU][0-3][0-9A-F]{3}$")


@dataclass
class Dtc:
    code: str
    title: str
    description: str
    category: str = ""
    causes: list[dict] = field(default_factory=list)  # {label, likelihood}
    symptoms: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    difficulty: str = ""
    diy_possible: bool | None = None
    cost_eur: list[float] = field(default_factory=list)
    hours: list[float] = field(default_factory=list)
    related_codes: list[str] = field(default_factory=list)

    def summary(self, max_causes: int = 4, max_symptoms: int = 4) -> str:
        """Compact rendering for the LLM: what it is, why it happens, what it feels like."""
        parts = [f"{self.code} — {self.title}", self.description]
        if self.causes:
            causes = "; ".join(f"{c['label']} ({c['likelihood']} likelihood)" for c in self.causes[:max_causes])
            parts.append(f"Common causes: {causes}")
        if self.symptoms:
            parts.append(f"Symptoms: {'; '.join(self.symptoms[:max_symptoms])}")
        repair = []
        if self.difficulty:
            repair.append(f"difficulty {self.difficulty}")
        if self.diy_possible is not None:
            repair.append("DIY possible" if self.diy_possible else "shop job")
        if len(self.hours) == 2:
            repair.append(f"{self.hours[0]}-{self.hours[1]} h")
        if repair:
            parts.append("Repair: " + ", ".join(repair))
        return "\n".join(parts)


def normalize_code(raw: str) -> str | None:
    """'p zero one seven one', 'P 0171', 'code 171' -> 'P0171'. None if it isn't a code."""
    text = raw.strip().lower().replace("-", " ")
    text = re.sub(r"\b(code|dtc|error|fault)\b", " ", text)
    tokens = re.findall(r"[a-z]+|[0-9]+", text)
    out = []
    for tok in tokens:
        if tok.isdigit():
            out.append(tok)
        elif tok in SPOKEN_DIGITS:
            out.append(SPOKEN_DIGITS[tok])
        elif tok in SPOKEN_LETTERS:
            out.append(SPOKEN_LETTERS[tok])
        elif len(tok) == 1 and tok in "pbcuabcdef":
            out.append(tok.upper())
        else:
            return None
    code = "".join(out).upper()
    if code and code[0].isdigit():  # "0171" spoken without the letter: assume powertrain
        code = "P" + code
    if len(code) == 4 and code[0] in "PBCU":  # "P171" -> "P0171"
        code = code[0] + "0" + code[1:]
    return code if CODE_RE.match(code) else None


def _build_cache() -> dict[str, dict]:
    codes: dict[str, dict] = {}
    for family in FAMILIES:
        path = OBDEX_DIR / f"{family}_enriched.yaml"
        if not path.exists():
            continue
        for entry in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            repair = entry.get("repair") or {}
            codes[entry["code"]] = {
                "code": entry["code"],
                "title": (entry.get("title") or {}).get("en", ""),
                "description": (entry.get("description") or {}).get("en", ""),
                "category": entry.get("category", ""),
                "causes": [
                    {"label": (c.get("label") or {}).get("en", ""), "likelihood": c.get("likelihood", "")}
                    for c in entry.get("common_causes") or []
                ],
                "symptoms": [s.get("en", "") for s in entry.get("symptoms") or []],
                "components": entry.get("affected_components") or [],
                "difficulty": repair.get("difficulty", ""),
                "diy_possible": repair.get("diy_possible"),
                "cost_eur": repair.get("estimated_cost_eur") or [],
                "hours": repair.get("estimated_hours") or [],
                "related_codes": entry.get("related_codes") or [],
            }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(codes), encoding="utf-8")
    return codes


class DtcDatabase:
    def __init__(self, cache_path: Path = CACHE_PATH):
        if cache_path.exists():
            self._codes = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            self._codes = _build_cache()

    def __len__(self) -> int:
        return len(self._codes)

    def lookup(self, raw_code: str) -> Dtc | None:
        code = normalize_code(raw_code)
        entry = self._codes.get(code) if code else None
        return Dtc(**entry) if entry else None

    def lookup_many(self, raw_codes: list[str]) -> list[Dtc]:
        return [dtc for raw in raw_codes if (dtc := self.lookup(raw))]


def main() -> None:
    codes = _build_cache()
    print(f"dtc: {len(codes)} codes -> {CACHE_PATH}")
    print(DtcDatabase().lookup("p zero one seven one").summary())


if __name__ == "__main__":
    main()

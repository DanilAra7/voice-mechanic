"""Convert the Motor Vehicle Maintenance & Repair Stack Exchange dump into Q&A documents.

Source: https://archive.org/details/stackexchange_20251231 (mechanics.stackexchange.com.7z),
content licensed CC BY-SA (2.5/3.0/4.0 depending on post date). Attribution = link to each question.

    uv run python -m mechanic.data.stackexchange
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from mechanic.data.common import PROCESSED_DIR, RAW_DIR, Doc, html_to_text, write_jsonl
from mechanic.vehicles import VEHICLES

DUMP_DIR = RAW_DIR / "stackexchange" / "mechanics"
MAX_ANSWERS = 3
MAX_ANSWER_CHARS = 2500
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")


def parse_tags(raw: str) -> list[str]:
    # Newer dumps use "|a|b|", older ones "<a><b>".
    return [t for t in re.split(r"[|<>]", raw or "") if t]


def match_vehicles(tags: list[str], year: int | None) -> list[str]:
    tagset = set(tags)
    out = []
    for v in VEHICLES.values():
        model_tag = v.stackexchange_tags[-1]
        if model_tag in tagset and (year is None or v.covers_year(year)):
            out.append(v.id)
    return out


def build_docs() -> list[Doc]:
    questions: dict[str, dict] = {}
    answers: dict[str, list[dict]] = defaultdict(list)
    for _, e in ET.iterparse(DUMP_DIR / "Posts.xml"):
        if e.tag != "row":
            continue
        a = dict(e.attrib)
        e.clear()
        if a.get("PostTypeId") == "1":
            questions[a["Id"]] = a
        elif a.get("PostTypeId") == "2":
            answers[a["ParentId"]].append(a)

    docs = []
    for qid, q in questions.items():
        accepted_id = q.get("AcceptedAnswerId")
        good = [a for a in answers.get(qid, []) if int(a["Score"]) >= 1 or a["Id"] == accepted_id]
        if not good or int(q["Score"]) < 0:
            continue
        good.sort(key=lambda a: (a["Id"] != accepted_id, -int(a["Score"])))
        tags = parse_tags(q.get("Tags", ""))
        title = q["Title"]
        m = YEAR_RE.search(title)
        year = int(m.group(1)) if m else None
        vehicle_ids = match_vehicles(tags, year)
        make = next((v.make for v in VEHICLES.values() if v.stackexchange_tags[0] in tags), None)

        parts = [f"Question: {title}", html_to_text(q.get("Body", ""))]
        for a in good[:MAX_ANSWERS]:
            label = "Accepted answer" if a["Id"] == accepted_id else f"Answer (score {a['Score']})"
            parts.append(f"{label}: {html_to_text(a['Body'])[:MAX_ANSWER_CHARS]}")

        docs.append(
            Doc(
                id=f"se-{qid}",
                source="stackexchange",
                url=f"https://mechanics.stackexchange.com/q/{qid}",
                title=title,
                text="\n\n".join(parts),
                vehicle_ids=vehicle_ids,
                make=make,
                year=year,
                tags=tags,
                score=float(q["Score"]) + max(int(a["Score"]) for a in good),
                solved=accepted_id is not None,
            )
        )
    return docs


def main() -> None:
    docs = build_docs()
    n = write_jsonl(PROCESSED_DIR / "stackexchange.jsonl", docs)
    per_vehicle = defaultdict(int)
    for d in docs:
        for vid in d.vehicle_ids:
            per_vehicle[vid] += 1
    print(f"stackexchange: {n} Q&A threads")
    for vid in VEHICLES:
        print(f"  {vid}: {per_vehicle[vid]}")


if __name__ == "__main__":
    main()

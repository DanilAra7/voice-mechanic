"""Scrape owner-reported problems from startmycar.com for the MVP models.

Listing pages already contain title, vehicle line, tags and the report body, so a detail page
is only fetched for reports that have replies. Non-English reports are skipped.
robots.txt allows /us/<make>/<model>/problems (checked 2026-09-17).

    uv run python -m mechanic.data.startmycar
"""

import re
from collections import defaultdict

from selectolax.parser import HTMLParser, Node

from mechanic.data.common import PROCESSED_DIR, Doc, PoliteFetcher, clean_ws, looks_english, write_jsonl
from mechanic.vehicles import VEHICLES, Vehicle

BASE = "https://www.startmycar.com"
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
REPLIES_RE = re.compile(r"(\d+)\s+repl(?:y|ies)")


def parse_card(card: Node) -> dict | None:
    link = card.css_first("h3 a[href]")
    title_node = card.css_first("h3")
    body = card.css_first(".js-report-body")
    if not title_node or not body:
        return None
    vehicle_line = card.css_first(".text-ellipsis")
    vline = clean_ws(vehicle_line.text()) if vehicle_line else ""
    m = YEAR_RE.search(vline)
    replies = REPLIES_RE.search(card.text(separator=" "))
    return {
        "id": card.attributes.get("data-denunciaid"),
        "url": BASE + link.attributes["href"] if link else None,
        "title": clean_ws((link or title_node).text()),
        "vehicle_line": vline,
        "year": int(m.group(1)) if m else None,
        "tags": [clean_ws(t.text()) for t in card.css(".TagLink")],
        "body": clean_ws(body.text(separator=" ")),
        "solved": "solucionado" in (card.attributes.get("class") or ""),
        "replies": int(replies.group(1)) if replies else 0,
    }


def parse_comments(html: str) -> list[tuple[bool, str]]:
    """Return (is_best_answer, text) for each comment on a detail page."""
    out = []
    for c in HTMLParser(html).css(".CommentCard"):
        text_node = c.css_first(".CommentCard__text")
        if text_node:
            out.append(("CommentCard--solution" in (c.attributes.get("class") or ""), clean_ws(text_node.text())))
    return out


def scrape_vehicle(fetcher: PoliteFetcher, vehicle: Vehicle) -> list[Doc]:
    docs = []
    seen: set[str] = set()
    page = 1
    while True:
        url = f"{BASE}/us/{vehicle.startmycar_slug}/problems" + (f"/page{page}" if page > 1 else "")
        html = fetcher.get(url)
        cards = HTMLParser(html).css("div.js-report") if html else []
        ids = {c.attributes.get("data-denunciaid") for c in cards}
        # Stop when a page is empty or repeats reports we've already seen (past the last page).
        if not cards or ids <= seen:
            break
        seen |= ids
        for card in cards:
            item = parse_card(card)
            if not item or not item["url"] or not looks_english(f"{item['title']} {item['body']}"):
                continue
            text = [f"Problem: {item['title']}", f"Vehicle: {item['vehicle_line']}", item["body"]]
            detail_html = fetcher.get(item["url"]) if item["replies"] else None
            if detail_html:
                for best, comment in parse_comments(detail_html):
                    if looks_english(comment):
                        text.append(f"{'Best answer' if best else 'Reply'}: {comment}")
            year = item["year"]
            docs.append(
                Doc(
                    id=f"smc-{item['id']}",
                    source="startmycar",
                    url=item["url"],
                    title=item["title"],
                    text="\n\n".join(text),
                    vehicle_ids=[vehicle.id] if year and vehicle.covers_year(year) else [],
                    make=vehicle.make,
                    model=vehicle.model,
                    year=year,
                    tags=item["tags"],
                    score=float(item["replies"]),
                    solved=item["solved"],
                )
            )
        page += 1
    return docs


def main() -> None:
    fetcher = PoliteFetcher()
    all_docs: list[Doc] = []
    stats = defaultdict(lambda: [0, 0])
    for vehicle in VEHICLES.values():
        docs = scrape_vehicle(fetcher, vehicle)
        all_docs.extend(docs)
        stats[vehicle.id] = [len(docs), sum(1 for d in docs if d.vehicle_ids)]
        print(f"{vehicle.id}: {len(docs)} English reports, {stats[vehicle.id][1]} in generation years", flush=True)
    n = write_jsonl(PROCESSED_DIR / "startmycar.jsonl", all_docs)
    print(f"startmycar: {n} docs ({fetcher.network_requests} network requests)")


if __name__ == "__main__":
    main()

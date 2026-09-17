"""Scrape carcarekiosk.com how-to pages for the MVP vehicles' generations.

Only the text (title, model years, video description) is kept; the steps themselves live in
the videos. robots.txt allows /videos/ and /video/ (checked 2026-09-17).

    uv run python -m mechanic.data.carcarekiosk
"""

import re
from collections import defaultdict

from selectolax.parser import HTMLParser

from mechanic.data.common import PROCESSED_DIR, Doc, PoliteFetcher, clean_ws, write_jsonl
from mechanic.vehicles import VEHICLES, Vehicle

BASE = "https://www.carcarekiosk.com"


def video_urls(fetcher: PoliteFetcher, vehicle: Vehicle) -> list[str]:
    html = fetcher.get(f"{BASE}/videos/{vehicle.carcarekiosk_path}")
    if not html:
        return []
    urls = {
        a.attributes["href"]
        for a in HTMLParser(html).css("a[href]")
        if a.attributes["href"].startswith(f"{BASE}/video/")
    }
    return sorted(urls)


def parse_video_page(html: str, url: str, vehicle: Vehicle) -> Doc | None:
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    title = clean_ws(h1.text()) if h1 else ""
    text = clean_ws(tree.body.text(separator="\n")) if tree.body else ""
    # "Video Description" is followed by the text we want, up to the coupon/footer blocks.
    m = re.search(r"Video Description\s*(.+?)(?:Advance Auto coupon|Back to top|$)", text, flags=re.S)
    description = clean_ws(m.group(1)) if m else ""
    if not title or not description:
        return None
    # URL shape: /video/<year>_<Make>_<Model>_<engine>/<category>/<action>
    category, action = url.rstrip("/").split("/")[-2:]
    return Doc(
        id="cck-" + url.removeprefix(f"{BASE}/video/").replace("/", "-"),
        source="carcarekiosk",
        url=url,
        title=title,
        text=f"{title}\n\n{description}",
        vehicle_ids=[vehicle.id],
        make=vehicle.make,
        model=vehicle.model,
        year=vehicle.years[0],
        tags=[category.replace("_", " "), action.replace("_", " ")],
    )


def main() -> None:
    fetcher = PoliteFetcher()
    docs: list[Doc] = []
    per_vehicle = defaultdict(int)
    for vehicle in VEHICLES.values():
        urls = video_urls(fetcher, vehicle)
        print(f"{vehicle.id}: {len(urls)} video pages")
        for url in urls:
            html = fetcher.get(url)
            if html and (doc := parse_video_page(html, url, vehicle)):
                docs.append(doc)
                per_vehicle[vehicle.id] += 1
    n = write_jsonl(PROCESSED_DIR / "carcarekiosk.jsonl", docs)
    print(f"carcarekiosk: {n} docs ({fetcher.network_requests} network requests)")
    for vid, count in per_vehicle.items():
        print(f"  {vid}: {count}")


if __name__ == "__main__":
    main()

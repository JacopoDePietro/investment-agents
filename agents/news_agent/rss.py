"""Free headlines from Google News RSS feeds (no key required)."""
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus
import requests

URL = "https://news.google.com/rss/search?q={q}+when:{d}d&hl=en-US&gl=US&ceid=US:en"

def fetch_headlines(queries, hours, limit):
    days = max(1, round(hours / 24))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen, out = set(), []
    for q in queries:
        try:
            r = requests.get(URL.format(q=quote_plus(q), d=days), timeout=20,
                             headers={"User-Agent": "Mozilla/5.0 (investment-agents)"})
            r.raise_for_status()
            items = ET.fromstring(r.content).findall(".//item")
        except Exception:
            continue
        for it in items:
            title = (it.findtext("title") or "").strip()
            src = it.findtext("source") or ""
            if src and title.endswith(" - " + src):
                title = title[: -len(src) - 3]
            try:
                pub = parsedate_to_datetime(it.findtext("pubDate"))
            except Exception:
                continue
            key = title.lower()[:80]
            if pub < cutoff or key in seen:
                continue
            seen.add(key)
            out.append({"title": title, "source": src, "link": it.findtext("link") or "",
                        "published": pub.strftime("%Y-%m-%d %H:%M")})
        time.sleep(1)
    out.sort(key=lambda x: x["published"], reverse=True)
    return out[:limit]

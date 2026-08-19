#!/usr/bin/env python3
"""
news_v2.py — recent injury/withdrawal headlines for a player (Google News RSS,
no API key). Display-only context for a prediction: the model knows nothing
about a knee taped yesterday, so the human reads this before staking.
  python news_v2.py "Jannik Sinner" --days 30
"""
import argparse
import urllib.parse
import urllib.request
import ssl
try:  # python.org macOS builds ship without system CA certs; use certifi if present
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

KEYWORDS = "injury OR injured OR withdraws OR withdrawal OR retires OR retired OR illness OR pulls out"


def fetch_news(player: str, days: int = 30, limit: int = 8):
    """-> list of dicts {date, title, source, link}, newest first."""
    q = f'"{player}" ({KEYWORDS}) when:{days}d'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        xml = urllib.request.urlopen(req, timeout=10, context=_SSL).read()
    except Exception as e:  # offline / blocked: never break the prediction
        return [{"date": "", "title": f"(news lookup failed: {e})", "source": "", "link": ""}]
    out = []
    for it in ET.fromstring(xml).iter("item"):
        title = it.findtext("title") or ""
        src = it.findtext("source") or ""
        try:
            date = parsedate_to_datetime(it.findtext("pubDate")).strftime("%Y-%m-%d")
        except Exception:
            date = ""
        out.append({"date": date, "title": title.removesuffix(f" - {src}"), "source": src,
                    "link": it.findtext("link") or ""})
    return sorted(out, key=lambda d: d["date"], reverse=True)[:limit]


def format_news(player: str, items) -> str:
    head = f"\n  --- News (last 30d, injury/withdrawal keywords): {player} ---"
    if not items:
        return head + "\n  (no matching headlines)"
    return head + "\n" + "\n".join(f"  {d['date']}  {d['title']}  [{d['source']}]" for d in items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("player"); ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    print(format_news(a.player, fetch_news(a.player, a.days)))

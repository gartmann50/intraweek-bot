#!/usr/bin/env python3
"""
70-day Highs scanner.

Outputs:
  backtests/hi70_thisweek.csv
  backtests/hi70_digest.txt

Week window (default):
  Monday..Friday of the week AFTER the most recent Friday, but never in the future.

Debug overrides:
  --use-current-week           Scan Monday..today of *this* week (capped at Fri).
  --anchor-friday YYYY-MM-DD   Use a specific anchor Friday (scan the following week).

Universe strategy (fast + unbiased):
  1) Pull ALL active common stocks (no alpha bias).
  2) Pull grouped-daily for 'week_end' to get close & volume.
  3) Keep Top-N by dollar volume (close*volume), default 1000 (--topvol).
  4) Apply min price / min dollar volume / optional market-cap lower bound (only if cap present).
  5) Scan those names for 70D breakouts over the chosen week window.

Requires POLYGON_API_KEY (env or --api-key).
"""

from __future__ import annotations
import argparse, csv, os, sys, time, requests
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from collections import Counter

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")
POLY_TICKERS = "https://api.polygon.io/v3/reference/tickers"
POLY_AGGS    = "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}"
POLY_GROUPED = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d}"

# ---------- helpers ----------
def ny_today() -> date:
    return datetime.now(NY).date()

def last_friday_ny() -> date:
    d = ny_today()
    while d.weekday() != 4:  # Friday=4
        d -= timedelta(days=1)
    return d

def ymd(d: date) -> str:
    return d.isoformat()

def backoff_sleep(k: int) -> None:
    time.sleep(min(0.5 * (2**k), 6.0))

def http_get_json(url: str, params: Dict[str, str], max_tries: int = 6) -> Dict:
    last = None
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                backoff_sleep(k); continue
            r.raise_for_status()
            return r.json() or {}
        except requests.HTTPError as e:
            last = e
            if r.status_code >= 500:
                backoff_sleep(k); continue
            break
        except Exception as e:
            last = e
            backoff_sleep(k)
    raise RuntimeError(f"GET failed {url} ({last})")

# ---------- Polygon ----------
def list_common_stocks_all(api_key: str, on_date: Optional[date]) -> List[Dict]:
    """Return ALL pages of active common stocks (no alpha bias)."""
    params = {"market":"stocks","active":"true","type":"CS","limit":"1000","apiKey":api_key}
    if on_date: params["date"] = ymd(on_date)
    results: List[Dict] = []
    cursor: Optional[str] = None
    while True:
        p = dict(params)
        if cursor: p["cursor"] = cursor
        j = http_get_json(POLY_TICKERS, p)
        results.extend(j.get("results", []) or [])
        nxt = j.get("next_url") or ""
        cursor = parse_qs(urlparse(nxt).query).get("cursor", [None])[0] if nxt else None
        if not cursor: break
    return results

def fetch_grouped_daily(api_key: str, d: date) -> Dict[str, Tuple[float, float, float]]:
    """Return {symbol: (close, volume, dollar_vol)} for a single day."""
    url = POLY_GROUPED.format(d=ymd(d))
    j = http_get_json(url, {"adjusted":"true","apiKey":api_key})
    data: Dict[str, Tuple[float, float, float]] = {}
    for r in j.get("results", []) or []:
        sym = str(r.get("

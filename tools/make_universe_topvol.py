#!/usr/bin/env python3
"""
Build a dynamic equity universe using Polygon:
- Start from ALL active common stocks (no alphabet bias).
- Rank by dollar volume (close*volume) on the most recent day with data.
- Keep Top-N (--topvol, default 1000).
- Apply min price / min dollar-volume and optional market-cap floor (only when cap is present).
Outputs:
  backtests/universe_topvol.txt   (newline-separated symbols)
  backtests/universe_topvol.csv   (symbol, name, mktcap, close, volume, dollar_vol)
"""

from __future__ import annotations
import argparse, csv, os, sys, time, requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # py<3.9

NY = ZoneInfo("America/New_York")
POLY_TICKERS = "https://api.polygon.io/v3/reference/tickers"
POLY_GROUPED = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d}"

def ymd(d: date) -> str: return d.isoformat()
def ny_today() -> date:   return datetime.now(NY).date()

def http_get_json(url: str, params: Dict[str,str], tries: int = 6) -> Dict:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(min(0.5*(2**k), 6.0)); continue
            r.raise_for_status()
            return r.json() or {}
        except Exception as e:
            last = e
            time.sleep(min(0.5*(2**k), 6.0))
    raise RuntimeError(f"GET {url} failed: {last}")

def list_all_common_stocks(api_key: str) -> List[Dict]:
    params = {"market":"stocks","active":"true","type":"CS","limit":"1000","apiKey":api_key}
    results, cursor = [], None
    while True:
        p = dict(params)
        if cursor: p["cursor"] = cursor
        j = http_get_json(POLY_TICKERS, p)
        results.extend(j.get("results", []) or [])
        nxt = j.get("next_url") or ""
        cursor = parse_qs(urlparse(nxt).query).get("cursor", [None])[0] if nxt else None
        if not cursor:
            break
    return results

def grouped_daily(api_key: str, d: date) -> Dict[str, Tuple[float,float,float]]:
    url = POLY_GROUPED.format(d=ymd(d))
    j = http_get_json(url, {"adjusted":"true","apiKey":api_key})
    out: Dict[str, Tuple[float,float,float]] = {}
    for r in j.get("results", []) or []:
        t = str(r.get("T") or "").upper()
        c = float(r.get("c", 0.0) or 0.0)
        v = float(r.get("v", 0.0) or 0.0)
        if t:
            out[t] = (c, v, c*v)
    return out

def grouped_with_fallback(api_key: str, end: date, lookback: int = 7) -> Tuple[date, Dict[str, Tuple[float,float,float]]]:
    d = end
    for _ in range(lookback):
        gd = grouped_daily(api_key, d)
        if gd:
            return d, gd
        d -= timedelta(days=1)
    return end, {}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY",""))
    p.add_argument("--topvol", type=int, default=1000)
    p.add_argument("--cap-min", type=float, default=1_000_000_000.0)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=10_000_000.0)
    p.add_argument("--out-dir", default="backtests")
    return p.parse_args()

def main() -> None:
    cfg = parse_args()
    key = (cfg.api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        sys.exit("FATAL: missing POLYGON_API_KEY")

    today = ny_today()
    gd_date, gd = grouped_with_fallback(key, today, 7)
    print(f"[universe] grouped-daily rows={len(gd)} on {ymd(gd_date)} (requested {ymd(today)})")
    if not gd:
        sys.exit("[universe] no grouped-daily data available")

    tickers = list_all_common_stocks(key)
    print(f"[universe] reference tickers: {len(tickers)}")

    rows = []
    for t in tickers:
        sym = str(t.get("ticker") or "").upper().strip()
        if not sym: continue
        if (t.get("primary_exchange") or "").upper() == "OTC": continue
        if sym not in gd: continue
        close, vol, dv = gd[sym]
        if close <= 0 or vol <= 0: continue
        if close < cfg.min_price or dv < cfg.min_dollar_vol: continue
        mc = float(t.get("market_cap") or 0.0)
        if cfg.cap_min and mc < cfg.cap_min: continue  # only enforce if cap present
        name = str(t.get("name") or "").strip()
        rows.append((sym, name, mc, close, vol, dv))

    rows.sort(key=lambda x: x[5], reverse=True)  # by dollar-vol
    if cfg.topvol and len(rows) > cfg.topvol:
        rows = rows[: cfg.topvol]

    os.makedirs(cfg.out_dir, exist_ok=True)
    txt = os.path.join(cfg.out_dir, "universe_topvol.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r[0] + "\n")
    csvp = os.path.join(cfg.out_dir, "universe_topvol.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","name","market_cap","close","volume","dollar_vol"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.0f}", f"{r[3]:.2f}", f"{r[4]:.0f}", f"{r[5]:.0f}"])

    print(f"[universe] kept: {len(rows)} | wrote {txt} and {csvp}")

if __name__ == "__main__":
    main()

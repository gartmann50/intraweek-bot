#!/usr/bin/env python3
"""
Build a dynamic equity universe by *liquidity*:
  - All active US common stocks (no OTC)
  - 21-day *average* dollar volume (ADV$) using Polygon grouped-daily (adjusted)
  - Keep Top-N by ADV$ (default 1000)
  - Basic price floor on the latest trading day with data

Outputs:
  backtests/universe_topvol.txt   (newline-separated symbols)
  backtests/universe_topvol.csv   (symbol,name,market_cap,close,adv_dollar)

Notes:
  - Market cap is shown but not required (no missing-cap hacks needed)
  - ADV$ smooths out one-off spikes; much cleaner than single-day $vol
"""

from __future__ import annotations
import argparse, csv, os, sys, time, requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")
POLY_TICKERS = "https://api.polygon.io/v3/reference/tickers"
POLY_GROUPED = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d}"

def ymd(d: date) -> str: return d.isoformat()
def ny_today() -> date:   return datetime.now(NY).date()

def http_get_json(url: str, params: Dict[str, str], tries: int = 6) -> Dict:
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

def list_all_common_stocks(api_key: str) -> Dict[str, Tuple[str, float]]:
    """Return {SYM: (name, market_cap)} for all active US CS tickers."""
    params = {"market":"stocks","active":"true","type":"CS","limit":"1000","apiKey":api_key}
    out: Dict[str, Tuple[str,float]] = {}
    cursor = None
    while True:
        p = dict(params)
        if cursor: p["cursor"] = cursor
        j = http_get_json(POLY_TICKERS, p)
        for t in j.get("results", []) or []:
            sym = str(t.get("ticker") or "").upper().strip()
            if not sym: continue
            if (t.get("primary_exchange") or "").upper() == "OTC":  # exclude OTC explicitly
                continue
            name = str(t.get("name") or "").strip()
            mc   = float(t.get("market_cap") or 0.0)
            out[sym] = (name, mc)
        nxt = j.get("next_url") or ""
        cursor = urlparse(nxt).query.split("cursor=")[-1] if nxt else None
        if not cursor:
            break
    return out

def grouped_map(api_key: str, d: date) -> Dict[str, Tuple[float,float,float]]:
    """Return {SYM: (close, volume, dollar_vol)} for a single day (adjusted)."""
    j = http_get_json(POLY_GROUPED.format(d=ymd(d)), {"adjusted":"true","apiKey":api_key})
    out: Dict[str, Tuple[float,float,float]] = {}
    for r in j.get("results", []) or []:
        sym = str(r.get("T") or "").upper()
        c   = float(r.get("c", 0.0) or 0.0)
        v   = float(r.get("v", 0.0) or 0.0)
        if sym:
            out[sym] = (c, v, c*v)
    return out

def latest_day_with_data(api_key: str, end: date, lookback: int = 7) -> Tuple[date, Dict[str, Tuple[float,float,float]]]:
    d = end
    for _ in range(lookback):
        mp = grouped_map(api_key, d)
        if mp: return d, mp
        d -= timedelta(days=1)
    return end, {}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a Top-N-by-ADV$ universe.")
    p.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY",""))
    p.add_argument("--adv-days", type=int, default=21, help="Window for average dollar volume.")
    p.add_argument("--topvol",   type=int, default=1000, help="Top-N to keep by ADV$.")
    p.add_argument("--min-price", type=float, default=5.0, help="Price floor on latest trading day.")
    p.add_argument("--out-dir", default="backtests")
    return p.parse_args()

def main() -> None:
    cfg = parse_args()
    key = (cfg.api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not key: sys.exit("FATAL: missing POLYGON_API_KEY")

    today = ny_today()
    end_date, end_map = latest_day_with_data(key, today, 7)
    if not end_map:
        sys.exit("[universe] no grouped-daily data available (holiday window?)")

    # Reference metadata (name, market cap); also gives us the active CS list
    meta = list_all_common_stocks(key)
    print(f"[universe] meta tickers: {len(meta)} | end date: {ymd(end_date)} rows: {len(end_map)}")

    # Build ADV$ over adv-days using grouped maps
    adv_days = max(5, cfg.adv_days)  # guard
    adv_sum: Dict[str, float]  = {}
    adv_cnt: Dict[str, int]    = {}
    last_close: Dict[str, float] = {sym: end_map[sym][0] for sym in end_map}  # price on end_date

    d = end_date
    for i in range(adv_days):
        mp = grouped_map(key, d)
        if mp:
            for sym, (_, _, dv) in mp.items():
                # only track symbols that exist in reference meta and are not OTC
                if sym in meta:
                    adv_sum[sym] = adv_sum.get(sym, 0.0) + dv
                    adv_cnt[sym] = adv_cnt.get(sym, 0) + 1
        d -= timedelta(days=1)

    # Average
    adv: Dict[str, float] = {}
    for sym, s in adv_sum.items():
        c = adv_cnt.get(sym, 0)
        if c > 0:
            adv[sym] = s / float(c)

    # Filter by price floor on end_date (if we have it)
    price_floor = cfg.min_price
    universe = []
    for sym, a in adv.items():
        price = last_close.get(sym, 0.0)
        if price <= 0 or price < price_floor:
            continue
        name, mc = meta.get(sym, ("", 0.0))
        universe.append((sym, name, mc, price, a))

    # Rank by ADV$ desc and keep top-N
    universe.sort(key=lambda x: x[4], reverse=True)
    if cfg.topvol and len(universe) > cfg.topvol:
        universe = universe[: cfg.topvol]

    os.makedirs(cfg.out_dir, exist_ok=True)
    txt = os.path.join(cfg.out_dir, "universe_topvol.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for r in universe:
            f.write(r[0] + "\n")

    csvp = os.path.join(cfg.out_dir, "universe_topvol.csv")
    with open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","name","market_cap","close","adv_dollar"])
        for sym, name, mc, close, adv_dollar in universe:
            w.writerow([sym, name, f"{mc:.0f}", f"{close:.2f}", f"{adv_dollar:.0f}"])

    print(f"[universe] kept: {len(universe)} | wrote {txt} and {csvp}")

if __name__ == "__main__":
    main()

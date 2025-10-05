#!/usr/bin/env python3
"""
Polygon builder: Top 500 by turnover (avg dollar volume), with incremental updates.

- Writes/updates CSVs in OUT_DIR like AAPL.csv (Date,Open,High,Low,Close,Volume)
- If a CSV exists, only fetches data AFTER the last Date in the file (incremental)
- Caches the chosen 500 tickers to universe/top500_turnover.txt; reuse by default
  (pass --refresh-universe to recompute)
"""

import argparse, os, sys, time, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
import pandas as pd

POLY_BASE = "https://api.polygon.io"
UNIVERSE_CACHE = Path("universe/top500_turnover.txt")

# ---------- HTTP helpers ----------
def _sleep_backoff(i: int): time.sleep(min(2**i, 30))
def _get_json(url: str, params: Dict[str, str], retries: int = 5) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                _sleep_backoff(i); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e; _sleep_backoff(i)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")

# ---------- Universe discovery ----------
def fetch_active_common_stocks(api_key: str, limit: int = 2000) -> List[str]:
    url = f"{POLY_BASE}/v3/reference/tickers"
    params = {"market":"stocks","active":"true","limit":1000,"apiKey":api_key,"type":"CS"}
    out = []
    while True:
        data = _get_json(url, params)
        out.extend([r["ticker"] for r in data.get("results", []) if r.get("type")=="CS"])
        next_url = data.get("next_url")
        if not next_url or len(out) >= limit: break
        url = next_url; params = {} if "apiKey=" in next_url else {"apiKey":api_key}
    return out[:limit]

def select_by_turnover(api_key: str, tickers: List[str], lookback_days: int = 60, top_n: int = 500) -> List[str]:
    """Rank by avg dollar volume (Close × Volume) over recent ~60 trading days."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.tseries.offsets.BDay(int(lookback_days*1.3))  # pad for holidays
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    scores: List[Tuple[str,float]] = []
    for i, t in enumerate(tickers, 1):
        url = f"{POLY_BASE}/v2/aggs/ticker/{t}/range/1/day/{start_s}/{end_s}"
        params = {"adjusted":"true","apiKey":api_key,"limit":50000}
        try:
            data = _get_json(url, params)
            bars = data.get("results", [])
            if not bars: continue
            df = pd.DataFrame(bars)
            if not {"c","v"} <= set(df.columns): continue
            adv = float((df["c"] * df["v"]).mean())
            if math.isfinite(adv): scores.append((t, adv))
        except Exception:
            pass
        time.sleep(0.02)
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_n]]

def load_or_build_universe(api_key: str, size: int, refresh: bool) -> List[str]:
    if UNIVERSE_CACHE.exists() and not refresh:
        return [l.strip() for l in UNIVERSE_CACHE.read_text().splitlines() if l.strip() and not l.startswith("#")]
    print("Discovering active U.S. common stocks…")
    candidates = fetch_active_common_stocks(api_key, limit=2000)
    print(f"Found {len(candidates)} active. Ranking top {size} by turnover…")
    chosen = select_by_turnover(api_key, candidates, lookback_days=60, top_n=size)
    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_CACHE.write_text("\n".join(chosen))
    return chosen

# ---------- Data fetch ----------
def fetch_aggs_daily(api_key: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted":"true","apiKey":api_key,"limit":50000}
    data = _get_json(url, params)
    bars = data.get("results", [])
    if not bars: return pd.DataFrame()
    df = pd.DataFrame(bars)
    if not {"t","o","h","l","c","v"} <= set(df.columns): return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    out = pd.DataFrame({
        "Date": df["Date"],
        "Open": df["o"],
        "High": df["h"],
        "Low": df["l"],
        "Close": df["c"],
        "Volume": df["v"],
    }).dropna(subset=["Date"]).sort_values("Date")
    return out

# ---------- Incremental writer ----------
def _read_existing_dates(csv_path: Path) -> Optional[pd.Timestamp]:
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if df.empty: return None
        return pd.to_datetime(df["Date"]).max().normalize()
    except Exception:
        return None

def upsert_csv(api_key: str, ticker: str, out_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    outp = out_dir / f"{ticker}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_dt = _read_existing_dates(outp)

    # Decide window
    fetch_start = start if last_dt is None else (last_dt + pd.tseries.offsets.BDay(1))
    if fetch_start.normalize() > end.normalize():
        # Already up-to-date
        print(f"  up-to-date ({ticker})")
        return False

    df_new = fetch_aggs_daily(api_key, ticker, fetch_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df_new.empty:
        print(f"  no new bars ({ticker})")
        return False

    if outp.exists() and last_dt is not None:
        # Append + dedupe
        df_old = pd.read_csv(outp, parse_dates=["Date"])
        merged = (
            pd.concat([df_old, df_new], ignore_index=True)
              .drop_duplicates(subset=["Date"])
              .sort_values("Date")
        )
        merged.to_csv(outp, index=False)
    else:
        df_new.to_csv(outp, index=False)
    return True

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY"))
    ap.add_argument("--out-dir", default="stock_data_500")
    ap.add_argument("--start", required=True)  # YYYY-MM-DD
    ap.add_argument("--end", required=True)    # YYYY-MM-DD
    ap.add_argument("--universe-size", type=int, default=500)
    ap.add_argument("--refresh-universe", action="store_true", help="Rebuild turnover ranking instead of using cache")
    ap.add_argument("--force", action="store_true", help="Ignore existing CSV contents and redownload full history")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("ERROR: No API key provided (set POLYGON_API_KEY)")

    out_dir = Path(args.out_dir)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end <= start: sys.exit("ERROR: end must be after start")

    # Universe
    tickers = load_or_build_universe(args.api_key, args.universe_size, args.refresh_universe)
    print(f"Universe size: {len(tickers)} (cached at {UNIVERSE_CACHE})")

    # Download/update
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}")
        if args.force:
            df = fetch_aggs_daily(args.api_key, t, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if df.empty: print("  no data"); continue
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"{t}.csv", index=False)
        else:
            changed = upsert_csv(args.api_key, t, out_dir, start, end)
            if not changed:
                pass  # already up to date / no new bars
        time.sleep(0.02)

    print(f"Done. Data in {out_dir}")

if __name__ == "__main__":
    main()

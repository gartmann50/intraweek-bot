#!/usr/bin/env python3
"""
Stock-500 data builder using Polygon.io

Purpose:
  Build/refresh a directory of daily OHLCV CSVs (like AAPL.csv, MSFT.csv)
  so your backtester always has data.

Main features:
  • Pulls daily aggregates (v2/aggs) from Polygon.
  • Selects top N U.S. common stocks by average turnover (Close×Volume).
  • Skips symbols already up-to-date unless --force.
  • Writes clean CSVs with columns:
        Date,Open,High,Low,Close,Volume
  • Works locally or in GitHub Actions with POLYGON_API_KEY secret.
  • Has --self-test mode for offline CI validation.

Usage:
  export POLYGON_API_KEY=yourkey
  python tools/make_universe_polygon.py \
      --out-dir stock_data_500 \
      --start 2021-01-01 --end 2025-10-03 \
      --universe-size 500
"""

import argparse, os, sys, time, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
import pandas as pd

POLY_BASE = "https://api.polygon.io"

# ------------------------
# helpers
# ------------------------
def _sleep_backoff(i: int):
    time.sleep(min(2 ** i, 30))

def _get_json(url: str, params: Dict[str, str], retries: int = 5) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                _sleep_backoff(i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            _sleep_backoff(i)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")

# ------------------------
# universe discovery
# ------------------------
def fetch_active_common_stocks(api_key: str, limit: int = 1500) -> List[str]:
    url = f"{POLY_BASE}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "apiKey": api_key,
        "type": "CS"
    }
    out = []
    while True:
        data = _get_json(url, params)
        results = data.get("results", [])
        out.extend([r["ticker"] for r in results if r.get("type") == "CS"])
        next_url = data.get("next_url")
        if not next_url or len(out) >= limit:
            break
        url = next_url
        params = {} if "apiKey=" in next_url else {"apiKey": api_key}
    return out[:limit]

def select_by_turnover(api_key: str, tickers: List[str], lookback_days: int = 60, top_n: int = 500) -> List[str]:
    """Rank by avg dollar volume (Close × Volume)."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.tseries.offsets.BDay(lookback_days * 1.3)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    scores = []
    for t in tickers:
        url = f"{POLY_BASE}/v2/aggs/ticker/{t}/range/1/day/{start_s}/{end_s}"
        params = {"adjusted": "true", "apiKey": api_key, "limit": 50000}
        try:
            data = _get_json(url, params)
            bars = data.get("results", [])
            if not bars:
                continue
            df = pd.DataFrame(bars)
            adv = float((df["c"] * df["v"]).mean())
            if math.isfinite(adv):
                scores.append((t, adv))
        except Exception:
            continue
        time.sleep(0.02)
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_n]]

# ------------------------
# download aggregates
# ------------------------
def fetch_aggs_daily(api_key: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "apiKey": api_key, "limit": 50000}
    data = _get_json(url, params)
    bars = data.get("results", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    return pd.DataFrame({
        "Date": df["Date"],
        "Open": df["o"],
        "High": df["h"],
        "Low": df["l"],
        "Close": df["c"],
        "Volume": df["v"],
    }).sort_values("Date")

# ------------------------
# main builder
# ------------------------
def build_universe(api_key: str, out_dir: Path, start: str, end: str,
                   universe_size: int = 500, force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Discovering active U.S. common stocks…")
    tickers = fetch_active_common_stocks(api_key, limit=2000)
    print(f"Found {len(tickers)} active. Ranking top {universe_size} by turnover…")
    top = select_by_turnover(api_key, tickers, lookback_days=60, top_n=universe_size)
    print(f"Selected {len(top)} symbols.")

    for i, t in enumerate(top, 1):
        path = out_dir / f"{t}.csv"
        if path.exists() and not force:
            try:
                last = pd.read_csv(path)["Date"].max()
                if last >= end:
                    print(f"[{i}/{len(top)}] {t}: up-to-date")
                    continue
            except Exception:
                pass
        print(f"[{i}/{len(top)}] {t}: downloading {start}→{end}")
        try:
            df = fetch_aggs_daily(api_key, t, start, end)
            if df.empty:
                print(f"  no data for {t}")
                continue
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"  error {t}: {e}")
        time.sleep(0.02)

    print(f"Done. Wrote CSVs to {out_dir}")

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY"))
    ap.add_argument("--out-dir", default="stock_data_500")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--universe-size", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        p = Path(args.out_dir)
        p.mkdir(exist_ok=True)
        for s in ["AAA","BBB","CCC"]:
            d = pd.date_range("2024-01-01", periods=10)
            df = pd.DataFrame({
                "Date": d,
                "Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1_000_000
            })
            df.to_csv(p / f"{s}.csv", index=False)
        print(f"Self-test CSVs written to {p}")
        return

    if not args.api_key:
        sys.exit("ERROR: No API key provided (set POLYGON_API_KEY)")

    build_universe(
        api_key=args.api_key,
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
        universe_size=args.universe_size,
        force=args.force,
    )

if __name__ == "__main__":
    main()

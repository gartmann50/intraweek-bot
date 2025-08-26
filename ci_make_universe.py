#!/usr/bin/env python3
"""
ci_make_universe.py
Builds a dynamic universe of tickers for CI.

Order of sources (auto):
1) If 'universe/us_large_400.txt' exists and is non-empty -> use it.
2) Else, if CSVs exist in stock_data_400/*.csv -> use those basenames.
3) Else, fetch a fresh list from Polygon v3/reference/tickers (type=CS).

Writes one ticker per line to --out-file (default: universe/dynamic_universe.txt).
"""

import argparse, os, sys, time
from pathlib import Path
import requests

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-file", default="universe/dynamic_universe.txt")
    p.add_argument("--fallback-file", default="universe/us_large_400.txt")
    p.add_argument("--csv-folder", default="stock_data_400")
    p.add_argument("--source", default="auto", choices=["auto","file","folder","polygon"])
    p.add_argument("--max", type=int, default=700, help="Cap number of tickers (for CI speed)")
    p.add_argument("--api-key", default=os.environ.get("POLYGON_API_KEY"))
    p.add_argument("--qps-delay", type=float, default=0.25)
    return p.parse_args()

def read_file_list(path: Path):
    if path.exists():
        tickers = [t.strip().upper() for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
        return [t for t in tickers if t]
    return []

def read_folder_csvs(folder: Path):
    if not folder.exists():
        return []
    tickers = []
    for p in sorted(folder.glob("*.csv")):
        # e.g., AAPL.csv -> AAPL
        tickers.append(p.stem.upper())
    return tickers

def fetch_polygon_universe(api_key: str, max_count: int, qps_delay: float):
    if not api_key:
        print("WARN: POLYGON_API_KEY not set; polygon source unavailable.", file=sys.stderr)
        return []
    # Reference endpoint (US common stocks)
    base = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "type": "CS",       # common stock (avoid ETFs/ETNs)
        "active": "true",
        "locale": "us",
        "limit": 1000,
        "order": "asc",
        "apiKey": api_key,
    }
    out, cursor = [], None
    while True:
        if cursor:
            params["cursor"] = cursor
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        results = js.get("results", []) or []
        for row in results:
            t = (row.get("ticker") or "").strip().upper()
            # light filters: skip odd formats
            if not t or t.startswith("$") or t.endswith("W"):  # skip some warrants
                continue
            out.append(t)
            if len(out) >= max_count:
                return out
        cursor = js.get("next_url") or js.get("next") or js.get("cursor")
        if not cursor:
            break
        if qps_delay:
            time.sleep(qps_delay)
    return out

def write_universe(tickers, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    unique = []
    seen = set()
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    out_file.write_text("\n".join(unique) + "\n", encoding="utf-8")
    return len(unique)

def main():
    a = parse_args()

    # Decide source
    source = a.source
    tickers = []
    if source in ("auto","file"):
        tickers = read_file_list(Path(a.fallback_file))
    if source == "auto" and not tickers:
        tickers = read_folder_csvs(Path(a.csv_folder))
    if (source in ("auto","polygon") and not tickers) or (source == "polygon"):
        got = fetch_polygon_universe(a.api_key, a.max, a.qps_delay)
        if got:
            tickers = got

    if not tickers:
        print("ERROR: Could not build universe from any source.", file=sys.stderr)
        return 2

    n = write_universe(tickers[:a.max], Path(a.out_file))
    print(f"Universe written: {n} tickers -> {a.out_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

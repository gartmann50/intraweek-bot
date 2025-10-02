#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan for 70-trading-day breakout highs for the *current week* (Mon..Fri after last Friday),
optionally filter to market cap >= threshold, and export the top results.

- Uses Polygon.io (daily bars + reference tickers)
- Relies on env POLYGON_API_KEY
- Minimal, robust, and noisy (DEBUG) so we can verify logic quickly

Outputs:
  backtests/hi70_thisweek.csv
  backtests/hi70_digest.txt
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timedelta, date, timezone
from typing import Dict, Any, List, Tuple

import requests
import pandas as pd


# -----------------------------
# Helpers: dates / week window
# -----------------------------

def last_friday_ny(today_utc: datetime | None = None) -> date:
    """Return last Friday in New York (calendar date)."""
    # We’ll keep it simple: use UTC 'today', compute last Friday by weekday math.
    # (GitHub runner uses UTC; this is fine for weekly window construction.)
    if today_utc is None:
        today_utc = datetime.now(timezone.utc)
    d = today_utc.date()
    # If today is Sat (5) or Sun (6) we still want the most recent Fri (4).
    while d.weekday() != 4:  # 0=Mon .. 4=Fri
        d -= timedelta(days=1)
    return d


def week_window_from_friday(friday: date) -> Tuple[date, date]:
    """Return (Mon..Fri) week window that *ends* at given Friday."""
    start = friday - timedelta(days=4)
    end = friday
    return (start, end)


# -----------------------------
# HTTP + Polygon helpers
# -----------------------------

class Http:
    def __init__(self, api_key: str, base: str = "https://api.polygon.io"):
        self.s = requests.Session()
        self.base = base.rstrip("/")
        self.key = api_key

    def get(self, path: str, params: Dict[str, Any] | None = None, max_tries: int = 6, sleep0: float = 0.5) -> Dict[str, Any]:
        if params is None:
            params = {}
        params = dict(params)
        params["apiKey"] = self.key
        url = f"{self.base}{path}"
        for k in range(max_tries):
            r = self.s.get(url, params=params, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json() or {}
                except Exception:
                    return {}
            # 429 backoff
            if r.status_code == 429:
                time.sleep(sleep0 * (2 ** k))
                continue
            # Some 4xx we just raise
            msg = f"HTTP {r.status_code} on {url} | body={r.text[:500]}"
            raise RuntimeError(msg)
        raise RuntimeError(f"HTTP 429 backoff exceeded on {url}")

    def all_cs_tickers(self, pages: int = 3, asof: date | None = None) -> List[Dict[str, Any]]:
    """
    Fetch up to 'pages' * 1000 common stocks (type=CS, active=true) as of a given date,
    so that 'market_cap' is populated.
    """
    out: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(pages):
        params = {
            "market": "stocks",
            "type": "CS",
            "active": "true",
            "limit": 1000,
        }
        if asof:
            params["date"] = asof.isoformat()   # <-- THIS unlocks market_cap
        if cursor:
            params["cursor"] = cursor
        j = self.get("/v3/reference/tickers", params)
        out.extend(j.get("results") or [])
        cursor = j.get("next_url_params", {}).get("cursor")
        if not cursor:
            break
    return out


    def agg_day(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        """Daily aggregates for a symbol in [start, end] inclusive."""
        # v2 aggs; ascending; big limit just in case
        path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        j = self.get(path, params=params)
        return j.get("results") or []


# -----------------------------
# Core scan
# -----------------------------

def build_bars_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows)
    # ensure columns exist
    for c in ["t", "o", "h", "l", "c", "v"]:
        if c not in df.columns:
            df[c] = pd.NA
    # index on UTC datetime; keep tz-naive for comparisons
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index(idx).sort_index()
    return df[["o", "h", "l", "c", "v"]].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan 70D highs for current week (Mon..Fri ending last Friday).")
    ap.add_argument("--cap-min", type=float, default=1_000_000_000.0, help="Market cap floor in USD (default: 1e9).")
    ap.add_argument("--top", type=int, default=10, help="How many results to output (default: 10).")
    ap.add_argument("--pages", type=int, default=3, help="How many 1000-sym pages to query from Polygon (default: 3).")
    ap.add_argument("--since-days", type=int, default=120, help="Extra history before week_end (default: 120).")
    ap.add_argument("--out-dir", type=str, default="backtests", help="Output directory (default: backtests).")
    ap.add_argument("--friday", type=str, default="", help="Override last Friday (YYYY-MM-DD).")
    ap.add_argument("--use-close", action="store_true", help="Use CLOSE breakout instead of HIGH (default: HIGH).")
    args = ap.parse_args()

    api_key = os.getenv("POLYGON_API_KEY") or os.getenv("POLY_KEY") or ""
    if not api_key:
        print("FATAL: POLYGON_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    # Compute week window
    if args.friday:
        friday = date.fromisoformat(args.friday)
    else:
        friday = last_friday_ny()
    week_start, week_end = week_window_from_friday(friday)

    # For rolling 70 we need at least 70 + since_days before week_end
    bars_from = week_end - timedelta(days=args.since_days + 90)

    print(f"Week window: {week_start} .. {week_end} | bars from: {bars_from} .. {week_end}")

    http = Http(api_key)
    # 1) Fetch tickers (active common stocks)
    tickers = http.all_cs_tickers(pages=args.pages)
    print(f"Fetched {len(tickers)} active common stocks from Polygon (pages={args.pages}).")

    # Build symbol -> market_cap map (may be None)
    cap_map: Dict[str, float] = {}
    for t in tickers:
        sym = (t.get("ticker") or "").upper()
        mc = t.get("market_cap")
        try:
            cap_map[sym] = float(mc) if mc is not None else float("nan")
        except Exception:
            cap_map[sym] = float("nan")

    syms = [ (t.get("ticker") or "").upper() for t in tickers if t.get("ticker") ]
    total = len(syms)
    print(f"Reference rows with non-null market_cap: {sum(pd.notna(pd.Series(list(cap_map.values()))))}/{total}")

    # -----------------------------
    # DEBUG counters
    # -----------------------------
    DEBUG = True
    DEBUG_EVERY = 200
    dbg_total = 0
    dbg_okbars = 0
    dbg_any_breakout_high = 0
    dbg_any_breakout_close = 0
    dbg_near: List[Tuple[float, str, date, float, float, float]] = []

    def near_report():
        if not dbg_near:
            print("[hi70][dbg] no near-high observations collected")
            return
        top = sorted(dbg_near, key=lambda x: x[0])[:10]
        print("\n[hi70][dbg] Top-10 nearest to prior-70D HIGH (gap %):")
        for gap, s, d, lc, lh, p70h in top:
            print(f"  {s:6s} gap={gap:6.2f}%  last={d}  close={lc:.2f}  high={lh:.2f}  prior70H={p70h:.2f}")
        print()

    # -----------------------------
    # Scan all symbols
    # -----------------------------
    found: List[Dict[str, Any]] = []

    for i, sym in enumerate(syms, start=1):
        dbg_total += 1

        try:
            rows = http.agg_day(sym, bars_from, week_end)
        except Exception as e:
            if i % DEBUG_EVERY == 0:
                print(f"[{i}/{total}] {sym}: agg fetch error: {e}")
            continue

        df = build_bars_df(rows)
        if df.empty or len(df) < 80:
            if i % DEBUG_EVERY == 0:
                print(f"[{i}/{total}] {sym}: too few bars ({len(df)})")
            continue

        dbg_okbars += 1

        highs = df["h"]
        closes = df["c"]
        prior70_high = highs.rolling(70, min_periods=70).max().shift(1)
        prior70_close = closes.rolling(70, min_periods=70).max().shift(1)

        # Any breakout anywhere in history (DEBUG only)
        brk_high_any = (highs >= prior70_high).fillna(False).any()
        brk_close_any = (closes >= prior70_close).fillna(False).any()
        if brk_high_any:
            dbg_any_breakout_high += 1
        if brk_close_any:
            dbg_any_breakout_close += 1

        # Collect how near the *last* bar is to prior70_high
        if not math.isnan(prior70_high.iloc[-1]):
            gap = max(0.0, (prior70_high.iloc[-1] - highs.iloc[-1]) / prior70_high.iloc[-1] * 100.0)
            dbg_near.append((gap, sym, df.index[-1].date(), float(closes.iloc[-1]), float(highs.iloc[-1]), float(prior70_high.iloc[-1])))

        # Breakout logic for THIS WEEK
        wk_mask = (df.index.date >= week_start) & (df.index.date <= week_end)
        if not wk_mask.any():
            # No bars this week (holiday/delist) => skip
            if i % DEBUG_EVERY == 0:
                print(f"[{i}/{total}] {sym}: no bars in week window")
            continue

        if args.use_close:
            hit = ((closes >= prior70_close) & wk_mask).fillna(False)
        else:
            hit = ((highs >= prior70_high) & wk_mask).fillna(False)

        if hit.any():
            first_hit_idx = df.index[hit].min()
            mc = cap_map.get(sym)
            # Apply cap-min at the end (unknown caps treated as 0)
            cap_val = 0.0 if (mc is None or math.isnan(mc)) else float(mc)
            if cap_val >= args.cap_min:
                found.append({
                    "symbol": sym,
                    "first_hit": first_hit_idx.date().isoformat(),
                    "last_close": float(closes.iloc[-1]),
                    "last_high": float(highs.iloc[-1]),
                    "prior70_high": float(prior70_high.iloc[-1]) if not math.isnan(prior70_high.iloc[-1]) else float("nan"),
                    "market_cap": cap_val,
                })

        # Progress
        if i % 200 == 0 or i == total:
            print(f"  – scanned {i}/{total}; candidates so far = {len(found)}")

    # -----------------------------
    # DEBUG summary
    # -----------------------------
    print(f"\n[hi70][dbg] SUMMARY: scanned={dbg_total} | okbars={dbg_okbars} | "
          f"any_breakout(high)={dbg_any_breakout_high} | any_breakout(close)={dbg_any_breakout_close}")
    near_report()

    # -----------------------------
    # Output
    # -----------------------------
    # Sort by market cap desc, take top K
    df_found = pd.DataFrame(found)
    if df_found.empty:
        print("Candidates that made a 70D high this week: 0")
    else:
        df_found = df_found.sort_values(["market_cap", "symbol"], ascending=[False, True])
        df_out = df_found.head(int(args.top)).copy()
        out_csv = os.path.join(args.out_dir, "hi70_thisweek.csv")
        df_out.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv}  (rows={len(df_out)})")

    # Digest
    digest = [
        f"Week {week_start}..{week_end}",
        f"Scanned symbols: {total}",
        f"Eligible (>=80 bars): {dbg_okbars}",
        f"Any breakout ever: HIGH={dbg_any_breakout_high} CLOSE={dbg_any_breakout_close}",
        f"Candidates this week (cap >= {int(args.cap_min):,}): {0 if df_found.empty else len(df_found)}",
        f"Top exported: {0 if df_found.empty else min(args.top, len(df_found))}",
    ]
    out_txt = os.path.join(args.out_dir, "hi70_digest.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(digest) + "\n")
    print(f"Wrote: {out_txt}")

    # Return non-zero only on fatal misconfig, not “no results”
    sys.exit(0)


if __name__ == "__main__":
    main()

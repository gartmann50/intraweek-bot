#!/usr/bin/env python3
"""
Scan for 70-day highs (no market-cap filter).

- Universe: active U.S. common stocks (Polygon v3/reference/tickers, type=CS).
- For each symbol, compute prior-70D high (rolling 70-high of "high", shifted by 1).
- Check within the most recent full trading week (Mon..Fri) whether
  any bar's high or close >= prior-70D high (two breakout flavors).
- Rank "nearest to prior-70D high" using the latest bar in week (gap %).
- Save:
    backtests/hi70_thisweek.csv : all in-week breakouts with metadata
    backtests/hi70_digest.txt   : summary + Top-10 nearest table

Env/Args:
- POLYGON_API_KEY environment variable is used by default.
- You can pass --api-key to override, --pages for reference paging,
  and --min-price / --min-dollar-vol for simple liquidity filters.

This file intentionally avoids market_cap to keep it fast and reliable.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from zoneinfo import ZoneInfo


# ---------- Utilities ----------

def log(msg: str) -> None:
    print(msg, flush=True)


def retry_get(url: str,
              params: Optional[Dict] = None,
              timeout: int = 30,
              retries: int = 6) -> requests.Response:
    """
    GET with exponential backoff. Honors Polygon 429s / transient 5xx.
    If `url` already contains query parameters (next_url), pass `params=None`.
    """
    for k in range(retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
            return r
        sleep_s = min(0.5 * (2 ** k), 8.0)
        time.sleep(sleep_s)
    # last attempt
    r.raise_for_status()
    return r


# ---------- Polygon helpers (no market-cap) ----------

def list_common_stocks(api_key: str, pages: int = 3) -> pd.DataFrame:
    """
    Return up to pages*1000 active US common stocks (type=CS).
    Only the ticker/symbol is required here.
    """
    base = "https://api.polygon.io/v3/reference/tickers"
    out = []
    next_url = None
    for _ in range(pages):
        params = {
            "market": "stocks",
            "type": "CS",
            "active": "true",
            "limit": 1000,
            "sort": "ticker",
            "apiKey": api_key,
        }
        url = next_url or base
        r = retry_get(url, params=None if next_url else params, timeout=30)
        j = r.json() or {}
        for row in j.get("results") or []:
            sym = (row.get("ticker") or "").upper().strip()
            if sym:
                out.append({"symbol": sym})
        next_url = j.get("next_url")
        if not next_url:
            break
    return pd.DataFrame(out).drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def fetch_daily_bars(sym: str, start: dt.date, end: dt.date, api_key: str) -> pd.DataFrame:
    """
    Polygon aggregates (1/day) for [start..end] inclusive.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    params = {
        "adjusted": "true",
        "limit": 50000,
        "sort": "asc",
        "apiKey": api_key,
    }
    r = retry_get(url, params=params, timeout=30)
    j = r.json() or {}
    rows = j.get("results") or []

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    # polygon uses ms epoch in 't'
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


# ---------- Date helpers ----------

def last_completed_week(ny_today: dt.date) -> Tuple[dt.date, dt.date]:
    """
    Return Monday..Friday of the most recent *completed* trading week in New York.
    If today is Mon..Fri, we still use that week's Mon..Fri (so scanning is "this week" in-progress).
    If weekend, we return the Mon..Fri of the week that ended Fri.
    """
    # Use NY clock for week semantics
    if ny_today.weekday() >= 5:   # Sat(5) or Sun(6)
        # go back to last Friday
        d = ny_today - dt.timedelta(days=ny_today.weekday() - 4 if ny_today.weekday() > 4 else 0)
    else:
        d = ny_today
    monday = d - dt.timedelta(days=d.weekday())
    friday = monday + dt.timedelta(days=4)
    return monday, friday


# ---------- Scan core ----------

@dataclasses.dataclass
class ScanArgs:
    api_key: str
    pages: int
    since_days: int
    top: int
    out_dir: str
    min_price: float
    min_dollar_vol: float


def compute_prior70h(df: pd.DataFrame) -> pd.Series:
    """rolling 70-high of 'high', shifted by 1 (prior!)."""
    s = df["high"].rolling(window=70, min_periods=70).max().shift(1)
    return s


def nearest_gap_pct(prior: float, value: float) -> float:
    """gap % to prior high; clipped at >= 0."""
    if not (prior and value and prior > 0):
        return math.inf
    gap = (prior - value) / prior * 100.0
    return max(0.0, gap)


def main() -> None:
    p = argparse.ArgumentParser(description="Scan 70D highs (no market-cap).")
    p.add_argument("--api-key", type=str, default=os.getenv("POLYGON_API_KEY", ""),
                   help="Polygon API key (defaults to POLYGON_API_KEY env)")
    p.add_argument("--pages", type=int, default=3, help="Ticker pages (1000 per page).")
    p.add_argument("--since-days", type=int, default=130,
                   help="How many calendar days before week_start to start fetching bars "
                        "(should be >= 100 to safely compute prior-70D high). Default 130.")
    p.add_argument("--top", type=int, default=10, help="Top-N nearest to prior70H to display.")
    p.add_argument("--out-dir", type=str, default="backtests", help="Output folder.")
    p.add_argument("--min-price", type=float, default=0.0, help="Skip last close < this price.")
    p.add_argument("--min-dollar-vol", type=float, default=0.0,
                   help="Skip if ADV20 (close*volume) < this (USD).")
    args = p.parse_args()

    if not args.api_key:
        log("FATAL: Missing Polygon API key (set POLYGON_API_KEY or pass --api-key).")
        sys.exit(1)

    cfg = ScanArgs(
        api_key=args.api_key,
        pages=args.pages,
        since_days=args.since_days,
        top=args.top,
        out_dir=args.out_dir,
        min_price=args.min_price,
        min_dollar_vol=args.min_dollar_vol,
    )

    # Determine NY week window and bar range
    ny_now = dt.datetime.now(ZoneInfo("America/New_York")).date()
    week_start, week_end = last_completed_week(ny_now)

    # start bars earlier so rolling(70) prior exists
    bars_start = week_start - dt.timedelta(days=cfg.since_days)
    bars_end = week_end

    log(f"Week window: {week_start.isoformat()} .. {week_end.isoformat()} "
        f"| bars from: {bars_start.isoformat()} .. {bars_end.isoformat()}")

    # Universe
    ref = list_common_stocks(cfg.api_key, pages=cfg.pages)
    log(f"Fetched {len(ref)} active common stocks from Polygon (pages={cfg.pages}).")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Scan
    scanned = 0
    okbars = 0
    any_b_high = 0
    any_b_close = 0
    candidates_rows: List[Dict] = []
    nearest_rows: List[Tuple[str, float, Dict]] = []  # (symbol, gap%, row dict)

    N = len(ref)
    chunk = max(200, min(200, N))  # print every ~200

    for i, row in ref.iterrows():
        sym = row["symbol"]
        scanned += 1

        try:
            df = fetch_daily_bars(sym, bars_start, bars_end, cfg.api_key)
            if df.empty or len(df) < 75:
                # not enough bars for prior-70H
                if scanned % chunk == 0:
                    log(f"  – scanned {scanned}/{N}; candidates so far = {len(candidates_rows)}")
                continue

            okbars += 1

            # basic liquidity screens
            last_close = float(df.iloc[-1]["close"])
            if cfg.min_price and last_close < cfg.min_price:
                if scanned % chunk == 0:
                    log(f"  – scanned {scanned}/{N}; candidates so far = {len(candidates_rows)}")
                continue

            if cfg.min_dollar_vol:
                adv20 = float((df.tail(20)["close"] * df.tail(20)["volume"]).mean())
                if adv20 < cfg.min_dollar_vol:
                    if scanned % chunk == 0:
                        log(f"  – scanned {scanned}/{N}; candidates so far = {len(candidates_rows)}")
                    continue

            df["prior70H"] = compute_prior70h(df)

            # keep just week rows
            wmask = (df["date"] >= week_start) & (df["date"] <= week_end)
            wdf = df.loc[wmask].copy()
            if wdf.empty:
                if scanned % chunk == 0:
                    log(f"  – scanned {scanned}/{N}; candidates so far = {len(candidates_rows)}")
                continue

            # in-week breakouts
            wdf["b_high"] = (wdf["high"] >= wdf["prior70H"])
            wdf["b_close"] = (wdf["close"] >= wdf["prior70H"])

            broke_high = bool(wdf["b_high"].fillna(False).any())
            broke_close = bool(wdf["b_close"].fillna(False).any())
            if broke_high:
                any_b_high += 1
            if broke_close:
                any_b_close += 1

            if broke_high or broke_close:
                # record all in-week breakout rows with some context
                for _, r in wdf.loc[(wdf["b_high"] | wdf["b_close"]).fillna(False)].iterrows():
                    candidates_rows.append({
                        "symbol": sym,
                        "date": r["date"],
                        "close": float(r["close"]),
                        "high": float(r["high"]),
                        "prior70H": float(r["prior70H"]) if pd.notna(r["prior70H"]) else None,
                        "break_high": bool(r["b_high"]),
                        "break_close": bool(r["b_close"]),
                    })

            # “nearest” uses the **latest** bar in week for this symbol
            last_w = wdf.dropna(subset=["prior70H"]).iloc[-1:]  # tail(1) but ensures prior exists
            if not last_w.empty:
                prior = float(last_w["prior70H"].iloc[0])
                gap = nearest_gap_pct(prior, float(last_w["close"].iloc[0]))
                # keep small gap (near prior) even if no breakout, for ranking table
                nearest_rows.append((
                    sym,
                    gap,
                    {
                        "symbol": sym,
                        "gap_pct": gap,
                        "date": last_w["date"].iloc[0],
                        "close": float(last_w["close"].iloc[0]),
                        "high": float(last_w["high"].iloc[0]),
                        "prior70H": prior,
                    }
                ))

        except Exception as e:
            # Keep going if a single symbol fails
            if scanned % chunk == 0:
                log(f"  – scanned {scanned}/{N}; candidates so far = {len(candidates_rows)}")
            continue

        if scanned % chunk == 0:
            log(f"  – scanned {scanned}/{N}; candidates so far = {len(candidates_rows)}")

    # ---------- Output ----------

    # candidates CSV
    cand_df = pd.DataFrame(candidates_rows)
    cand_df = cand_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    out_csv = os.path.join(cfg.out_dir, "hi70_thisweek.csv")
    cand_df.to_csv(out_csv, index=False)

    # top nearest
    nearest_rows = [x for x in nearest_rows if math.isfinite(x[1])]
    nearest_rows.sort(key=lambda x: (x[1], x[0]))  # by gap %, then symbol
    topK = nearest_rows[: cfg.top]

    # human digest
    digest_lines: List[str] = []
    digest_lines.append(f"[hi70][dbg] SUMMARY: scanned={scanned} | okbars={okbars} | "
                        f"any_breakout(high)={any_b_high} | any_breakout(close)={any_b_close}")
    digest_lines.append(f"[hi70][dbg] Top-{cfg.top} nearest to prior-70D HIGH (gap %):")
    for sym, gap, info in topK:
        digest_lines.append(
            f"  {sym:<6}  gap= {gap:>6.2f}%  "
            f"date={info['date']}  close={info['close']:.2f}  "
            f"high={info['high']:.2f}  prior70H={info['prior70H']:.2f}"
        )

    out_txt = os.path.join(cfg.out_dir, "hi70_digest.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(digest_lines) + "\n")

    # log tail
    for ln in digest_lines[:2]:
        log(ln)
    for ln in digest_lines[2: 2 + cfg.top]:
        log(ln)

    log(f"Wrote: {out_txt}")
    log(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

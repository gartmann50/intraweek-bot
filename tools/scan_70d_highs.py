#!/usr/bin/env python3
import os, sys, time, json, math
import argparse
import datetime as dt
from typing import Optional, Dict, List, Tuple

import requests
import pandas as pd
import numpy as np


# ---------- utilities ----------
def last_friday_europe_oslo() -> dt.date:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Oslo")
    d = dt.datetime.now(tz).date()
    while d.weekday() != 4:
        d -= dt.timedelta(days=1)
    return d

def http_get_json(url: str, params: Dict[str, str], retries: int = 5, sleep: float = 0.5) -> dict:
    """GET with exponential backoff; raises on final failure."""
    for k in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 429:
            if r.status_code >= 400:
                try:
                    j = r.json()
                except Exception:
                    j = {"text": r.text[:500]}
                if k == retries - 1:
                    raise RuntimeError(f"GET {r.url} failed: {r.status_code} {j}")
            else:
                return r.json() or {}
        time.sleep(sleep * (2 ** k))
    raise RuntimeError(f"GET {url} exhausted retries")

# ---------- listing with correct pagination (cursor) ----------
def list_common_stocks(api_key: str, pages: int, max_syms: int, asof_date: Optional[str]) -> List[dict]:
    base = "https://api.polygon.io/v3/reference/tickers"
    out: List[dict] = []
    cursor: Optional[str] = None

    for _ in range(pages):
        if cursor:
            params = {"cursor": cursor, "apiKey": api_key}
        else:
            params = {
                "market": "stocks",
                "type": "CS",
                "active": "true",
                "order": "asc",
                "sort": "ticker",  # safe sort key
                "limit": "1000",
                "apiKey": api_key,
            }
            if asof_date:
                params["date"] = asof_date

        j = http_get_json(base, params)
        results = j.get("results") or []
        if not isinstance(results, list) or not results:
            break
        out.extend(results)
        if len(out) >= max_syms:
            break

        nxt = j.get("next_url")
        if not nxt:
            break
        if "cursor=" in nxt:
            cursor = nxt.split("cursor=", 1)[1]
        else:
            break

    if len(out) > max_syms:
        out = out[:max_syms]
    return out

# ---------- daily bars ----------
def fetch_daily_bars(api_key: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": api_key}
    j = http_get_json(url, params)
    rows = j.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows)
    # rename Polygon columns to simple names
    cols = {"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=cols)
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df

def within_week_70d_high(df: pd.DataFrame, week_start: dt.date, week_end: dt.date, lookback: int = 70) -> bool:
    if df.empty:
        return False
    s = pd.Series(df["high"].values, index=pd.to_datetime(df["date"]).dt.date)
    if len(s) < lookback + 1:
        return False
    # rolling max on last 70 days stepping through week window
    dates = [d for d in s.index if week_start <= d <= week_end]
    if not dates:
        return False
    for d in dates:
        # window is (d - 70d, d - 1d)
        start = d - dt.timedelta(days=lookback)
        window = s[(s.index >= start) & (s.index < d)]
        if len(window) >= lookback and s.loc[d] >= window.max():
            return True
    return False

# ---------- enrichment (per-ticker details) ----------
def fetch_market_caps_for(api_key: str, tickers: List[str], asof_date: Optional[str] = None) -> Dict[str, float]:
    """Ask /v3/reference/tickers/{ticker} for market_cap only for the shortlist."""
    out: Dict[str, float] = {}
    base = "https://api.polygon.io/v3/reference/tickers/"
    for i, s in enumerate(tickers, 1):
        params = {"apiKey": api_key}
        if asof_date:
            params["date"] = asof_date
        j = http_get_json(base + s, params)
        cap = (j.get("results") or {}).get("market_cap")
        if cap:
            try:
                out[s] = float(cap)
            except Exception:
                pass
        # be kind
        if i % 10 == 0:
            time.sleep(0.1)
    return out

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Scan 70D highs; filter to $1B+ AFTER enrichment.")
    p.add_argument("--cap-min", type=float, default=1_000_000_000.0, help="Min market cap (USD) for final shortlist")
    p.add_argument("--top", type=int, default=10, help="Top N to keep (sorted by market cap desc)")
    p.add_argument("--pages", type=int, default=3, help="Pages of /v3/reference/tickers (1000 per page)")
    p.add_argument("--max-syms", type=int, default=3000, help="Max symbols to consider before bar fetch")
    p.add_argument("--since-days", type=int, default=140, help="Bars back from week start to fetch (>= lookback)")
    p.add_argument("--lookback", type=int, default=70, help="Lookback days for high")
    p.add_argument("--friday", type=str, default="", help="Override Friday YYYY-MM-DD (optional)")
    p.add_argument("--out-dir", type=str, default="backtests", help="Output folder")
    args = p.parse_args()

    api_key = os.getenv("POLYGON_API_KEY") or os.getenv("POLY_KEY") or ""
    if not api_key:
        print("FATAL: POLYGON_API_KEY missing", file=sys.stderr)
        sys.exit(1)

    # Week window (Mon..Fri)
    if args.friday:
        fri = dt.date.fromisoformat(args.friday)
    else:
        fri = last_friday_europe_oslo()
    mon = fri - dt.timedelta(days=4)
    start_bars = mon - dt.timedelta(days=max(args.since_days, args.lookback + 5))

    print(f"Week window: {mon} .. {fri} | bars from: {start_bars} .. {fri}")

    # 1) list universe (no cap filter up front)
    ref = list_common_stocks(api_key, pages=args.pages, max_syms=args.max_syms, asof_date=str(fri))
    print(f"Fetched {len(ref)} active common stocks from Polygon (pages={args.pages}).")

    # NOTE: many plans return results without market_cap here; we do enrichment later
    nonnull_caps = sum(1 for r in ref if (r or {}).get("market_cap"))
    print(f"Reference rows with non-null market_cap: {nonnull_caps}/{len(ref)}")

    syms = [r.get("ticker") for r in ref if isinstance(r, dict) and r.get("ticker")]
    # 2) for each symbol, fetch bars & test 70D high in this week
    candidates: List[Tuple[str, float]] = []  # (symbol, last_close)
    kept = 0
    for i, s in enumerate(syms, 1):
        try:
            df = fetch_daily_bars(api_key, s, start=str(start_bars), end=str(fri))
            if df.empty:
                continue
            if within_week_70d_high(df, mon, fri, lookback=args.lookback):
                last_close = float(df["close"].iloc[-1])
                candidates.append((s, last_close))
        except Exception as e:
            # soft fail for a few symbols
            pass

        if i % 200 == 0:
            print(f"… scanned {i}/{len(syms)}; candidates so far = {len(candidates)}")

    print(f"Candidates that made a 70D high this week: {len(candidates)}")

    if not candidates:
        # still write empty files for downstream steps
        os.makedirs(args.out_dir, exist_ok=True)
        pd.DataFrame(columns=["symbol","market_cap","last_close"]).to_csv(
            os.path.join(args.out_dir, "hi70_thisweek.csv"), index=False
        )
        with open(os.path.join(args.out_dir, "hi70_digest.txt"), "w") as f:
            f.write("No 70D-high candidates this week.\n")
        return

    # 3) Enrich ONLY candidates with market cap
    cand_syms = [s for s, _ in candidates]
    caps = fetch_market_caps_for(api_key, cand_syms, asof_date=str(fri))
    nonnull = sum(1 for s in cand_syms if s in caps)
    print(f"Enriched market caps for {nonnull}/{len(cand_syms)} candidates.")

    rows = []
    for s, last_close in candidates:
        mc = caps.get(s, np.nan)
        rows.append({"symbol": s, "market_cap": mc, "last_close": last_close})
    df = pd.DataFrame(rows)

    # 4) Filter by $ cap, sort, top N
    kept = df[df["market_cap"].notna() & (df["market_cap"] >= args.cap_min)].copy()
    kept.sort_values("market_cap", ascending=False, inplace=True)
    kept = kept.head(args.top)
    print(f"Kept after cap≥{args.cap_min:,.0f}: {len(kept)}")

    # 5) Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "hi70_thisweek.csv")
    kept.to_csv(out_csv, index=False)

    with open(os.path.join(args.out_dir, "hi70_digest.txt"), "w") as f:
        if kept.empty:
            f.write("No $1B+ 70D-high breakouts this week.\n")
        else:
            f.write("*70D Highs ($1B+), sorted by market cap*\n")
            for _, r in kept.iterrows():
                f.write(f"- {r['symbol']}: mcap ${r['market_cap']:,.0f}, last_close {r['last_close']:,.2f}\n")

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

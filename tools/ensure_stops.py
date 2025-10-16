#!/usr/bin/env python3
"""
Ensure (seed or tighten) stop orders on Alpaca.

Order of preference for stop levels:
1) backtests/stops.csv (column: new_stop or stop) if present
2) Compute ATR(14) * 1.75 via Polygon aggregates, if POLYGON_API_KEY exists
3) Otherwise skip symbol.

Rules:
- Long positions => place SELL stop
- Short positions => place BUY stop
- GTC time in force
- Only tighten: new stop must be "better" than current (lower for long, higher for short)
- Round stop prices to Alpaca's tick sizes:
    >= $1.00 -> $0.01;  < $1.00 -> $0.0001
  For long (sell stop) round DOWN; for short (buy stop) round UP

Outputs a summary at the end.
"""

from __future__ import annotations
import os
import math
import time
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

try:
    import pandas as pd
except Exception:
    pd = None  # We'll guard if pandas is missing


# -------- Config --------
ATR_WIN = 14
ATR_MULT = 1.75

ALPACA_ENV = (os.getenv("ALPACA_ENV") or "paper").strip().lower()
BASE_TRADER = "https://paper-api.alpaca.markets" if ALPACA_ENV.startswith("paper") else "https://api.alpaca.markets"
BASE_DATA   = "https://data.alpaca.markets"       # not used here, we use Polygon for bars if available

AK = os.getenv("ALPACA_KEY", "").strip()
AS = os.getenv("ALPACA_SECRET", "").strip()
if not (AK and AS):
    print("FATAL: Missing ALPACA_KEY / ALPACA_SECRET", file=sys.stderr)
    sys.exit(2)

H_TRADE = {"APCA-API-KEY-ID": AK, "APCA-API-SECRET-KEY": AS}

POLYGON_KEY = (os.getenv("POLYGON_API_KEY") or "").strip()


# -------- Helpers --------
def round_to_tick(px: float, side: str) -> float:
    """
    Round price to Alpaca equity tick:
      >= $1.00 -> 0.01
       < $1.00 -> 0.0001
    For long (sell stop) round DOWN; for short (buy stop) round UP.
    """
    if px is None or math.isnan(px):
        return px
    tick = 0.01 if px >= 1.0 else 0.0001
    if side == "long":
        pxr = math.floor(px / tick) * tick
    else:  # short
        pxr = math.ceil(px / tick) * tick
    decimals = 2 if px >= 1.0 else 4
    pxr = max(tick, pxr)
    return float(f"{pxr:.{decimals}f}")


def better(curr: float, new: float, side: str) -> bool:
    """Is 'new' a tighter/better stop than 'curr' for the given side?"""
    if curr is None:
        return True
    if side == "long":
        return new < curr
    else:
        return new > curr


def get_positions() -> Dict[str, dict]:
    """Return map symbol -> position json."""
    r = requests.get(f"{BASE_TRADER}/v2/positions", headers=H_TRADE, timeout=20)
    if r.status_code != 200:
        print(f"[WARN] positions {r.status_code} {r.text}")
        return {}
    out = {}
    for p in (r.json() or []):
        try:
            sym = str(p.get("symbol") or "").upper()
            out[sym] = p
        except Exception:
            continue
    return out


def get_open_stop_orders() -> Dict[str, List[dict]]:
    """Return map symbol -> list of open stop/stop_limit orders."""
    r = requests.get(f"{BASE_TRADER}/v2/orders", params={"status": "open", "limit": 500},
                     headers=H_TRADE, timeout=20)
    if r.status_code != 200:
        print(f"[WARN] open orders {r.status_code} {r.text}")
        return {}
    bysym: Dict[str, List[dict]] = {}
    for od in (r.json() or []):
        typ = (od.get("type") or "").lower()
        if typ in ("stop", "stop_limit"):
            s = (od.get("symbol") or "").upper()
            bysym.setdefault(s, []).append(od)
    return bysym


def polygon_bars_daily(symbol: str, lookback_days: int = 80) -> Optional[pd.DataFrame]:
    """Fetch recent daily bars via Polygon. Returns DataFrame with columns: t, o, h, l, c, v."""
    if not POLYGON_KEY or pd is None:
        return None
    # Polygon aggregates v2
    import datetime as dt
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        return None
    j = r.json() or {}
    rows = j.get("results") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # unify col names
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "ts"})
    return df


def compute_atr_from_bars(df: pd.DataFrame, win: int = ATR_WIN) -> Optional[float]:
    """Wilder ATR from daily bars DataFrame (with open/high/low/close). Returns last ATR."""
    if df is None or df.empty or "high" not in df.columns:
        return None
    # Need at least win+1 closes
    if len(df) < win + 1:
        return None
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    trs = []
    prev_close = close[0]
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
        trs.append(tr)
        prev_close = close[i]
    # Wilder smoothing: first ATR = mean of first 'win' TRs; then recursive
    if len(trs) < win:
        return None
    import numpy as np
    atr0 = float(np.mean(trs[:win]))
    atr = atr0
    for tr in trs[win:]:
        atr = (atr * (win - 1) + tr) / win
    return float(atr)


def load_stops_csv(path="backtests/stops.csv") -> Dict[str, float]:
    """Return symbol -> stop (float) from backtests/stops.csv if present."""
    out: Dict[str, float] = {}
    if not (pd and os.path.isfile(path)):
        return out
    try:
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        # prefer new_stop then stop
        col = "new_stop" if "new_stop" in cols else ("stop" if "stop" in cols else None)
        symcol = "symbol" if "symbol" in cols else None
        if not col or not symcol:
            return out
        for _, r in df.iterrows():
            s = str(r[symcol]).upper()
            try:
                v = float(r[col])
            except Exception:
                continue
            out[s] = v
    except Exception as e:
        print(f"[WARN] load stops.csv: {e}")
    return out


@dataclass
class StopTarget:
    symbol: str
    side: str         # 'long' or 'short'
    qty: float
    last_close: float
    stop_price: float


def build_targets() -> List[StopTarget]:
    """Create StopTarget list from positions + stops.csv or Polygon ATR."""
    positions = get_positions()
    if not positions:
        print("[stops] no positions; nothing to do.")
        return []

    # Load explicit stops if available
    explicit = load_stops_csv()
    targets: List[StopTarget] = []

    for sym, pos in positions.items():
        try:
            qty = float(pos.get("qty") or 0.0)
        except Exception:
            qty = 0.0
        if abs(qty) < 1e-8:
            continue

        side = "long" if qty > 0 else "short"
        try:
            last_close = float(pos.get("avg_entry_price") or 0.0)
        except Exception:
            last_close = 0.0

        # Prefer explicit stops from file
        if sym in explicit:
            stop = explicit[sym]
            stop = round_to_tick(stop, side)
            targets.append(StopTarget(sym, side, abs(qty), last_close, stop))
            continue

        # Else attempt Polygon ATR stop
        stop: Optional[float] = None
        if POLYGON_KEY and pd is not None:
            df = polygon_bars_daily(sym, lookback_days=90)
            if df is not None and not df.empty:
                last_close = float(df["close"].iloc[-1])
                atr = compute_atr_from_bars(df, win=ATR_WIN)
                if atr and atr > 0:
                    raw = (last_close - ATR_MULT * atr) if side == "long" else (last_close + ATR_MULT * atr)
                    stop = round_to_tick(raw, side)

        if stop is None:
            print(f"[info] {sym}: no stops.csv and no Polygon ATR — skip.")
            continue

        targets.append(StopTarget(sym, side, abs(qty), last_close, stop))

    return targets


def ensure_or_tighten(targets: List[StopTarget]) -> Tuple[int, int, int]:
    """Place or tighten stops. Returns (placed, skipped, failed)."""
    placed = skipped = failed = 0
    existing = get_open_stop_orders()

    for t in targets:
        curr = None
        oid = None
        for od in existing.get(t.symbol, []):
            try:
                sp = od.get("stop_price") or od.get("limit_price")
                sp = float(sp) if sp is not None else None
            except Exception:
                sp = None
            if sp is not None:
                curr, oid = sp, od.get("id")
                break

        # sanity: ensure on correct side of market
        ns = float(t.stop_price)
        if t.side == "long" and ns >= t.last_close:
            ns = round_to_tick(t.last_close - 0.02, t.side)
        if t.side == "short" and ns <= t.last_close:
            ns = round_to_tick(t.last_close + 0.02, t.side)

        if curr is None or better(curr, ns, t.side):
            # Replace existing if any
            if oid:
                try:
                    requests.delete(f"{BASE_TRADER}/v2/orders/{oid}", headers=H_TRADE, timeout=20)
                except Exception:
                    pass
            payload = {
                "symbol": t.symbol,
                "qty": t.qty,
                "side": "sell" if t.side == "long" else "buy",
                "type": "stop",
                "time_in_force": "gtc",
                "stop_price": ns,
            }
            r = requests.post(f"{BASE_TRADER}/v2/orders", headers=H_TRADE, json=payload, timeout=25)
            if r.status_code in (200, 201):
                print(f"[{'placed' if curr is None else 'tightened'}] {t.symbol} stop={ns} side={t.side} tif=gtc")
                placed += 1
            else:
                print(f"[WARN] place fail {t.symbol} {r.status_code} {r.text}")
                failed += 1
        else:
            print(f"[keep ] {t.symbol} existing_stop={curr}")
            skipped += 1

        # Small polite delay
        time.sleep(0.1)

    return placed, skipped, failed


def main():
    # Report symbols detected (for transparency)
    targets = build_targets()
    syms = [t.symbol for t in targets]
    print(f"[stops] symbols: {syms!r}")

    if not targets:
        print("[stops] no symbols to process")
        return

    placed, skipped, failed = ensure_or_tighten(targets)
    print(f"[stops] summary: placed={placed}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()

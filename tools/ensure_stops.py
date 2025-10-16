#!/usr/bin/env python3
"""
Ensure (seed or tighten) stop orders on Alpaca.

Priority for stop levels:
1) backtests/stops.csv (column new_stop or stop)
2) Polygon ATR(14) * 1.75 if POLYGON_API_KEY is available
3) Otherwise skip

Rules:
- Long positions => SELL stop; Short positions => BUY stop
- Stops must be whole-share qty on Alpaca -> round down to int(abs(qty))
- Skip if resulting integer qty is 0 (position < 1 share)
- Tick rounding:
    px >= 1.00 -> 0.01;  px < 1.00 -> 0.0001
  Long: round price DOWN; Short: round price UP
- Only tighten (new better than existing)
- GTC time in force
"""

from __future__ import annotations
import os, math, time, sys
from typing import Dict, List, Optional, Tuple
import requests

try:
    import pandas as pd
except Exception:
    pd = None

# ------- Config -------
ATR_WIN  = 14
ATR_MULT = 1.75

ALPACA_ENV = (os.getenv("ALPACA_ENV") or "paper").strip().lower()
BASE_TRADER = "https://paper-api.alpaca.markets" if ALPACA_ENV.startswith("paper") else "https://api.alpaca.markets"

AK = os.getenv("ALPACA_KEY", "").strip()
AS = os.getenv("ALPACA_SECRET", "").strip()
if not (AK and AS):
    print("FATAL: Missing ALPACA_KEY/ALPACA_SECRET", file=sys.stderr); sys.exit(2)
H_TRADE = {"APCA-API-KEY-ID": AK, "APCA-API-SECRET-KEY": AS}

POLYGON_KEY = (os.getenv("POLYGON_API_KEY") or "").strip()

# ------- Helpers -------
def round_to_tick(px: float, side: str) -> float:
    if px is None or math.isnan(px): return px
    tick = 0.01 if px >= 1.0 else 0.0001
    if side == "long":
        pxr = math.floor(px / tick) * tick
    else:
        pxr = math.ceil(px / tick) * tick
    pxr = max(tick, pxr)
    decimals = 2 if px >= 1.0 else 4
    return float(f"{pxr:.{decimals}f}")

def better(curr: Optional[float], new: float, side: str) -> bool:
    if curr is None: return True
    return (new < curr) if side == "long" else (new > curr)

def get_positions() -> Dict[str, dict]:
    r = requests.get(f"{BASE_TRADER}/v2/positions", headers=H_TRADE, timeout=20)
    if r.status_code != 200:
        print(f"[WARN] positions {r.status_code} {r.text}"); return {}
    out = {}
    for p in (r.json() or []):
        s = str(p.get("symbol") or "").upper()
        out[s] = p
    return out

def get_open_stop_orders() -> Dict[str, List[dict]]:
    r = requests.get(f"{BASE_TRADER}/v2/orders", params={"status": "open", "limit": 500},
                     headers=H_TRADE, timeout=20)
    if r.status_code != 200:
        print(f"[WARN] open orders {r.status_code} {r.text}"); return {}
    bysym: Dict[str, List[dict]] = {}
    for od in (r.json() or []):
        if (od.get("type") or "").lower() in ("stop","stop_limit"):
            bysym.setdefault((od.get("symbol") or "").upper(), []).append(od)
    return bysym

def polygon_bars_daily(symbol: str, lookback_days: int = 90) -> Optional[pd.DataFrame]:
    if not POLYGON_KEY or pd is None: return None
    import datetime as dt
    end = dt.date.today(); start = end - dt.timedelta(days=lookback_days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted":"true","sort":"asc","limit":50000,"apiKey":POLYGON_KEY}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200: return None
    rows = (r.json() or {}).get("results") or []
    if not rows: return None
    df = pd.DataFrame(rows).rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","t":"ts"})
    return df

def compute_atr_from_bars(df: pd.DataFrame, win: int = ATR_WIN) -> Optional[float]:
    if df is None or df.empty or len(df) < win + 1: return None
    hi = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    cl = df["close"].astype(float).values
    trs = []
    prev = cl[0]
    for i in range(1, len(cl)):
        trs.append(max(hi[i]-lo[i], abs(hi[i]-prev), abs(lo[i]-prev)))
        prev = cl[i]
    if len(trs) < win: return None
    import numpy as np
    atr = float(np.mean(trs[:win]))
    for tr in trs[win:]:
        atr = (atr*(win-1)+tr)/win
    return float(atr)

def load_stops_csv(path="backtests/stops.csv") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not (pd and os.path.isfile(path)): return out
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        symcol = "symbol" if "symbol" in df.columns else None
        col = "new_stop" if "new_stop" in df.columns else ("stop" if "stop" in df.columns else None)
        if not symcol or not col: return out
        for _, r in df.iterrows():
            s = str(r[symcol]).upper()
            try: out[s] = float(r[col])
            except Exception: pass
    except Exception as e:
        print(f"[WARN] load stops.csv: {e}")
    return out

# ------- Build targets -------
class StopTarget:
    __slots__ = ("symbol","side","qty_int","last_close","stop_price")
    def __init__(self, symbol:str, side:str, qty_int:int, last_close:float, stop_price:float):
        self.symbol=symbol; self.side=side; self.qty_int=qty_int
        self.last_close=last_close; self.stop_price=stop_price

def build_targets() -> List[StopTarget]:
    positions = get_positions()
    if not positions:
        print("[stops] no positions; nothing to do."); return []

    explicit = load_stops_csv()
    targets: List[StopTarget] = []

    for sym, pos in positions.items():
        try:
            qty_raw = float(pos.get("qty") or 0.0)
        except Exception:
            qty_raw = 0.0
        if abs(qty_raw) < 1e-8:
            continue

        side = "long" if qty_raw > 0 else "short"
        qty_int = int(abs(qty_raw))  # whole shares only for stop orders
        if qty_int < 1:
            print(f"[skip ] {sym}: fractional position {qty_raw} (<1 share) — cannot place stop.")
            continue
        if abs(qty_raw - qty_int) > 1e-6:
            print(f"[note ] {sym}: qty {qty_raw} -> using whole shares {qty_int} for stop order.")

        # last close proxy (fallback to entry if no bars)
        try:
            last_close = float(pos.get("avg_entry_price") or 0.0)
        except Exception:
            last_close = 0.0

        # Prefer explicit from csv
        if sym in explicit:
            stop = round_to_tick(explicit[sym], side)
            targets.append(StopTarget(sym, side, qty_int, last_close, stop))
            continue

        # Else compute via Polygon ATR
        stop = None
        if POLYGON_KEY and pd is not None:
            df = polygon_bars_daily(sym, 90)
            if df is not None and not df.empty:
                last_close = float(df["close"].iloc[-1])
                atr = compute_atr_from_bars(df, ATR_WIN)
                if atr and atr > 0:
                    raw = (last_close - ATR_MULT*atr) if side=="long" else (last_close + ATR_MULT*atr)
                    stop = round_to_tick(raw, side)

        if stop is None:
            print(f"[info ] {sym}: no stops.csv & no ATR -> skip.")
            continue

        targets.append(StopTarget(sym, side, qty_int, last_close, stop))

    return targets

# ------- Ensure / Tighten -------
def get_existing_stop(symbol:str) -> Tuple[Optional[float], Optional[str]]:
    """Return (current_stop_price, order_id) if a stop exists."""
    r = requests.get(f"{BASE_TRADER}/v2/orders", params={"status":"open","symbols":symbol,"limit":200},
                     headers=H_TRADE, timeout=20)
    if r.status_code != 200:
        return None, None
    for od in (r.json() or []):
        if (od.get("type") or "").lower() in ("stop","stop_limit"):
            sp = od.get("stop_price") or od.get("limit_price")
            try:
                return (float(sp), od.get("id"))
            except Exception:
                return None, od.get("id")
    return None, None

def ensure_or_tighten(targets: List[StopTarget]) -> Tuple[int,int,int]:
    placed=skipped=failed=0
    for t in targets:
        curr, oid = get_existing_stop(t.symbol)

        ns = float(t.stop_price)
        # sanity against last close
        if t.side=="long" and ns >= t.last_close:
            ns = round_to_tick(t.last_close - 0.02, t.side)
        if t.side=="short" and ns <= t.last_close:
            ns = round_to_tick(t.last_close + 0.02, t.side)

        if curr is None or better(curr, ns, t.side):
            if oid:
                try: requests.delete(f"{BASE_TRADER}/v2/orders/{oid}", headers=H_TRADE, timeout=20)
                except Exception: pass
            payload = {
                "symbol": t.symbol,
                "qty": t.qty_int,                      # WHOLE SHARES
                "side": "sell" if t.side=="long" else "buy",
                "type": "stop",
                "time_in_force": "gtc",
                "stop_price": ns,
            }
            r = requests.post(f"{BASE_TRADER}/v2/orders", headers=H_TRADE, json=payload, timeout=25)
            if r.status_code in (200,201):
                action = "placed" if curr is None else "tightened"
                print(f"[{action}] {t.symbol} qty={t.qty_int} stop={ns} side={t.side} tif=gtc")
                placed += 1
            else:
                print(f"[WARN ] place fail {t.symbol} ({t.qty_int} sh) {r.status_code} {r.text}")
                failed += 1
        else:
            print(f"[keep ] {t.symbol} existing_stop={curr} <= better unchanged")
            skipped += 1

        time.sleep(0.1)
    return placed, skipped, failed

def main():
    targets = build_targets()
    syms = [t.symbol for t in targets]
    print(f"[stops] symbols: {syms!r}")
    if not targets:
        print("[stops] no symbols to process"); return
    placed, skipped, failed = ensure_or_tighten(targets)
    print(f"[stops] summary: placed={placed}, skipped={skipped}, failed={failed}")

if __name__ == "__main__":
    main()

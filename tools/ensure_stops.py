#!/usr/bin/env python3
"""
ensure_stops.py
Seed (or tighten) GTC stop orders for newly-bought symbols, using ATR(14)*1.75.

Behavior
- Reads allow-list from backtests/buy_symbols.txt (one symbol per line).
  If missing/empty, falls back to all current positions.
- Waits up to ~60s for those positions to appear in Alpaca (post-buy race proof).
- Computes ATR(14) from Alpaca 1D bars; stop = close - k*ATR (long), close + k*ATR (short).
- Places GTC stop orders; if a stop exists, replaces only if tighter (better).

Env (required)
  ALPACA_KEY, ALPACA_SECRET
  ALPACA_ENV=paper|live   (defaults to paper)

Optional
  ATR_WIN=14
  ATR_MULT=1.75
  ALLOW_ALL_POS=true  (if you want to ignore file and seed for all open positions)
"""

from __future__ import annotations
import os, time, math, sys, json, typing as T
import requests
import numpy as np
import pandas as pd

def base_url() -> str:
    env = os.getenv("ALPACA_ENV","paper").lower()
    return "https://paper-api.alpaca.markets" if env.startswith("paper") else "https://api.alpaca.markets"

def headers() -> dict:
    key = os.getenv("ALPACA_KEY","").strip()
    sec = os.getenv("ALPACA_SECRET","").strip()
    if not key or not sec:
        print("FATAL: missing ALPACA_KEY/ALPACA_SECRET", file=sys.stderr)
        sys.exit(2)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

def load_allow_symbols(path="backtests/buy_symbols.txt") -> list[str]:
    try:
        with open(path) as f:
            syms = [s.strip().upper() for s in f if s.strip()]
        return syms
    except FileNotFoundError:
        return []

def get_positions(H: dict) -> list[dict]:
    r = requests.get(f"{base_url()}/v2/positions", headers=H, timeout=20)
    if r.status_code != 200:
        return []
    return r.json() or []

def wait_for_positions(H: dict, needed: set[str], tries: int = 30, delay_s: float = 2.0) -> set[str]:
    """Poll positions until all 'needed' are present, or timeout."""
    if not needed:
        return set()
    for _ in range(tries):
        pos = get_positions(H)
        have = { (p.get("symbol") or "").upper() for p in pos }
        if needed.issubset(have):
            return have
        time.sleep(delay_s)
    return have  # last seen

def fetch_daily_bars(H: dict, sym: str, limit: int = 100) -> pd.DataFrame | None:
    r = requests.get(
        f"{base_url()}/v2/stocks/{sym}/bars",
        params={"timeframe":"1Day","limit":limit},
        headers=H, timeout=30
    )
    if r.status_code != 200: return None
    j = r.json() or {}
    bars = j.get("bars") or []
    if not bars: return None
    df = pd.DataFrame(bars)
    # expected fields: t (ISO), o,h,l,c,v etc.
    need = {"o","h","l","c"}
    if not need.issubset(df.columns):
        return None
    return df[["o","h","l","c"]].rename(columns={"o":"open","h":"high","l":"low","c":"close"}).astype(float)

def atr14(df: pd.DataFrame, win: int) -> float | None:
    if df is None or df.empty or len(df) < win+1:
        return None
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    # Wilder's ATR (EMA-like) or simple rolling mean; use rolling mean here for clarity:
    atr = pd.Series(tr).rolling(win, min_periods=win).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else None

def open_stop_orders(H: dict) -> dict[str, list[dict]]:
    r = requests.get(f"{base_url()}/v2/orders", params={"status":"open","limit":500}, headers=H, timeout=30)
    if r.status_code != 200: return {}
    bysym: dict[str, list[dict]] = {}
    for od in r.json() or []:
        if od.get("type") in ("stop","stop_limit"):
            s = (od.get("symbol") or "").upper()
            bysym.setdefault(s, []).append(od)
    return bysym

def tighten_only(curr: float | None, new: float, side: str) -> bool:
    """Return True if new stop is 'tighter' than current (better)."""
    if curr is None:
        return True
    if side == "long":
        return new > curr
    else:
        return new < curr

def ensure_stops_for(H: dict, targets: set[str], win: int, mult: float) -> dict:
    pos_list = get_positions(H)
    pos = { (p.get("symbol") or "").upper(): p for p in pos_list }
    bysym_openstops = open_stop_orders(H)

    results = {"placed":0, "skipped":0, "failed":0, "symbols":[]}

    for s in sorted(targets):
        if s not in pos:
            results["skipped"] += 1
            continue
        qty = abs(float(pos[s].get("qty") or 0))
        if qty <= 0:
            results["skipped"] += 1
            continue

        side = "long" if float(pos[s].get("qty",0)) > 0 else "short"
        df = fetch_daily_bars(H, s, limit=max(64, win+20))
        if df is None or df.empty:
            print(f"- {s}: no bars; skip.")
            results["skipped"] += 1
            continue

        a = atr14(df, win)
        close = float(df["close"].iloc[-1])
        if a is None or not np.isfinite(a) or close <= 0:
            print(f"- {s}: invalid ATR/close; skip.")
            results["skipped"] += 1
            continue

        if side == "long":
            new_stop = close - mult * a
        else:
            new_stop = close + mult * a

        curr, oid = None, None
        for od in bysym_openstops.get(s, []):
            try:
                curr = float(od.get("stop_price") or od.get("limit_price"))
                oid  = od.get("id")
                break
            except Exception:
                pass

        if not tighten_only(curr, new_stop, side):
            print(f"- {s}: existing stop better (curr={curr:.4f}) vs new={new_stop:.4f}; keep.")
            results["skipped"] += 1
            continue

        # replace worse one if exists
        if oid:
            try:
                requests.delete(f"{base_url()}/v2/orders/{oid}", headers=H, timeout=20)
            except Exception:
                pass

        order = {
            "symbol": s,
            "qty": qty,
            "side": "sell" if side=="long" else "buy",
            "type": "stop",
            "time_in_force": "gtc",
            "stop_price": round(new_stop, 4),
        }
        r = requests.post(f"{base_url()}/v2/orders", headers=H, json=order, timeout=30)
        if r.status_code in (200,201):
            print(f"- {s}: placed/tightened stop @ {order['stop_price']:.4f} (side={side}, qty={qty})")
            results["placed"] += 1
            results["symbols"].append(s)
        else:
            print(f"- {s}: STOP FAILED {r.status_code} {r.text}")
            results["failed"] += 1
    return results

def main():
    H = headers()
    atr_win  = int(os.getenv("ATR_WIN","14") or 14)
    atr_mult = float(os.getenv("ATR_MULT","1.75") or 1.75)

    # Preferred target list: newly bought weekly symbols
    allow = set(load_allow_symbols())
    allow_all = str(os.getenv("ALLOW_ALL_POS","")).lower() == "true"

    if not allow or allow_all:
        # Fall back to all open positions (or explicit override)
        pos = get_positions(H)
        allow = { (p.get("symbol") or "").upper() for p in pos if float(p.get("qty") or 0) != 0 }

    if not allow:
        print("No target symbols found (no buy_symbols.txt and no positions). Nothing to do.")
        return

    print("[stops] target symbols:", sorted(allow))

    # Race-proof: wait for positions to appear
    have = wait_for_positions(H, allow, tries=30, delay_s=2.0)
    missing = allow - have
    if missing:
        print(f"[stops] WARNING: positions not visible for {sorted(missing)}; continuing with what we have.")

    res = ensure_stops_for(H, allow - missing if have else allow, atr_win, atr_mult)
    print("[stops] summary:", json.dumps(res))

if __name__ == "__main__":
    main()

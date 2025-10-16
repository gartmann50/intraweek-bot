#!/usr/bin/env python3
"""
Place weekly buys using ONLY whole shares.

Inputs (via env + files):
- ENV:
    ALPACA_KEY, ALPACA_SECRET, ALPACA_ENV  (paper|live)
    NOTIONAL_PER  (e.g. "5000")
    WEEK          (YYYY-MM-DD, exported earlier in the job)
- Files:
    backtests/buy_symbols.txt   # one symbol per line
    stock_data_400/             # cached EOD CSVs (used to fetch last close)

Behavior:
- For each symbol, get last close from cache (fallback to Alpaca last trade).
- qty = floor(NOTIONAL_PER / last_price). If qty < 1 -> skip.
- Market DAY order with explicit qty (no notional param => whole shares only).
- Writes backtests/buy_exec_report.csv for audit.
"""
from __future__ import annotations

import os, sys, math, glob, time, json
import pathlib as p
import requests
import pandas as pd

ALPACA_ENV = os.getenv("ALPACA_ENV", "paper").strip().lower()
BASE = "https://paper-api.alpaca.markets" if ALPACA_ENV.startswith("paper") else "https://api.alpaca.markets"
H = {
    "APCA-API-KEY-ID":     os.environ.get("ALPACA_KEY", ""),
    "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
}
DATA_DIR = p.Path("stock_data_400")
BUY_FILE = p.Path("backtests/buy_symbols.txt")
REPORT = p.Path("backtests/buy_exec_report.csv")

def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def last_close_from_cache(sym: str) -> float | None:
    """Return last close from our cached CSVs, or None."""
    if not DATA_DIR.exists():
        return None
    paths = sorted(glob.glob(str(DATA_DIR / f"{sym}*.csv")))
    for path in paths:
        try:
            df = pd.read_csv(path, usecols=[0,4])  # Date, Close
            if len(df):
                return float(df.iloc[-1, 1])
        except Exception:
            pass
    return None

def last_trade_from_alpaca(sym: str) -> float | None:
    """Fallback: last trade from Alpaca data API (works for paper/live if allowed)."""
    try:
        r = requests.get(f"{BASE}/v2/stocks/{sym}/trades/latest", headers=H, timeout=10)
        if r.status_code == 200:
            j = r.json() or {}
            t = j.get("trade") or {}
            px = t.get("p")
            return float(px) if px is not None else None
    except Exception:
        pass
    return None

def money(s: float | int) -> str:
    return f"${s:,.2f}"

def place_market_day(sym: str, qty: int) -> dict:
    """Place a market DAY order with whole-share qty."""
    payload = {
        "symbol": sym,
        "qty": qty,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{BASE}/v2/orders", headers=H, json=payload, timeout=15)
    try:
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        try:
            print(f"[order] FAIL {sym} x{qty} -> {r.status_code} {r.text}")
        except Exception:
            print(f"[order] FAIL {sym} x{qty} -> {e}")
        return {"error": str(e), "status": r.status_code, "body": r.text}

def main():
    # Sanity
    if not H["APCA-API-KEY-ID"] or not H["APCA-API-SECRET-KEY"]:
        die("Missing ALPACA_KEY/ALPACA_SECRET in env.")
    try:
        notional = float(os.getenv("NOTIONAL_PER", "0").replace(",","").strip() or 0)
    except Exception:
        die("Invalid NOTIONAL_PER (must be number).")
    if notional <= 0:
        die("NOTIONAL_PER must be > 0.")
    week = os.getenv("WEEK", "").strip()
    if not week:
        print("[warn] WEEK not set; proceeding anyway.")

    if not BUY_FILE.exists():
        print("[buy] no buy_symbols.txt → nothing to buy.")
        return

    syms = [s.strip().upper() for s in BUY_FILE.read_text().splitlines() if s.strip()]
    if not syms:
        print("[buy] empty buy_symbols.txt → nothing to buy.")
        return

    rows = []
    placed = 0
    skipped = 0
    for sym in syms:
        # 1) Fetch a price
        px = last_close_from_cache(sym)
        src = "cache"
        if px is None:
            px = last_trade_from_alpaca(sym)
            src = "alpaca" if px is not None else "none"

        if px is None or px <= 0:
            print(f"[buy] {sym}: no price → skip.")
            rows.append({"symbol": sym, "price": "", "src": src, "qty": 0, "status": "skip_no_price"})
            skipped += 1
            continue

        # 2) Whole-share qty
        qty = int(math.floor(notional / px))
        if qty < 1:
            print(f"[buy] {sym}: price {money(px)} > budget {money(notional)} → skip.")
            rows.append({"symbol": sym, "price": px, "src": src, "qty": 0, "status": "skip_too_expensive"})
            skipped += 1
            continue

        print(f"[buy] {sym}: price {money(px)} (src={src}) notional {money(notional)} ⇒ qty {qty}")
        od = place_market_day(sym, qty)
        status = "ok" if od and od.get("id") else "fail"
        if status == "ok":
            placed += 1
        rows.append({
            "symbol": sym,
            "price": px,
            "src": src,
            "qty": qty,
            "status": status,
            "order_id": (od or {}).get("id", ""),
        })
        time.sleep(0.3)  # gentle rate

    # Write report
    p.Path("backtests").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(f"[buy] placed={placed}, skipped={skipped}. report: {REPORT}")

if __name__ == "__main__":
    main()

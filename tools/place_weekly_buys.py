#!/usr/bin/env python3
"""
API-only weekly buys (no local CSV / no DATA_DIR).

Inputs:
  backtests/buy_symbols.txt  (one symbol per line)

Env:
  ALPACA_KEY, ALPACA_SECRET
  ALPACA_ENV in {paper, live}  (default: paper)
  NOTIONAL_PER (e.g. "5000")

Outputs:
  backtests/buy_exec_report.csv
  backtests/model_symbols.txt   (symbols this model bought/holds this week)
"""

import os
import sys
import math
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
import pandas as pd

TRADING_PAPER = "https://paper-api.alpaca.markets"
TRADING_LIVE  = "https://api.alpaca.markets"
DATA_BASE     = "https://data.alpaca.markets/v2"

ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_ENV    = os.environ.get("ALPACA_ENV", "paper").lower()
NOTIONAL_PER  = float(os.environ.get("NOTIONAL_PER", "0") or "0")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

def trading_base() -> str:
    return TRADING_PAPER if ALPACA_ENV.startswith("paper") else TRADING_LIVE

def read_symbols(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    syms: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s:
            syms.append(s)
    return syms

def latest_trade_price(sym: str) -> Optional[float]:
    try:
        r = requests.get(f"{DATA_BASE}/stocks/{sym}/trades/latest", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            j = r.json() or {}
            t = j.get("trade") or {}
            p = t.get("p")
            if isinstance(p, (int, float)) and p > 0:
                return float(p)
    except Exception as e:
        print(f"[price] trade error {sym}: {e}")
    return None

def latest_quote_mid(sym: str) -> Optional[float]:
    try:
        r = requests.get(f"{DATA_BASE}/stocks/{sym}/quotes/latest", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            j = r.json() or {}
            q = j.get("quote") or {}
            bp = q.get("bp")
            ap = q.get("ap")
            if isinstance(bp, (int, float)) and isinstance(ap, (int, float)) and ap > 0:
                return (float(bp) + float(ap)) / 2.0
    except Exception as e:
        print(f"[price] quote error {sym}: {e}")
    return None

def resolve_price(sym: str) -> Optional[float]:
    p = latest_trade_price(sym)
    if p and p > 0:
        return p
    p = latest_quote_mid(sym)
    if p and p > 0:
        return p
    return None

def asset_is_tradable(sym: str) -> bool:
    try:
        r = requests.get(f"{trading_base()}/v2/assets/{sym}", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            j = r.json() or {}
            return bool(j.get("tradable", False))
    except Exception:
        pass
    return False

def place_market_buy(sym: str, qty: int) -> Dict[str, Any]:
    o = {"symbol": sym, "qty": qty, "side": "buy", "type": "market", "time_in_force": "day"}
    try:
        r = requests.post(f"{trading_base()}/v2/orders", headers=HEADERS, json=o, timeout=30)
        if r.status_code in (200, 201):
            oid = (r.json() or {}).get("id", "")
            return {"ok": True, "order_id": oid, "status": "placed"}
        return {"ok": False, "order_id": "", "status": f"fail_{r.status_code}", "msg": r.text[:300]}
    except Exception as e:
        return {"ok": False, "order_id": "", "status": "exception", "msg": str(e)[:300]}

def append_model_symbols(new_syms: List[str], path: str = "backtests/model_symbols.txt") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = set(read_symbols(path))
    for s in new_syms:
        existing.add(s)
    p.write_text("\n".join(sorted(existing)) + ("\n" if existing else ""), encoding="utf-8")

def main() -> int:
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("ERROR: Missing ALPACA_KEY/ALPACA_SECRET")
        return 2
    if NOTIONAL_PER <= 0:
        print("ERROR: NOTIONAL_PER must be > 0")
        return 2

    Path("backtests").mkdir(exist_ok=True)

    buy_list = read_symbols("backtests/buy_symbols.txt")
    if not buy_list:
        print("[buy] backtests/buy_symbols.txt empty or missing -> nothing to do")
        return 0

    rows = []
    bought_syms: List[str] = []

    for sym in buy_list:
        price = resolve_price(sym)
        if not price:
            rows.append({"symbol": sym, "status": "skip_no_price"})
            print(f"[buy] {sym}: skip (no price from Alpaca data API)")
            continue

        if not asset_is_tradable(sym):
            rows.append({"symbol": sym, "status": "skip_not_tradable", "price_used": price})
            print(f"[buy] {sym}: skip (not tradable)")
            continue

        qty = int(math.floor(NOTIONAL_PER / price))
        if qty < 1:
            rows.append({"symbol": sym, "status": "skip_small_budget", "price_used": price})
            print(f"[buy] {sym}: skip (budget too small for whole shares at {price:.2f})")
            continue

        res = place_market_buy(sym, qty)
        row = {"symbol": sym, "price_used": price, "qty": qty, **res}
        rows.append(row)

        if res.get("ok"):
            bought_syms.append(sym)
            print(f"[buy] placed {sym} qty={qty} at est_price={price:.2f} id={res.get('order_id','')}")
        else:
            print(f"[buy] FAILED {sym} qty={qty}: {res.get('status')} {res.get('msg','')}")

        time.sleep(0.25)

    pd.DataFrame(rows).to_csv("backtests/buy_exec_report.csv", index=False)

    # Track model symbols for later "flatten model only" logic
    if bought_syms:
        append_model_symbols(bought_syms, "backtests/model_symbols.txt")

    print(f"[buy] done. report=backtests/buy_exec_report.csv, model_symbols={'yes' if bought_syms else 'no'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

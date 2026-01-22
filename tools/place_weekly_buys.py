#!/usr/bin/env python3
"""
Place weekly buy orders using whole shares only.

Price source fallback chain (first that works wins):
  1) Alpaca latest trade (/v2/stocks/{sym}/trades/latest) -> price
  2) Alpaca latest quote (/v2/stocks/{sym}/quotes/latest) -> (bid+ask)/2
  3) Local CSV in DATA_DIR (last close)

Requires env:
  ALPACA_KEY, ALPACA_SECRET, ALPACA_ENV in {paper,live}
  NOTIONAL_PER  (e.g. "5000")
Optional:
  DATA_DIR (default: stock_data_400)
Inputs:
  backtests/buy_symbols.txt  (one symbol per line)
Outputs:
  backtests/buy_exec_report.csv
  backtests/model_symbols.txt   <-- NEW: symbols successfully placed by this model
"""

import os, sys, json, glob, math, time
from typing import Optional, Tuple, List
import requests
import pandas as pd
from pathlib import Path

TRADING_PAPER = "https://paper-api.alpaca.markets"
TRADING_LIVE  = "https://api.alpaca.markets"
DATA_BASE     = "https://data.alpaca.markets/v2"

ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_ENV    = os.environ.get("ALPACA_ENV", "paper").lower()
NOTIONAL_PER  = float(os.environ.get("NOTIONAL_PER", "0") or "0")
DATA_DIR      = os.environ.get("DATA_DIR", "stock_data_400")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

def trading_base() -> str:
    return TRADING_PAPER if ALPACA_ENV.startswith("paper") else TRADING_LIVE

def read_buy_symbols(path="backtests/buy_symbols.txt") -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"[buy] no list found at {p} -> nothing to do")
        return []
    syms = []
    for line in p.read_text().splitlines():
        s = line.strip().upper()
        if s:
            syms.append(s)
    return syms

def alpaca_latest_trade(sym: str) -> Optional[float]:
    try:
        r = requests.get(f"{DATA_BASE}/stocks/{sym}/trades/latest", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            j = r.json() or {}
            # v2 format: {"symbol":"AAPL","trade":{"p": 123.45, ...}}
            t = (j.get("trade") or {})
            p = t.get("p")
            if isinstance(p, (int, float)) and p > 0:
                return float(p)
    except Exception as e:
        print(f"[price] trade err {sym}: {e}")
    return None

def alpaca_latest_quote_mid(sym: str) -> Optional[float]:
    try:
        r = requests.get(f"{DATA_BASE}/stocks/{sym}/quotes/latest", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            j = r.json() or {}
            # v2 format: {"symbol":"AAPL","quote":{"bp":..., "ap":...}}
            q = (j.get("quote") or {})
            bp, ap = q.get("bp"), q.get("ap")
            if isinstance(bp, (int, float)) and isinstance(ap, (int, float)) and ap > 0:
                return (float(bp) + float(ap)) / 2.0
    except Exception as e:
        print(f"[price] quote err {sym}: {e}")
    return None

def local_last_close(sym: str) -> Optional[float]:
    try:
        files = sorted(glob.glob(f"{DATA_DIR}/{sym}*.csv"))
        for fp in files:
            try:
                df = pd.read_csv(fp, usecols=[0,4])  # Date, Close
                if len(df):
                    v = float(df.iloc[-1,1])
                    if v > 0:
                        return v
            except Exception:
                pass
    except Exception as e:
        print(f"[price] local err {sym}: {e}")
    return None

def resolve_price(sym: str) -> Optional[float]:
    for fn in (alpaca_latest_trade, alpaca_latest_quote_mid, local_last_close):
        p = fn(sym)
        if p and p > 0:
            return float(p)
    return None

def asset_meta(sym: str) -> Tuple[bool,bool]:
    """tradable, fractionable (we only buy whole shares, but useful to log)"""
    try:
        r = requests.get(f"{trading_base()}/v2/assets/{sym}", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            j = r.json() or {}
            return bool(j.get("tradable", False)), bool(j.get("fractionable", False))
    except Exception:
        pass
    return False, False

def place_market_buy(sym: str, qty: int) -> Tuple[bool,str]:
    try:
        o = {
            "symbol": sym,
            "qty": qty,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }
        r = requests.post(f"{trading_base()}/v2/orders", headers=HEADERS, json=o, timeout=20)
        if r.status_code in (200, 201):
            oid = (r.json() or {}).get("id", "")
            print(f"[buy] placed {sym} qty={qty} id={oid}")
            return True, oid
        else:
            print(f"[buy] FAIL {sym} qty={qty} code={r.status_code} msg={r.text[:300]}")
    except Exception as e:
        print(f"[buy] EXC {sym} qty={qty}: {e}")
    return False, ""

def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("Missing ALPACA_KEY/ALPACA_SECRET"); sys.exit(1)
    if NOTIONAL_PER <= 0:
        print("NOTIONAL_PER must be > 0"); sys.exit(1)

    Path("backtests").mkdir(exist_ok=True)
    symbols = read_buy_symbols()
    if not symbols:
        print("[buy] no symbols to buy; exit 0")
        Path("backtests/model_symbols.txt").write_text("")  # NEW: keep file present
        return

    rows = []
    placed = 0

    # NEW: track symbols successfully submitted by this model
    model_syms: List[str] = []

    for s in symbols:
        price = resolve_price(s)
        if not price or price <= 0:
            print(f"[buy] {s}: no price -> skip.")
            rows.append({"symbol": s, "status": "skip_no_price"})
            continue

        tradable, fractionable = asset_meta(s)
        if not tradable:
            print(f"[buy] {s}: not tradable -> skip.")
            rows.append({"symbol": s, "status": "skip_not_tradable", "price": price})
            continue

        qty = int(math.floor(NOTIONAL_PER / price))
        if qty < 1:
            print(f"[buy] {s}: floor({NOTIONAL_PER}/{price:.2f}) < 1 -> skip.")
            rows.append({"symbol": s, "status": "skip_small_budget", "price": price})
            continue

        ok, oid = place_market_buy(s, qty)
        rows.append({
            "symbol": s, "status": "placed" if ok else "failed",
            "price_used": price, "qty": qty, "order_id": oid,
            "fractionable": fractionable
        })
        if ok:
            placed += 1
            model_syms.append(s)  # NEW
        time.sleep(0.2)  # be polite

    # NEW: write allow-list for Friday flatten to use
    Path("backtests/model_symbols.txt").write_text(
        "\n".join(sorted(set(model_syms))) + ("\n" if model_syms else "")
    )
    print("[buy] model_symbols.txt:", sorted(set(model_syms)))

    df = pd.DataFrame(rows)
    df.to_csv("backtests/buy_exec_report.csv", index=False)
    print(f"[buy] placed={placed}, skipped={len(symbols)-placed}")
    print(f"[buy] report: backtests/buy_exec_report.csv")

if __name__ == "__main__":
    main()

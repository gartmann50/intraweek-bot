#!/usr/bin/env python3
"""
Place weekly buys safely:
- Reads backtests/buy_symbols.txt (one symbol per line).
- Posts MARKET DAY orders to Alpaca:
    * Prefer notional per name
    * If asset is not fractionable, compute integer qty via last daily close
- Skips anything not in the allow-list file
- Attaches client_order_id = IWOPEN-<run>-<symbol>
Env required: ALPACA_KEY, ALPACA_SECRET, (optional) ALPACA_ENV=paper
"""

from __future__ import annotations
import os, sys, json, math, requests

BASE = "https://paper-api.alpaca.markets" if os.getenv("ALPACA_ENV","paper").startswith("paper") \
       else "https://api.alpaca.markets"
H = {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY",""),
     "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET","")}

def must_env():
    miss = [k for k in ("ALPACA_KEY","ALPACA_SECRET") if not os.getenv(k)]
    if miss:
        print(f"Missing env: {miss}", file=sys.stderr)
        sys.exit(2)

def load_symbols(path="backtests/buy_symbols.txt"):
    try:
        with open(path) as f:
            syms = [s.strip().upper() for s in f if s.strip()]
        if not syms:
            print("No symbols in buy_symbols.txt — nothing to do.")
        return syms
    except FileNotFoundError:
        print("buy_symbols.txt not found — nothing to do.")
        return []

def is_fractionable(sym: str) -> bool:
    r = requests.get(f"{BASE}/v2/assets/{sym}", headers=H, timeout=20)
    if r.status_code != 200: return False
    return bool((r.json() or {}).get("fractionable", False))

def last_close(sym: str) -> float:
    r = requests.get(f"{BASE}/v2/stocks/{sym}/bars",
                     params={"timeframe":"1Day","limit":1},
                     headers=H, timeout=20)
    if r.status_code==200 and (r.json() or {}).get("bars"):
        return float(r.json()["bars"][0].get("c", 0) or 0)
    return 0.0

def main():
    must_env()
    per = float(os.getenv("NOTIONAL_PER","5000") or 5000)
    allow = load_symbols()
    if not allow: return
    placed = []
    run = os.getenv("GITHUB_RUN_ID","?")
    wf  = os.getenv("GITHUB_WORKFLOW","OPEN")
    for s in allow:
        frac = is_fractionable(s)
        payload = {"symbol": s, "side":"buy", "type":"market", "time_in_force":"day"}
        if frac:
            payload["notional"] = round(per, 2)
        else:
            px = last_close(s)
            if px <= 0:
                print(f"- {s}: no price; skip non-fractionable.")
                continue
            qty = max(1, int(per // px))
            if qty < 1:
                print(f"- {s}: per={per} < price={px:.2f}; skip.")
                continue
            # safety clamp ~≤1.25x target
            if px * qty > per * 1.25:
                qty = max(1, int((per*1.1) // px))
            payload["qty"] = qty
        payload["client_order_id"] = f"IWOPEN-{wf}-{run}-{s}"[:48]
        r = requests.post(f"{BASE}/v2/orders", headers=H, json=payload, timeout=30)
        if r.status_code not in (200,201):
            print(f"- {s}: FAIL {r.status_code} {r.text}")
            continue
        print(f"- {s}: OK {json.dumps(payload)}")
        placed.append((s, payload.get("notional"), payload.get("qty")))
    print("Placed:", placed)

if __name__ == "__main__":
    main()

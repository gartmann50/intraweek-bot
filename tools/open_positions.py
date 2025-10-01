#!/usr/bin/env python3
import os, sys, requests

BASE = ("https://paper-api.alpaca.markets"
        if os.getenv("ALPACA_ENV", "paper").startswith("paper")
        else "https://api.alpaca.markets")

K = os.getenv("ALPACA_KEY")
S = os.getenv("ALPACA_SECRET")
if not K or not S:
    print("Alpaca not configured; skipping.")
    sys.exit(0)

H = {"APCA-API-KEY-ID": K, "APCA-API-SECRET-KEY": S}

try:
    r = requests.get(f"{BASE}/v2/positions", headers=H, timeout=20)
    r.raise_for_status()
    pos = r.json()
except Exception as e:
    print("Positions error:", e)
    sys.exit(1)

if not pos:
    print("Open positions: none")
    sys.exit(0)

def fmt(p):
    return f"{p.get('symbol')}:{p.get('qty')}@{p.get('avg_entry_price')}"

print("Open positions:", ", ".join(fmt(p) for p in pos))

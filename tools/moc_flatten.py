#!/usr/bin/env python3
import os, sys, requests, datetime as dt
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

BASE = ("https://paper-api.alpaca.markets"
        if os.getenv("ALPACA_ENV", "paper").startswith("paper")
        else "https://api.alpaca.markets")
K = os.getenv("ALPACA_KEY")
S = os.getenv("ALPACA_SECRET")
if not K or not S:
    print("Alpaca not configured; skipping.")
    sys.exit(0)

H = {"APCA-API-KEY-ID": K, "APCA-API-SECRET-KEY": S}
force = "--force" in sys.argv

if not force and ZoneInfo:
    now_ny = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if now_ny.weekday() != 4:  # not Friday
        print("Not Friday; skip flatten.")
        sys.exit(0)

# Get open positions
r = requests.get(f"{BASE}/v2/positions", headers=H, timeout=20)
if r.status_code != 200:
    print("positions error", r.status_code, r.text)
    sys.exit(1)
pos = r.json()
if not pos:
    print("MOC placed: 0")
    sys.exit(0)

placed, errors = 0, 0
for p in pos:
    sym = p.get("symbol")
    qty = p.get("qty")
    if not sym or not qty:
        continue
    try:
        q = float(qty)
    except Exception:
        continue

    side = "sell" if q > 0 else "buy"
    order = {
        "symbol": sym,
        "qty": abs(q),
        "side": side,
        "type": "market",
        "time_in_force": "cls"  # Market-On-Close at Alpaca
    }
    rr = requests.post(f"{BASE}/v2/orders", json=order, headers=H, timeout=20)
    if rr.status_code == 200:
        placed += 1
    else:
        errors += 1
        print("order error", sym, rr.status_code, rr.text)

print(f"MOC placed: {placed}  errors: {errors}")

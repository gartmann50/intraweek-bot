#!/usr/bin/env python3
import os, json, sys, requests

def base():
    env = os.getenv("ALPACA_ENV","paper").lower()
    return "https://paper-api.alpaca.markets" if env.startswith("paper") else "https://api.alpaca.markets"

def H():
    return {
        "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
    }

def bad(msg): 
    print("BLOCKED:", msg); sys.exit(78)

# ---- parse payload
try:
    intent = json.loads(os.getenv("INTENT_JSON","{}") or "{}")
except Exception as e:
    bad(f"invalid JSON ({e})")

action = (intent.get("action") or "").lower()   # "buy" | "close"
symbol = (intent.get("symbol") or "").upper()
notional = float(intent.get("notional") or 0)

# ---- safety rails
cap = float(os.getenv("INTENT_CAP_USD","10000") or 10000)
allowlist = [s.strip().upper() for s in (os.getenv("INTENT_ALLOWLIST","") or "").split(",") if s.strip()]
allow_fractional = (os.getenv("ALLOW_FRACTIONAL","false").strip().lower() in ("1","true","yes","y","on"))

if action not in ("buy","close"): bad("action must be 'buy' or 'close'")
if not symbol: bad("missing symbol")

if allowlist and symbol not in allowlist:
    bad(f"{symbol} not in allowlist")

if action == "buy":
    if notional <= 0: bad("buy requires positive notional")
    if notional > cap: bad(f"notional>{cap} blocked")

    # asset guard
    a = requests.get(f"{base()}/v2/assets/{symbol}", headers=H(), timeout=20)
    if a.status_code != 200: bad(f"asset lookup failed {a.status_code}")
    aj = a.json() or {}
    if not aj.get("tradable", False): bad(f"{symbol} not tradable")
    if not aj.get("fractionable", False) and not allow_fractional:
        # you already run non-fractional; we keep that default
        pass

    od = {
        "symbol": symbol,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "notional": float(notional),
    }
    r = requests.post(f"{base()}/v2/orders", headers=H(), json=od, timeout=30)
    print("ORDER_BUY:", r.status_code, r.text)
    sys.exit(0 if r.status_code in (200,201) else 1)

if action == "close":
    r = requests.delete(f"{base()}/v2/positions/{symbol}", headers=H(), timeout=30)
    print("CLOSE_POS:", r.status_code, r.text)
    sys.exit(0 if r.status_code in (200,204) else 1)

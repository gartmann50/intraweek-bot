# tools/moc_flatten.py
#!/usr/bin/env python3
import os, sys, time, math, requests, datetime as dt
from zoneinfo import ZoneInfo

BASE = "https://paper-api.alpaca.markets" if os.getenv("ALPACA_ENV","paper").startswith("paper") else "https://api.alpaca.markets"
KEY  = os.getenv("ALPACA_KEY")
SEC  = os.getenv("ALPACA_SECRET")
if not KEY or not SEC:
    print("Alpaca not configured; exiting.")
    sys.exit(0)

H = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}
FORCE = "--force" in sys.argv

# Only run on Friday in New York unless forced
if not FORCE:
    now_ny = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if now_ny.weekday() != 4:
        print("Not Friday in New York—skipping MOC.")
        sys.exit(0)

def list_open_orders():
    r = requests.get(f"{BASE}/v2/orders", params={"status":"open","limit":500}, headers=H, timeout=20)
    return r.json() if r.status_code == 200 else []

def cancel_all_for(sym: str):
    """Cancel open orders for symbol and wait until they are gone."""
    hit = False
    for od in list_open_orders():
        if od.get("symbol") == sym:
            oid = od.get("id")
            try:
                requests.delete(f"{BASE}/v2/orders/{oid}", headers=H, timeout=15)
                hit = True
            except Exception:
                pass
    if hit:
        for _ in range(10):  # wait up to ~10s for “held_for_orders” to clear
            time.sleep(1.0)
            if all(od.get("symbol") != sym for od in list_open_orders()):
                break

# Fetch positions
r = requests.get(f"{BASE}/v2/positions", headers=H, timeout=20)
positions = r.json() if r.status_code == 200 and isinstance(r.json(), list) else []

placed, errors = 0, 0

for p in positions:
    sym = (p.get("symbol") or "").upper().strip()
    try:
        qty = float(p.get("qty") or 0.0)
    except Exception:
        continue
    if not sym or qty == 0:
        continue

    side = "sell" if qty > 0 else "buy"
    absq = abs(qty)
    int_q = math.floor(absq + 1e-8)        # integer part
    frac_q = absq - int_q                  # fractional remainder

    # 1) cancel working orders so qty is not “held”
    cancel_all_for(sym)

    # 2) close fractional now (must be DAY for fractionals)
    if frac_q > 1e-6:
        od = {"symbol": sym, "qty": frac_q, "side": side, "type": "market", "time_in_force": "day"}
        rr = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
        if rr.status_code != 200:
            print("fractional close error", sym, rr.status_code, rr.text[:220])
            errors += 1
        else:
            print(f"Closed fractional {sym}: {side} {frac_q:.6f} as DAY")
            time.sleep(0.5)

    # 3) close integer with MOC; if rejected, fallback to DAY now
    if int_q > 0:
        od = {"symbol": sym, "qty": int_q, "side": side, "type": "market", "time_in_force": "cls"}
        rr = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
        if rr.status_code != 200:
            print("MOC error", sym, rr.status_code, rr.text[:220], "-> fallback DAY")
            od["time_in_force"] = "day"
            rr2 = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
            if rr2.status_code != 200:
                print("fallback error", sym, rr2.status_code, rr2.text[:220])
                errors += 1
            else:
                placed += 1
        else:
            placed += 1

print(f"MOC placed (integer parts): {placed}, errors: {errors}")

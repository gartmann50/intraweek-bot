#!/usr/bin/env python3
import os, sys, time, math, argparse, requests, datetime as dt
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description="Flatten all positions. Default places MOC for integer part; --immediate closes all now.")
parser.add_argument("--force", action="store_true", help="Run even if today is not Friday (NY).")
parser.add_argument("--immediate", action="store_true",
                    help="Close integer part immediately with DAY market orders (no MOC).")
args = parser.parse_args()

BASE = "https://paper-api.alpaca.markets" if os.getenv("ALPACA_ENV","paper").startswith("paper") else "https://api.alpaca.markets"
KEY  = os.getenv("ALPACA_KEY")
SEC  = os.getenv("ALPACA_SECRET")
if not KEY or not SEC:
    print("Alpaca not configured; exiting.")
    sys.exit(0)

H = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}

# Only require Friday for scheduled runs; allow tests with --force
if not args.force:
    now_ny = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if now_ny.weekday() != 4:
        print("Not Friday in New York — skipping flatten.")
        sys.exit(0)

def list_open_orders():
    try:
        r = requests.get(f"{BASE}/v2/orders", params={"status":"open","limit":500},
                         headers=H, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def cancel_all_for(sym: str):
    """Cancel any working orders for symbol, then wait a moment for held qty to free."""
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
        for _ in range(10):  # wait up to ~10s for 'held_for_orders' flags to clear
            time.sleep(1)
            if all(od.get("symbol") != sym for od in list_open_orders()):
                break

# pull positions
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

    side = "sell" if qty > 0 else "buy"  # sell longs, buy to cover shorts
    absq = abs(qty)
    int_q = math.floor(absq + 1e-8)      # integer part
    frac_q = absq - int_q                # fractional remainder

    cancel_all_for(sym)

    # 1) close fractional remainder now (DAY market)
    if frac_q > 1e-6:
        od = {"symbol": sym, "qty": frac_q, "side": side, "type": "market", "time_in_force": "day"}
        rr = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
        if rr.status_code != 200:
            print("fractional close error", sym, rr.status_code, rr.text[:220])
            errors += 1
        else:
            print(f"Closed fractional {sym}: {side} {frac_q:.6f} as DAY")
            time.sleep(0.4)  # small pause

    # 2) close integer part
    if int_q > 0:
        if args.immediate:
            # close NOW
            od = {"symbol": sym, "qty": int_q, "side": side, "type": "market", "time_in_force": "day"}
            rr = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
            if rr.status_code != 200:
                print("DAY close error", sym, rr.status_code, rr.text[:220])
                errors += 1
            else:
                placed += 1
                print(f"Closed integer {sym}: {side} {int_q} as DAY")
        else:
            # schedule for close (preferred on Friday)
            od = {"symbol": sym, "qty": int_q, "side": side, "type": "market", "time_in_force": "cls"}
            rr = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
            if rr.status_code != 200:
                # fallback to DAY if MOC rejected
                print("MOC error", sym, rr.status_code, rr.text[:220], "-> fallback DAY")
                od["time_in_force"] = "day"
                rr2 = requests.post(f"{BASE}/v2/orders", headers=H, json=od, timeout=20)
                if rr2.status_code != 200:
                    print("fallback error", sym, rr2.status_code, rr2.text[:220])
                    errors += 1
                else:
                    placed += 1
                    print(f"Closed integer {sym}: {side} {int_q} as DAY (fallback)")
            else:
                placed += 1
                print(f"MOC placed for {sym}: {side} {int_q}")

mode = "IMMEDIATE (DAY)" if args.immediate else "MOC (int) + DAY (frac)"
print(f"Flatten mode: {mode} | integer orders placed: {placed}, errors: {errors}")

#!/usr/bin/env python3
import os, math, requests

def _base_url():
    return ("https://paper-api.alpaca.markets"
            if os.getenv("ALPACA_ENV","paper").startswith("paper")
            else "https://api.alpaca.markets")

def _headers():
    return {
        "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"]
    }

def _cancel_open_for_symbol(base, H, sym):
    r = requests.get(f"{base}/v2/orders", params={"status":"open","limit":500}, headers=H, timeout=20)
    if not r.ok: return
    for od in r.json():
        if od.get("symbol") == sym:
            oid = od.get("id")
            try: requests.delete(f"{base}/v2/orders/{oid}", headers=H, timeout=10)
            except: pass

def _place_mkt(base, H, sym, side, qty, tif):
    # qty can be fractional; Alpaca accepts string numbers best
    data = {"symbol": sym, "side": side, "type": "market", "time_in_force": tif, "qty": str(qty)}
    r = requests.post(f"{base}/v2/orders", headers=H, json=data, timeout=20)
    return r

def main():
    force = os.getenv("FORCE","false").lower() == "true"
    base  = _base_url()
    H     = _headers()

    r = requests.get(f"{base}/v2/positions", headers=H, timeout=20)
    if not r.ok:
        print("positions error", r.status_code, r.text)
        return

    pos = r.json() if isinstance(r.json(), list) else []
    if not pos:
        print("No positions to flatten.")
        return

    placed = errors = 0
    for p in pos:
        sym = (p.get("symbol") or "").upper()
        try:
            qty = float(p.get("qty") or 0)
        except:
            continue
        if not sym or qty == 0: 
            continue

        side = "sell" if qty > 0 else "buy"
        qty_abs = abs(qty)

        # Always cancel existing working orders for the symbol first
        _cancel_open_for_symbol(base, H, sym)

        if force:
            # Force = flatten NOW (market DAY), full quantity (ints + fractions)
            tif = "day"
            rr = _place_mkt(base, H, sym, side, qty_abs, tif)
            if rr.ok:
                print(f"FORCE close placed: {sym} {side} {qty_abs} {tif}")
                placed += 1
            else:
                print(f"FORCE close error {sym}: {rr.status_code} {rr.text[:240]}")
                errors += 1
        else:
            # Scheduled Friday run:
            # 1) Close the integer part with MOC
            int_qty = math.floor(qty_abs + 1e-8)
            frac    = qty_abs - int_qty

            if int_qty > 0:
                rr = _place_mkt(base, H, sym, side, int_qty, "cls")
                if rr.ok:
                    print(f"MOC placed: {sym} {side} {int_qty}")
                    placed += 1
                else:
                    print(f"MOC error {sym}: {rr.status_code} {rr.text[:240]}")
                    errors += 1

            # 2) Close fractional remainder with DAY (required by Alpaca)
            if frac > 1e-6:
                rr = _place_mkt(base, H, sym, side, frac, "day")
                if rr.ok:
                    print(f"Closed fractional: {sym} {side} {frac} DAY")
                    placed += 1
                else:
                    print(f"Fractional close error {sym}: {rr.status_code} {rr.text[:240]}")
                    errors += 1

    print(f"Flatten summary: placed={placed}, errors={errors}, force={force}")

if __name__ == "__main__":
    main()

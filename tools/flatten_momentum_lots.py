#!/usr/bin/env python3
import os, math, requests
import pandas as pd

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
    if not r.ok:
        return
    for od in r.json():
        if (od.get("symbol") or "").upper() == sym:
            oid = od.get("id")
            if oid:
                try:
                    requests.delete(f"{base}/v2/orders/{oid}", headers=H, timeout=15)
                except Exception:
                    pass

def _place_mkt(base, H, sym, side, qty, tif):
    data = {"symbol": sym, "side": side, "type": "market", "time_in_force": tif, "qty": str(qty)}
    return requests.post(f"{base}/v2/orders", headers=H, json=data, timeout=25)

def main():
    base = _base_url()
    H    = _headers()
    force = os.getenv("FORCE","false").lower() == "true"
    week_friday = os.getenv("WEEK_FRIDAY","").strip()
    lots_csv = os.getenv("LOTS_CSV","backtests/momentum_lots.csv")

    df = pd.read_csv(lots_csv)
    if week_friday:
        df = df[df["week_friday"].astype(str) == week_friday]

    # Aggregate qty per symbol (momentum lots)
    lots = df.groupby("symbol", as_index=False)["qty"].sum()
    if lots.empty:
        print("No momentum lots found for week; nothing to flatten.")
        return 0

    # Current positions map
    r = requests.get(f"{base}/v2/positions", headers=H, timeout=20)
    if not r.ok:
        print("positions error", r.status_code, r.text[:300])
        return 2
    pos_map = {}
    for p in (r.json() or []):
        sym = (p.get("symbol") or "").upper()
        try:
            qty = float(p.get("qty") or 0)
        except Exception:
            continue
        if sym and qty != 0:
            pos_map[sym] = qty

    placed = errors = 0
    for _, row in lots.iterrows():
        sym = str(row["symbol"]).upper()
        want = float(row["qty"])
        if want <= 0:
            continue

        cur = float(pos_map.get(sym, 0.0))
        if cur == 0:
            print(f"[skip] {sym}: no position (already flat).")
            continue

        # Sell only up to current long qty (this bot expects longs; extend if you short)
        qty_to_close = min(want, abs(cur))
        side = "sell" if cur > 0 else "buy"

        _cancel_open_for_symbol(base, H, sym)

        if force:
            rr = _place_mkt(base, H, sym, side, qty_to_close, "day")
            if rr.ok:
                print(f"FORCE close: {sym} {side} {qty_to_close} DAY")
                placed += 1
            else:
                print(f"FORCE error {sym}: {rr.status_code} {rr.text[:240]}")
                errors += 1
        else:
            # MOC requires integer qty
            int_qty = int(math.floor(qty_to_close + 1e-8))
            frac = qty_to_close - int_qty

            if int_qty > 0:
                rr = _place_mkt(base, H, sym, side, int_qty, "cls")
                if rr.ok:
                    print(f"MOC placed: {sym} {side} {int_qty}")
                    placed += 1
                else:
                    print(f"MOC error {sym}: {rr.status_code} {rr.text[:240]}")
                    errors += 1

            # If fractional exists (unlikely since we buy whole shares), close with DAY
            if frac > 1e-6:
                rr = _place_mkt(base, H, sym, side, frac, "day")
                if rr.ok:
                    print(f"Closed fractional: {sym} {side} {frac} DAY")
                    placed += 1
                else:
                    print(f"Fractional error {sym}: {rr.status_code} {rr.text[:240]}")
                    errors += 1

    print(f"Momentum flatten summary: placed={placed}, errors={errors}, force={force}")
    return 0 if errors == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())

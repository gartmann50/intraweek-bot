#!/usr/bin/env python3
import os, time, math, datetime as dt, requests, pandas as pd

# --------- Config (env) ----------
ALPACA_ENV   = os.getenv("ALPACA_ENV", "paper")
BASE_TRADER  = "https://paper-api.alpaca.markets" if ALPACA_ENV.startswith("paper") else "https://api.alpaca.markets"
POLY_KEY     = os.environ["POLYGON_API_KEY"]  # must be set
ATR_WIN      = int(float(os.getenv("ATR_WIN",  "14")))
ATR_MULT     = float(os.getenv("ATR_MULT", "1.75"))
LOOKBACK_D   = int(os.getenv("LOOKBACK_CAL_DAYS", "240"))  # calendar days for bars
RATE_SLEEP_S = float(os.getenv("POLY_SLEEP_SEC", "0.15"))  # small delay to respect rate limits

H_TRADE = {
    "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
}

# --------- Helpers ----------
def now_utc():
    return dt.datetime.now(dt.timezone.utc)

def polygon_bars(sym: str):
    """Fetch daily bars from Polygon (adjusted, asc)"""
    start = (now_utc() - dt.timedelta(days=LOOKBACK_D)).date().isoformat()
    end   = now_utc().date().isoformat()
    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}"
    params = {"adjusted":"true", "sort":"asc", "limit":"50000", "apiKey": POLY_KEY}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        return None, f"poly_http_{r.status_code}"
    js = r.json() or {}
    rows = js.get("results") or []
    if not rows:
        return None, "poly_empty"
    df = pd.DataFrame([{
        "date":  dt.datetime.utcfromtimestamp(int(x.get("t",0))/1000.0),
        "open":  x.get("o"),
        "high":  x.get("h"),
        "low":   x.get("l"),
        "close": x.get("c"),
    } for x in rows])
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    return df, None

def last_close_and_atr(df: pd.DataFrame, win: int):
    if df is None or len(df) < win + 2:
        return None, None, "too_few_bars"
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(win, min_periods=win).mean().iloc[-1]
    last_close = float(df["close"].iloc[-1])
    if not (pd.notna(atr) and pd.notna(last_close)):
        return None, None, "nan"
    return float(last_close), float(atr), None

def current_positions():
    r = requests.get(f"{BASE_TRADER}/v2/positions", headers=H_TRADE, timeout=20)
    r.raise_for_status()
    out = {}
    for p in r.json() or []:
        sym = str(p.get("symbol","")).upper()
        qty = float(p.get("qty") or 0.0)
        out[sym] = {"qty": qty, "side": "long" if qty > 0 else "short"}
    return out

def open_stop_orders_by_symbol():
    r = requests.get(f"{BASE_TRADER}/v2/orders",
                     params={"status":"open","limit":500},
                     headers=H_TRADE, timeout=20)
    r.raise_for_status()
    by = {}
    for od in r.json() or []:
        if od.get("type") in ("stop","stop_limit"):
            by.setdefault(od.get("symbol"),[]).append(od)
    return by

def better(curr, new, side):
    return new > curr if side == "long" else new < curr

def ensure_stop(sym, side, qty, last_close, atr, existing):
    if side == "long":
        ns = max(0.01, last_close - ATR_MULT * atr)
        stop_side = "sell"
    else:
        ns = last_close + ATR_MULT * atr
        stop_side = "buy"

    curr, oid = None, None
    for od in existing.get(sym, []):
        try:
            sp = od.get("stop_price") or od.get("limit_price")
            sp = float(sp) if sp is not None else None
        except Exception:
            sp = None
        if sp is not None:
            curr, oid = sp, od.get("id")
            break

    if curr is None or better(curr, ns, side):
        if oid:
            requests.delete(f"{BASE_TRADER}/v2/orders/{oid}", headers=H_TRADE, timeout=20)
        payload = {
            "symbol": sym,
            "qty": abs(float(qty)),
            "side": stop_side,
            "type": "stop",
            "time_in_force": "gtc",
            "stop_price": round(ns, 4),
        }
        r = requests.post(f"{BASE_TRADER}/v2/orders", headers=H_TRADE, json=payload, timeout=25)
        if r.status_code in (200, 201):
            print(f"[{'placed' if curr is None else 'tightened'}] {sym} stop={payload['stop_price']} side={stop_side} tif=gtc")
            return True
        print(f"[WARN] place fail {sym}: {r.status_code} {r.text}")
        return False
    else:
        print(f"[keep ] {sym} existing_stop={curr}")
        return True

# --------- Main ----------
def main():
    pos = current_positions()
    if not pos:
        print("No open positions; nothing to do.")
        return
    print("[stops] symbols:", sorted(pos.keys()))
    existing = open_stop_orders_by_symbol()

    placed = skipped = failed = 0
    for i, (sym, info) in enumerate(sorted(pos.items())):
        # light rate control for Polygon free tier (~5 req/s); we call ≤ positions
        if i: time.sleep(RATE_SLEEP_S)

        qty  = info["qty"]; side = info["side"]
        if qty == 0:
            print(f"- {sym}: qty=0; skip."); skipped += 1; continue

        df, perr = polygon_bars(sym)
        if df is None:
            print(f"- {sym}: polygon {perr}; skip."); skipped += 1; continue

        lc, atr, aerr = last_close_and_atr(df, ATR_WIN)
        if lc is None:
            print(f"- {sym}: {aerr}; skip."); skipped += 1; continue

        try:
            ok = ensure_stop(sym, side, qty, lc, atr, existing)
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            ok = False

        if ok: placed += 1
        else:  failed  += 1

    print(f"[stops] summary: placed={placed}, skipped={skipped}, failed={failed}")

if __name__ == "__main__":
    main()

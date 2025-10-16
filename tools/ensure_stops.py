#!/usr/bin/env python3
import os, glob, math, requests, pandas as pd

# --- Config / env
ALPACA_ENV  = os.getenv("ALPACA_ENV", "paper")
BASE_TRADER = "https://paper-api.alpaca.markets" if ALPACA_ENV.startswith("paper") else "https://api.alpaca.markets"
BASE_DATA   = "https://data.alpaca.markets"      # Alpaca market data host
FEED        = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' (paper/free) or 'sip' (paid)
DATA_DIR    = os.getenv("DATA_DIR", "stock_data_400")
ATR_WIN     = int(float(os.getenv("ATR_WIN", "14")))
ATR_MULT    = float(os.getenv("ATR_MULT", "1.75"))
LOOKBACK_D  = int(os.getenv("LOOKBACK_DAYS", "120"))  # bars to fetch if no CSV

H_TRADER = {
    "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"]
}
H_DATA = H_TRADER  # same headers for data API

def local_last_close_and_atr(sym: str):
    """Try local CSVs first."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{sym}*.csv")))
    for f in files[::-1]:
        try:
            df = pd.read_csv(f, usecols=[0,1,2,3,4], header=0)
            df = df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"})
            if len(df) < ATR_WIN + 2:
                continue
            return last_close_and_atr_from_df(df)
        except Exception:
            pass
    return None, None

def alpaca_bars(sym: str, limit: int = LOOKBACK_D):
    """Fetch daily bars from Alpaca data API."""
    # docs: GET /v2/stocks/{symbol}/bars
    url = f"{BASE_DATA}/v2/stocks/{sym}/bars"
    params = {
        "timeframe": "1Day",
        "limit": limit,
        "adjustment": "all",
        "feed": FEED
    }
    r = requests.get(url, headers=H_DATA, params=params, timeout=20)
    r.raise_for_status()
    j = r.json() or {}
    bars = j.get("bars", [])
    if not bars:
        return None
    # Normalize to a DataFrame that matches our ATR function
    df = pd.DataFrame([{
        "date":   b.get("t"),
        "open":   b.get("o"),
        "high":   b.get("h"),
        "low":    b.get("l"),
        "close":  b.get("c"),
        "volume": b.get("v")
    } for b in bars])
    # Make sure values are numeric
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def last_close_and_atr_from_df(df: pd.DataFrame):
    """Compute ATR from a standard OHLC df (daily)."""
    df = df.dropna(subset=["open","high","low","close"]).copy()
    if len(df) < ATR_WIN + 2:
        return None, None
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WIN, min_periods=ATR_WIN).mean().iloc[-1]
    last_close = float(df["close"].iloc[-1])
    if not math.isfinite(atr) or not math.isfinite(last_close):
        return None, None
    return last_close, float(atr)

def last_close_and_atr(sym: str):
    """Try local CSVs; else Alpaca data API."""
    lc, atr = local_last_close_and_atr(sym)
    if lc is not None and atr is not None:
        return lc, atr
    df = alpaca_bars(sym)
    if df is None:
        return None, None
    return last_close_and_atr_from_df(df)

def current_positions():
    r = requests.get(f"{BASE_TRADER}/v2/positions", headers=H_TRADER, timeout=20)
    r.raise_for_status()
    out = {}
    for p in r.json() or []:
        sym = str(p.get("symbol","")).upper()
        try:
            qty = float(p.get("qty") or 0)
        except Exception:
            qty = 0.0
        out[sym] = {"qty": qty, "side": "long" if qty > 0 else "short"}
    return out

def open_stop_orders_by_symbol():
    r = requests.get(f"{BASE_TRADER}/v2/orders", params={"status":"open","limit":500}, headers=H_TRADER, timeout=20)
    r.raise_for_status()
    by = {}
    for od in r.json() or []:
        if od.get("type") in ("stop","stop_limit"):
            by.setdefault(od.get("symbol"), []).append(od)
    return by

def better(curr, new, side):
    return new > curr if side == "long" else new < curr

def ensure_stop_for(sym, side, qty, last_close, atr, existing):
    if side == "long":
        ns = max(0.01, last_close - ATR_MULT * atr)
        stop_side = "sell"
    else:
        ns = last_close + ATR_MULT * atr
        stop_side = "buy"

    # If there is already a stop, check if we can tighten
    current = None
    curr_id = None
    for od in existing.get(sym, []):
        try:
            sp = od.get("stop_price") or od.get("limit_price")
            sp = float(sp) if sp is not None else None
        except Exception:
            sp = None
        if sp is not None:
            current = sp
            curr_id = od.get("id")
            break

    if current is None or better(current, ns, side):
        if curr_id:
            requests.delete(f"{BASE_TRADER}/v2/orders/{curr_id}", headers=H_TRADER, timeout=20)
        payload = {
            "symbol": sym,
            "qty": abs(float(qty)),
            "side": stop_side,
            "type": "stop",
            "time_in_force": "gtc",      # GTC so it persists
            "stop_price": round(ns, 4)
        }
        r = requests.post(f"{BASE_TRADER}/v2/orders", headers=H_TRADER, json=payload, timeout=20)
        if r.status_code not in (200, 201):
            print(f"[WARN] stop place fail {sym}: {r.status_code} {r.text}")
        else:
            print(f"[{'placed' if current is None else 'tightened'}] {sym} stop={payload['stop_price']} side={stop_side} tif=gtc")
    else:
        print(f"[keep ] {sym} existing_stop={current}")

def main():
    pos = current_positions()
    if not pos:
        print("No open positions; nothing to do.")
        return

    print("[stops] target symbols:", sorted(pos.keys()))
    existing = open_stop_orders_by_symbol()

    placed = skipped = failed = 0
    failed_syms = []

    for sym, info in sorted(pos.items()):
        qty = info["qty"]
        if qty == 0:
            skipped += 1
            continue
        lc, atr = last_close_and_atr(sym)
        if lc is None or atr is None:
            print(f"- {sym}: no bars; skip.")
            skipped += 1
            continue
        try:
            ensure_stop_for(sym, info["side"], qty, lc, atr, existing)
            placed += 1
        except Exception as e:
            print(f"[ERROR] {sym} {e}")
            failed += 1
            failed_syms.append(sym)

    print(f"[stops] summary: {{\"placed\": {placed}, \"skipped\": {skipped}, \"failed\": {failed}, \"symbols\": {failed_syms}}}")

if __name__ == "__main__":
    main()

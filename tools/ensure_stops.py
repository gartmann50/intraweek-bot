#!/usr/bin/env python3
import os, glob, math, datetime as dt, requests, pandas as pd

# ---------- Config ----------
ALPACA_ENV   = os.getenv("ALPACA_ENV", "paper")
BASE_TRADER  = "https://paper-api.alpaca.markets" if ALPACA_ENV.startswith("paper") else "https://api.alpaca.markets"
BASE_DATA    = "https://data.alpaca.markets"
FEED         = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' for paper, 'sip' for live
DATA_DIR     = os.getenv("DATA_DIR", "stock_data_400")
ATR_WIN      = int(float(os.getenv("ATR_WIN", "14")))
ATR_MULT     = float(os.getenv("ATR_MULT", "1.75"))
LOOKBACK_CAL = int(os.getenv("LOOKBACK_CAL_DAYS", "210"))   # calendar days for start window
POLY_KEY     = os.getenv("POLYGON_API_KEY", "")

H = {
    "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
}

# ---------- Helpers ----------
def now_utc():
    return dt.datetime.now(dt.timezone.utc)

def start_iso(days: int) -> str:
    return (now_utc() - dt.timedelta(days=days)).isoformat()

def local_last_close_and_atr(sym: str):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{sym}*.csv")))
    for f in files[::-1]:
        try:
            df = pd.read_csv(f, usecols=[0,1,2,3,4])
            df = df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"})
            if len(df) >= ATR_WIN + 2:
                return last_close_and_atr_from_df(df)
        except Exception:
            pass
    return None, None, "no_local_csv"

def alpaca_bars(sym: str, limit: int = 1000):
    """Alpaca v2 bars with explicit start window."""
    url = f"{BASE_DATA}/v2/stocks/{sym}/bars"
    params = {
        "timeframe": "1Day",
        "limit":     limit,
        "feed":      FEED,
        "adjustment":"all",
        "start":     start_iso(LOOKBACK_CAL),   # <- important!
    }
    r = requests.get(url, headers=H, params=params, timeout=25)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"
    j = r.json() or {}
    bars = j.get("bars", [])
    if not bars:
        return None, "empty"
    df = pd.DataFrame([{
        "date":  b.get("t"),
        "open":  b.get("o"),
        "high":  b.get("h"),
        "low":   b.get("l"),
        "close": b.get("c"),
        "vol":   b.get("v")
    } for b in bars])
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None

def polygon_bars(sym: str, limit: int = 1000):
    """Optional fallback to Polygon aggregates (1D)."""
    if not POLY_KEY:
        return None, "no_polygon_key"
    # 240 calendar days back to be safe
    start = (now_utc() - dt.timedelta(days=240)).date().isoformat()
    end   = now_utc().date().isoformat()
    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}"
    params = {"adjusted":"true", "sort":"asc", "limit":"50000", "apiKey": POLY_KEY}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        return None, f"poly_http_{r.status_code}"
    j = r.json() or {}
    rows = j.get("results", [])
    if not rows:
        return None, "poly_empty"
    df = pd.DataFrame([{
        "date":  dt.datetime.utcfromtimestamp(int(x.get("t",0))/1000.0),
        "open":  x.get("o"),
        "high":  x.get("h"),
        "low":   x.get("l"),
        "close": x.get("c")
    } for x in rows])
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None

def last_close_and_atr_from_df(df: pd.DataFrame):
    df = df.dropna(subset=["open","high","low","close"]).copy()
    if len(df) < ATR_WIN + 2:
        return None, None, "too_few_bars"
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WIN, min_periods=ATR_WIN).mean().iloc[-1]
    last_close = float(df["close"].iloc[-1])
    if not (pd.notna(atr) and pd.notna(last_close)):
        return None, None, "nan"
    return float(last_close), float(atr), None

def last_close_and_atr(sym: str):
    lc, atr, reason = local_last_close_and_atr(sym)
    if lc is not None and atr is not None:
        return lc, atr, "local_csv"
    df, err = alpaca_bars(sym)
    if df is not None:
        lc, atr, reason2 = last_close_and_atr_from_df(df)
        if lc is not None:
            return lc, atr, "alpaca"
        return None, None, reason2 or "alp_err"
    # optional fallback to Polygon
    df, err2 = polygon_bars(sym)
    if df is not None:
        lc, atr, reason2 = last_close_and_atr_from_df(df)
        if lc is not None:
            return lc, atr, "polygon"
        return None, None, reason2 or "poly_err"
    return None, None, f"no_data({err}|{err2})"

def current_positions():
    r = requests.get(f"{BASE_TRADER}/v2/positions", headers=H, timeout=20)
    r.raise_for_status()
    out = {}
    for p in r.json() or []:
        sym = str(p.get("symbol","")).upper()
        qty = float(p.get("qty") or 0.0)
        out[sym] = {"qty": qty, "side": "long" if qty > 0 else "short"}
    return out

def open_stop_orders_by_symbol():
    r = requests.get(f"{BASE_TRADER}/v2/orders",
                     params={"status":"open","limit":500}, headers=H, timeout=20)
    r.raise_for_status()
    by = {}
    for od in r.json() or []:
        if od.get("type") in ("stop","stop_limit"):
            by.setdefault(od.get("symbol"), []).append(od)
    return by

def better(curr, new, side):  # tighten?
    return new > curr if side == "long" else new < curr

def ensure_stop(sym, side, qty, last_close, atr, existing):
    if side == "long":
        ns = max(0.01, last_close - ATR_MULT * atr)
        stop_side = "sell"
    else:
        ns = last_close + ATR_MULT * atr
        stop_side = "buy"

    # inspect existing stop
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

    # (re)place if better
    if curr is None or better(curr, ns, side):
        if oid:
            requests.delete(f"{BASE_TRADER}/v2/orders/{oid}", headers=H, timeout=20)
        payload = {
            "symbol": sym,
            "qty": abs(float(qty)),
            "side": stop_side,
            "type": "stop",
            "time_in_force": "gtc",     # persist across days
            "stop_price": round(ns, 4),
        }
        r = requests.post(f"{BASE_TRADER}/v2/orders", headers=H, json=payload, timeout=25)
        if r.status_code in (200, 201):
            print(f"[{'placed' if curr is None else 'tightened'}] {sym} stop={payload['stop_price']} side={stop_side} tif=gtc")
            return True
        else:
            print(f"[WARN] stop place fail {sym}: {r.status_code} {r.text}")
            return False
    else:
        print(f"[keep ] {sym} existing_stop={curr}")
        return True

# ---------- Main ----------
def main():
    pos = current_positions()
    if not pos:
        print("No open positions; nothing to do.")
        return
    print("[stops] target symbols:", sorted(pos.keys()))

    existing = open_stop_orders_by_symbol()

    placed = skipped = failed = 0
    for sym, info in sorted(pos.items()):
        qty  = info["qty"]
        side = info["side"]
        if qty == 0:
            print(f"- {sym}: qty=0; skip.")
            skipped += 1
            continue

        lc, atr, src = last_close_and_atr(sym)
        if lc is None or atr is None:
            print(f"- {sym}: no bars; skip. (src={src})")
            skipped += 1
            continue

        ok = False
        try:
            ok = ensure_stop(sym, side, qty, lc, atr, existing)
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
        if ok:
            placed += 1
        else:
            failed += 1

    print(f"[stops] summary: {{\"placed\": {placed}, \"skipped\": {skipped}, \"failed\": {failed}}}")

if __name__ == "__main__":
    main()

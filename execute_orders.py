#!/usr/bin/env python3
"""
execute_orders.py — place Monday-open buys for Top-K picks (or sells), using Alpaca REST.
- Default: buy Top-K with equal *notional* (or share qty if notional unsupported).
- Designed to be called right after your "gate to open" step.
- Saves an orders log artifact at backtests/orders_<YYYYMMDD>.json
"""
import argparse, json, math, os, pathlib, sys
from datetime import datetime, timezone
import pandas as pd
import requests
import time  # for a short wait until fills

def last_close_from_csv(data_dir: pathlib.Path, sym: str) -> float:
    p = data_dir / f"{sym}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing csv {p}")
    df = pd.read_csv(p, usecols=[0,4])  # Date, Close
    if df.empty:
        raise ValueError(f"empty csv {p}")
    return float(df.iloc[-1, 1])

def topk_from_picklist(picklist: pathlib.Path, week: str, topk: int) -> list[str]:
    df = pd.read_csv(picklist)
    wk = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
    if not wk:
        raise SystemExit("picklist missing week column")
    df[wk] = pd.to_datetime(df[wk], errors="coerce").dt.date
    w = datetime.fromisoformat(week).date()
    sub = df[df[wk] == w].copy()
    if "rank" in sub.columns:
        sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce")
        sub = sub.sort_values(["rank","symbol"], ascending=[True,True])
    elif "score" in sub.columns:
        sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
        sub = sub.sort_values(["score","symbol"], ascending=[False,True])
    return [s.upper() for s in sub["symbol"].astype(str).head(topk).tolist()]

def alpaca_base(env: str) -> str:
    return "https://paper-api.alpaca.markets" if env.lower().startswith("paper") else "https://api.alpaca.markets"

def place_alpaca_order(base: str, key: str, secret: str, payload: dict) -> dict:
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Content-Type": "application/json"}
    r = requests.post(f"{base}/v2/orders", headers=h, data=json.dumps(payload), timeout=20)
    # Don’t explode on 4xx — record it
    try:
        j = r.json()
    except Exception:
        j = {"error": f"Non-JSON response status {r.status_code}"}
    if r.status_code >= 300:
        j.setdefault("error", f"HTTP {r.status_code}")
    return j

def atr14_from_csv(data_dir, sym):
    p = (data_dir / f"{sym}.csv")
    df = pd.read_csv(p).rename(columns=str.lower)
    for k in ("date","high","low","close"):
        if k not in df: raise ValueError(f"{p.name} missing {k}")
    df = df.sort_values("date")
    prev = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-prev).abs(),
                    (df["low"]-prev).abs()], axis=1).max(axis=1)
    a = tr.rolling(14, min_periods=14).mean().iloc[-1]
    return float(a) if pd.notna(a) else None

def wait_filled(base, key, secret, oid, timeout=120, interval=2):
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    t0 = time.time(); last = {}
    while time.time()-t0 < timeout:
        r = requests.get(f"{base}/v2/orders/{oid}", headers=h, timeout=15)
        try: j = r.json()
        except: j = {}
        last = j or last
        if (j.get("status","").lower() in {"filled","partially_filled","canceled","rejected","expired"}):
            return j
        time.sleep(interval)
    return last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", choices=["alpaca"], default="alpaca")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--week", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--side", choices=["buy","sell"], required=True)
    ap.add_argument("--notional", type=float, default=5000.0, help="USD per trade (fallback to qty if notional not allowed)")
    ap.add_argument("--qty", type=int, default=0, help="override share qty (if >0, ignores notional)")
    ap.add_argument("--init-stop-atr-mult", type=float, default=1.75,
                help="Entry-day stop = fill - ATR14*mult (DAY). If ATR missing, falls back to --init-stop-pct.")
    ap.add_argument("--init-stop-pct", type=float, default=0.10,
                help="Fallback: entry-day stop at pct below fill (DAY) if ATR unavailable.")
    
    
    # Alpaca creds come from env
    args = ap.parse_args()

    # inputs
    data_dir = pathlib.Path(args.data_dir)
    picks = topk_from_picklist(pathlib.Path(args.picklist), args.week, args.topk)

    # creds
    alp_key = os.getenv("ALPACA_KEY","")
    alp_secret = os.getenv("ALPACA_SECRET","")
    alp_env = os.getenv("ALPACA_ENV","paper")
    if not alp_key or not alp_secret:
        print("Missing ALPACA_KEY/ALPACA_SECRET env.", file=sys.stderr)
        sys.exit(1)
    base = alpaca_base(alp_env)

    # choose order params
    # We are called *after* the gate, so we’re at ~09:30:03 NY time → send MARKET DAY
    time_in_force = "day"
    order_type = "market"

    # build orders
    orders = []
    for sym in picks:
        body = {
            "symbol": sym,
            "side": args.side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if args.qty > 0:
            body["qty"] = str(args.qty)
        else:
            # prefer notional (fractional) if supported; else compute integer qty from last close
            try:
                last_close = last_close_from_csv(data_dir, sym)
                qty = max(1, int(args.notional // last_close))
                body["notional"] = str(int(args.notional))
                # If notional is rejected, we’ll retry with qty below.
                body["_qty_fallback"] = str(qty)
            except Exception:
                body["qty"] = "1"
        orders.append(body)

    results = []
    for body in orders:
        j = place_alpaca_order(base, alp_key, alp_secret, {k:v for k,v in body.items() if not k.startswith("_")})
        # Fallback if notional isn’t allowed
        if "error" in j and "notional" in json.dumps(j).lower() and "_qty_fallback" in body:
            qbody = {k:v for k,v in body.items() if not k.startswith("_")}
            qbody.pop("notional", None)
            qbody["qty"] = body["_qty_fallback"]
            j2 = place_alpaca_order(base, alp_key, alp_secret, qbody)
            j = {"first_attempt": j, "retry_qty": j2}
        results.append({"symbol": body["symbol"], "request": body, "response": j})

        if args.side == "buy" and (args.init_stop_atr_mult > 0 or args.init_stop_pct > 0):
          for row in results:
              sym = row["symbol"]
              oid = (row["response"].get("id") or row["response"].get("retry_qty",{}).get("id"))
              if not oid: 
                 print(f"{sym}: no order id; skip stop."); 
                 continue
        fill = wait_filled(base, alp_key, alp_secret, oid, timeout=90, interval=1)
        # qty + fill price
        qty = int(float(fill.get("filled_qty","0") or 0)) or get_position_qty(base, alp_key, alp_secret, sym)
        try: price = float(fill.get("filled_avg_price"))
        except: price = last_close_from_csv(data_dir, sym)
        if qty <= 0:
            print(f"{sym}: skip stop (qty=0)"); 
            continue
        # compute stop
        atr = None
        try: atr = atr14_from_csv(data_dir, sym)
        except Exception: pass
        if atr and atr > 0:
            stop_px = max(0.01, price - args.init_stop_atr_mult * atr)
            basis = f"ATR14*{args.init_stop_atr_mult}"
        else:
            stop_px = max(0.01, price * (1.0 - args.init_stop_pct))
            basis = f"{args.init_stop_pct*100:.1f}%"
        # submit DAY stop
        payload = {"symbol": sym, "side": "sell", "type": "stop",
                   "time_in_force": "day", "qty": str(qty),
                   "stop_price": f"{stop_px:.2f}"}
        s = place_alpaca_order(base, alp_key, alp_secret, payload)
        row["init_stop"] = {"qty": qty, "stop_price": round(stop_px,2), "basis": basis, "response": s}
        sid = s.get("id") or s.get("error","n/a")
        print(f"{sym}: DAY stop placed qty={qty} stop={stop_px:.2f} basis={basis} id={sid}")
    
    # log file
    outdir = pathlib.Path("backtests"); outdir.mkdir(exist_ok=True, parents=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = outdir / f"orders_{stamp}.json"
    out.write_text(json.dumps({"week": args.week, "side": args.side, "orders": results}, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(results)} orders.")
    # pretty print a summary
    for row in results:
        sym = row["symbol"]
        resp = row["response"]
        oid = resp.get("id") or resp.get("retry_qty",{}).get("id") or resp.get("error")
        status = resp.get("status") or resp.get("retry_qty",{}).get("status") or "error"
        print(f"{sym}: order_id={oid} status={status}")

if __name__ == "__main__":
    main()


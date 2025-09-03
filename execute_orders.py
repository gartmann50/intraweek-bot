#!/usr/bin/env python3
"""
execute_orders.py — place buys or sells using Alpaca REST.

- Buys: Top-K from picklist for a given week; equal notional (fallback to qty)
- Sells: pass --side sell and a symbol list if you extend; here we sell none.
- After buy fills, submits a DAY stop at max(fill - ATR14*mult, pct fallback).
- Logs to backtests/orders_<YYYYMMDD>.json
"""

import argparse, json, math, os, pathlib, sys, time
from datetime import datetime, timezone

import pandas as pd
import requests


# ---------- Helpers ----------

def alpaca_base(env: str) -> str:
    return ("https://paper-api.alpaca.markets"
            if env.lower().startswith("paper")
            else "https://api.alpaca.markets")


def place_alpaca_order(base: str, key: str, secret: str, payload: dict) -> dict:
    """POST /v2/orders with good error logging; always return a JSON-ish dict."""
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
         "Content-Type": "application/json"}
    r = requests.post(f"{base}/v2/orders", headers=h, json=payload, timeout=20)

    if r.status_code >= 400:
        # Show broker's actual message in the Actions logs
        print(f"Alpaca error {r.status_code}: {r.text[:500]}")

    try:
        j = r.json()
    except Exception:
        j = {"error": f"Non-JSON response status {r.status_code}", "body": r.text[:500]}

    if r.status_code >= 300:
        j.setdefault("error", f"HTTP {r.status_code}")

    return j


def wait_filled(base: str, key: str, secret: str, order_id: str,
                timeout: int = 120, interval: float = 1.5) -> dict:
    """Poll order status until filled / terminal or timeout; return last JSON."""
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        r = requests.get(f"{base}/v2/orders/{order_id}", headers=h, timeout=15)
        try:
            j = r.json()
        except Exception:
            j = {}
        if j:
            last = j
        st = (j.get("status") or "").lower()
        if st in {"filled", "partially_filled", "canceled", "rejected", "expired", "stopped"}:
            return j
        time.sleep(interval)
    return last


def get_position_qty(base: str, key: str, secret: str, symbol: str) -> int:
    """Return current position quantity for symbol; 0 if none."""
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    r = requests.get(f"{base}/v2/positions/{symbol}", headers=h, timeout=20)
    if r.status_code == 404:
        return 0
    try:
        r.raise_for_status()
        j = r.json()
        q = j.get("qty") or "0"
        return int(float(q))
    except Exception:
        return 0


def last_close_from_csv(data_dir: pathlib.Path, sym: str) -> float:
    p = data_dir / f"{sym}.csv"
    df = pd.read_csv(p, usecols=[0, 4])  # Date, Close
    if df.empty:
        raise ValueError(f"empty csv {p}")
    return float(df.iloc[-1, 1])


def atr14_from_csv(data_dir: pathlib.Path, sym: str):
    p = data_dir / f"{sym}.csv"
    df = pd.read_csv(p).rename(columns=str.lower)
    for k in ("date", "high", "low", "close"):
        if k not in df:
            raise ValueError(f"{p.name} missing {k}")
    df = df.sort_values("date")
    prev = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(),
                    (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    a = tr.rolling(14, min_periods=14).mean().iloc[-1]
    return float(a) if pd.notna(a) else None


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
        sub = sub.sort_values(["rank", "symbol"], ascending=[True, True])
    elif "score" in sub.columns:
        sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
        sub = sub.sort_values(["score", "symbol"], ascending=[False, True])
    return [s.upper() for s in sub["symbol"].astype(str).head(topk).tolist()]


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", choices=["alpaca"], default="alpaca")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--week", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--side", choices=["buy", "sell"], required=True)
    ap.add_argument("--notional", type=float, default=5000.0,
                   help="USD per trade (fallback to qty if notional rejected)")
    ap.add_argument("--qty", type=int, default=0,
                   help="override share qty (if >0, ignores notional)")
    ap.add_argument("--init-stop-atr-mult", type=float, default=1.75,
                   help="Entry-day stop = fill - ATR14*mult (DAY).")
    ap.add_argument("--init-stop-pct", type=float, default=0.10,
                   help="Fallback: pct below fill if ATR unavailable.")
    args = ap.parse_args()

    # Env/config
    alp_key = os.getenv("ALPACA_KEY", "")
    alp_secret = os.getenv("ALPACA_SECRET", "")
    alp_env = os.getenv("ALPACA_ENV", "paper")
    if not alp_key or not alp_secret:
        print("Missing ALPACA_KEY/ALPACA_SECRET.", file=sys.stderr)
        sys.exit(1)

    base = alpaca_base(alp_env)
    data_dir = pathlib.Path(args.data_dir)

    # Fill-wait behaviour (manual tests usually skip waiting)
    NO_WAIT = os.getenv("NO_WAIT_FOR_FILL", "0") == "1"
    FILL_TIMEOUT = int(os.getenv("FILL_TIMEOUT_SEC", "120"))
    FILL_POLL = float(os.getenv("FILL_POLL_SEC", "1.5"))

    picks = topk_from_picklist(pathlib.Path(args.picklist), args.week, args.topk)
    print(f"Picks (Top-{args.topk}) for WEEK={args.week}: {', '.join(picks)}")

    # Build the intended orders
    orders = []
    for sym in picks:
        body = {
            "symbol": sym,
            "side": args.side,
            "type": "market",
            "time_in_force": "day",
        }
        if args.qty > 0:
            body["qty"] = str(args.qty)
        else:
            # Prefer notional; keep qty fallback in a private key
            try:
                px = last_close_from_csv(data_dir, sym)
                qty_fb = max(1, int(args.notional // px))
            except Exception:
                px = None
                qty_fb = 1
            body["notional"] = str(int(args.notional))
            body["_qty_fallback"] = str(qty_fb)
        orders.append(body)

    results = []

    for body in orders:
        req = {k: v for k, v in body.items() if not k.startswith("_")}
        j = place_alpaca_order(base, alp_key, alp_secret, req)

        # If notional not allowed, retry with share qty
        if "error" in j and "notional" in json.dumps(j).lower() and "_qty_fallback" in body:
            qreq = {k: v for k, v in req.items() if k != "notional"}
            qreq["qty"] = body["_qty_fallback"]
            j2 = place_alpaca_order(base, alp_key, alp_secret, qreq)
            response = {"first_attempt": j, "retry_qty": j2}
            order_id = j2.get("id")
            order_status = j2.get("status")
        else:
            response = j
            order_id = j.get("id")
            order_status = j.get("status")

        row = {"symbol": body["symbol"], "request": req, "response": response}
        results.append(row)

        # Only place initial stop after a BUY and only when we have (or can infer) a fill
        if args.side != "buy":
            continue

        sym = body["symbol"]

        if not order_id:
            print(f"{sym}: no order id; skip stop.")
            continue

        fill = {}
        if NO_WAIT:
            print("NO_WAIT_FOR_FILL=1 — not waiting for fills; stop may be skipped if qty=0.")
        else:
            fill = wait_filled(base, alp_key, alp_secret, order_id,
                               timeout=FILL_TIMEOUT, interval=FILL_POLL)

        # Determine qty & price
        qty = int(float(fill.get("filled_qty", "0") or 0))
        if qty <= 0:
            # Fallback: ask positions (works a few seconds after fills post)
            qty = get_position_qty(base, alp_key, alp_secret, sym)

        if qty <= 0:
            print(f"{sym}: skip stop (qty=0)")
            continue

        try:
            price = float(fill.get("filled_avg_price"))
        except Exception:
            price = None
        if not price:
            try:
                price = last_close_from_csv(data_dir, sym)
            except Exception:
                price = None

        if not price:
            print(f"{sym}: no price available; skip stop.")
            continue

        # Compute stop price
        basis = ""
        try:
            atr = atr14_from_csv(data_dir, sym)
        except Exception:
            atr = None

        if atr and atr > 0 and args.init_stop_atr_mult > 0:
            stop_px = max(0.01, price - args.init_stop_atr_mult * atr)
            basis = f"ATR14*{args.init_stop_atr_mult}"
        else:
            stop_px = max(0.01, price * (1.0 - args.init_stop_pct))
            basis = f"{args.init_stop_pct*100:.1f}%"

        # Submit DAY stop order
        stop_payload = {
            "symbol": sym, "side": "sell", "type": "stop", "time_in_force": "day",
            "qty": str(qty), "stop_price": f"{stop_px:.2f}"
        }
        sresp = place_alpaca_order(base, alp_key, alp_secret, stop_payload)
        row["init_stop"] = {
            "qty": qty, "stop_price": round(stop_px, 2), "basis": basis, "response": sresp
        }
        sid = sresp.get("id") or sresp.get("error", "n/a")
        print(f"{sym}: DAY stop placed qty={qty} stop={stop_px:.2f} basis={basis} id={sid}")

    # Write log
    outdir = pathlib.Path("backtests"); outdir.mkdir(exist_ok=True, parents=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = outdir / f"orders_{stamp}.json"
    out.write_text(json.dumps({"week": args.week, "side": args.side, "orders": results}, indent=2),
                   encoding="utf-8")
    print(f"Wrote {out} with {len(results)} orders.")
    for row in results:
        sym = row["symbol"]
        resp = row["response"]
        # best-effort id + status for summary
        if isinstance(resp, dict) and "retry_qty" in resp:
            oid = resp.get("retry_qty", {}).get("id") or resp.get("first_attempt", {}).get("id")
            st = resp.get("retry_qty", {}).get("status") or resp.get("first_attempt", {}).get("status")
        else:
            oid = resp.get("id") or resp.get("error")
            st = resp.get("status") or "error"
        print(f"{sym}: order_id={oid} status={st}")


if __name__ == "__main__":
    main()

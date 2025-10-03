#!/usr/bin/env python3
"""
Create options snapshots (Polygon) for either:
  - 'picklist'  (default)  -> backtests/picklist_highrsi_trend.csv
  - 'hi70'                 -> backtests/hi70_thisweek.csv

Outputs a plain text file with ±10% OTM call/put near ~target-days expiry
for up to --topk symbols.

Examples:
  # picks
  python tools/options_snapshot_polygon.py \
    --mode picklist --input backtests/picklist_highrsi_trend.csv \
    --topk 10 --target-days 45 --pct 0.10 \
    --out backtests/options_snapshot.txt

  # 70D breakouts
  python tools/options_snapshot_polygon.py \
    --mode hi70 --input backtests/hi70_thisweek.csv \
    --topk 10 --target-days 45 --pct 0.10 \
    --out backtests/options_snapshot_hi70.txt

Requires env: POLYGON_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import pandas as pd


BASE = "https://api.polygon.io"


# ---------------- Utilities ----------------

def must_env(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        sys.exit(f"FATAL: {key} is empty")
    return v


def pget(path: str, params: Dict[str, str], tries: int = 5) -> Dict:
    last = None
    for k in range(tries):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(0.8 * (k + 1))
                continue
            r.raise_for_status()
            return r.json() or {}
        except Exception as e:
            last = e
            time.sleep(0.6 * (k + 1))
    raise RuntimeError(f"GET {path} failed ({last})")


# ---------------- Polygon helpers ----------------

def prev_close(sym: str, api: str) -> Optional[float]:
    try:
        j = pget(f"/v2/aggs/ticker/{sym}/prev", {"adjusted": "true", "apiKey": api})
        res = (j.get("results") or [])
        if res:
            return float(res[0].get("c", 0.0) or 0.0)
    except Exception:
        pass
    return None


def list_expiries(sym: str, api: str, horizon_days: int = 180) -> List[str]:
    start = date.today(); end = start + timedelta(days=horizon_days)
    params = {
        "underlying_ticker": sym,
        "expiration_date.gte": start.isoformat(),
        "expiration_date.lte": end.isoformat(),
        "limit": "1000",
        "order": "asc",
        "sort": "expiration_date",
        "apiKey": api,
    }
    exps: List[str] = []
    cursor = None
    while True:
        p = params.copy()
        if cursor: p["cursor"] = cursor
        j = pget("/v3/reference/options/contracts", p)
        for r in j.get("results", []) or []:
            d = r.get("expiration_date")
            if d and d not in exps: exps.append(d)
        nxt = j.get("next_url")
        if not nxt: break
        q = parse_qs(urlparse(nxt).query)
        cursor = (q.get("cursor") or [None])[0]
        if not cursor: break
        if len(exps) > 3000: break
    return exps


def nearest_expiry(exps: List[str], target_days: int) -> Optional[str]:
    if not exps: return None
    today = date.today()
    target = today + timedelta(days=target_days)
    best, best_abs = None, 10**9
    for e in exps:
        try: d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception: continue
        diff = abs((d - target).days)
        if diff < best_abs or (diff == best_abs and d >= today):
            best, best_abs = e, diff
    return best


def list_strikes_for(sym: str, exp: str, cp: str, api: str) -> List[float]:
    params = {
        "underlying_ticker": sym,
        "expiration_date": exp,
        "contract_type": "call" if cp.upper()=="C" else "put",
        "limit": "1000",
        "sort": "strike_price",
        "order": "asc",
        "apiKey": api,
    }
    out: List[float] = []
    cursor = None
    while True:
        p = params.copy()
        if cursor: p["cursor"] = cursor
        j = pget("/v3/reference/options/contracts", p)
        for r in j.get("results", []) or []:
            sp = r.get("strike_price")
            if sp is not None:
                try: out.append(float(sp))
                except Exception: pass
        nxt = j.get("next_url")
        if not nxt: break
        q = parse_qs(urlparse(nxt).query)
        cursor = (q.get("cursor") or [None])[0]
        if not cursor: break
        if len(out) > 5000: break
    return sorted(list(set(out)))


def nearest_otm_strike(last: float, strikes: List[float], pct: float, cp: str) -> Optional[float]:
    if not strikes or last is None: return None
    target = last * (1.0 + pct if cp.upper()=="C" else 1.0 - pct)
    if cp.upper()=="C":
        cands = [s for s in strikes if s >= target]
        return (min(cands, key=lambda s: abs(s-target)) if cands
                else min(strikes, key=lambda s: abs(s-target)))
    else:
        cands = [s for s in strikes if s <= target]
        return (min(cands, key=lambda s: abs(s-target)) if cands
                else min(strikes, key=lambda s: abs(s-target)))


def opt_ticker(sym: str, exp_yyyy_mm_dd: str, cp: str, strike: float) -> str:
    ymd = exp_yyyy_mm_dd.replace("-", "")
    k = int(round(strike * 1000))
    return f"O:{sym.upper()}{ymd}{cp.upper()}{k:08d}"


def last_nbbo(option: str, api: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        j = pget(f"/v2/last/nbbo/{option}", {"apiKey": api})
        q = j.get("results") or j.get("last") or {}
        b = q.get("bid") or q.get("P") or q.get("bP")
        a = q.get("ask") or q.get("p") or q.get("aP")
        b = float(b) if b is not None else None
        a = float(a) if a is not None else None
        if b is not None or a is not None: return b, a
    except Exception:
        pass
    # fallback
    try:
        j = pget(f"/v3/quotes/options/{option}", {"order":"desc","limit":"1","apiKey":api})
        res = (j.get("results") or [])
        if res:
            r = res[0]
            b = r.get("bid_price"); a = r.get("ask_price")
            return (float(b) if b is not None else None, float(a) if a is not None else None)
    except Exception:
        pass
    return None, None


def build_for_symbol(sym: str, pct: float, target_days: int, api: str) -> str:
    last = prev_close(sym, api)
    if not last:
        return f"{sym}: (underlying price unavailable)\n"

    exps = list_expiries(sym, api)
    exp = nearest_expiry(exps, target_days)
    if not exp:
        return f"{sym} (close {last:.2f}): no expiries found\n"

    strikes_c = list_strikes_for(sym, exp, "C", api)
    strikes_p = list_strikes_for(sym, exp, "P", api)

    sc = nearest_otm_strike(last, strikes_c, pct, "C")
    sp = nearest_otm_strike(last, strikes_p, pct, "P")

    out = [f"{sym}  close {last:.2f}  |  expiry {exp}  (~{target_days}d)"]

    def leg(cp: str, s: Optional[float]) -> str:
        if s is None: return f"  {cp}: (N/A)"
        t = opt_ticker(sym, exp, cp[-1], s) if cp in ("C","P") else opt_ticker(sym, exp, "C" if "CALL" in cp else "P", s)
        b, a = last_nbbo(t, api)
        mid = None if (b is None or a is None) else (b + a) / 2.0
        f = lambda x: "NA" if x is None else f"{x:.2f}"
        return f"  {cp} {s:.0f}: bid {f(b)}  ask {f(a)}  mid {f(mid)}  ({t})"

    out.append(leg("CALL +10%", sc))
    out.append(leg("PUT  -10%", sp))
    return "\n".join(out) + "\n"


# ---------------- Symbol selection ----------------

def symbols_from_picklist(path: str, topk: int) -> List[str]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    wk = None
    for c in ("week_start","week"):
        if c in df.columns: wk = c; break
    if not wk:
        return []
    latest = pd.to_datetime(df[wk], errors="coerce").dropna().dt.date.max()
    sel = df[pd.to_datetime(df[wk], errors="coerce").dt.date == latest].copy()
    if "rank" in sel.columns:
        sel = sel.sort_values(["rank","symbol"], ascending=[True,True])
    elif "score" in sel.columns:
        sel = sel.sort_values(["score","symbol"], ascending=[False,True])
    syms = (sel["symbol"].dropna().astype(str).str.upper().str.strip().tolist())
    uniq: List[str] = []
    for s in syms:
        if s not in uniq: uniq.append(s)
    return uniq[: max(1, topk)]


def symbols_from_hi70(path: str, topk: int) -> List[str]:
    """Top N symbols from hi70 CSV by market_cap desc (fallback: as-is)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    cols = {c.lower(): c for c in df.columns}
    sym = cols.get("symbol", "symbol")
    if "market_cap" in cols:
        df = df.sort_values(cols["market_cap"], ascending=False)
    syms = (df[sym].dropna().astype(str).str.upper().str.strip().tolist())
    uniq: List[str] = []
    for s in syms:
        if s not in uniq: uniq.append(s)
    return uniq[: max(1, topk)]


# ---------------- Main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["picklist","hi70"], default="picklist")
    ap.add_argument("--input", required=True, help="CSV path (picklist or hi70)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-days", type=int, default=45)
    ap.add_argument("--pct", type=float, default=0.10, help="OTM percent (0.10=10%)")
    args = ap.parse_args()

    api = must_env("POLYGON_API_KEY")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "hi70":
        syms = symbols_from_hi70(args.input, args.topk)
    else:
        syms = symbols_from_picklist(args.input, args.topk)

    if not syms:
        open(args.out, "w", encoding="utf-8").write("(no symbols to snapshot)\n")
        print(f"[opts] wrote {args.out} (no symbols)")
        return

    lines: List[str] = []
    for s in syms:
        try:
            lines.append(build_for_symbol(s, args.pct, args.target_days, api).rstrip())
        except Exception:
            lines.append(f"{s}: (snapshot unavailable)")
        time.sleep(0.35)  # respect rate limits a bit

    body = "\n".join(lines).rstrip() + "\n"
    open(args.out, "w", encoding="utf-8").write(body)
    print(f"[opts] wrote {args.out} for {len(syms)} symbols")


if __name__ == "__main__":
    main()

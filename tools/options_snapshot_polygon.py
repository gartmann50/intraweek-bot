#!/usr/bin/env python3
"""
Create an options snapshot for weekly picks using Polygon.

- Finds underlying last close via /v2/aggs/ticker/{SYM}/prev
- Finds expiries via /v3/reference/options/contracts
- Picks expiry nearest --target-days (default 45, good 30–60 window)
- Picks strikes ~ +10% (call) and ~ -10% (put) from underlying last close
- Fetches NBBO via /v2/last/nbbo/O:... (falls back to /v3/quotes/options/... if needed)
- Writes backtests/options_snapshot.txt for the email to include

Requires env: POLYGON_API_KEY
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import pandas as pd


BASE = "https://api.polygon.io"


def must_env(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        sys.exit(f"FATAL: {key} is empty")
    return v


def pget(path: str, params: Dict[str, str], tries: int = 5) -> Dict:
    """HTTP GET with light retry."""
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
    """Distinct expiration dates in the next horizon_days."""
    start = date.today()
    end = start + timedelta(days=horizon_days)

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
        if cursor:
            p["cursor"] = cursor
        j = pget("/v3/reference/options/contracts", p)
        for r in j.get("results", []) or []:
            d = r.get("expiration_date")
            if d and d not in exps:
                exps.append(d)
        nxt = j.get("next_url")
        if not nxt:
            break
        q = parse_qs(urlparse(nxt).query)
        cursor = (q.get("cursor") or [None])[0]
        if not cursor:
            break
        if len(exps) > 3000:
            break
    return exps


def nearest_expiry(exps: List[str], target_days: int) -> Optional[str]:
    if not exps:
        return None
    today = date.today()
    target = today + timedelta(days=target_days)
    best, best_abs = None, 10**9
    for e in exps:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        diff = abs((d - target).days)
        if diff < best_abs or (diff == best_abs and d >= today):
            best, best_abs = e, diff
    return best


def list_strikes_for(sym: str, exp: str, cp: str, api: str) -> List[float]:
    """All strikes for given expiry/side."""
    params = {
        "underlying_ticker": sym,
        "expiration_date": exp,
        "contract_type": "call" if cp.upper() == "C" else "put",
        "limit": "1000",
        "sort": "strike_price",
        "order": "asc",
        "apiKey": api,
    }
    out: List[float] = []
    cursor = None
    while True:
        p = params.copy()
        if cursor:
            p["cursor"] = cursor
        j = pget("/v3/reference/options/contracts", p)
        for r in j.get("results", []) or []:
            sp = r.get("strike_price")
            if sp is not None:
                try:
                    out.append(float(sp))
                except Exception:
                    pass
        nxt = j.get("next_url")
        if not nxt:
            break
        q = parse_qs(urlparse(nxt).query)
        cursor = (q.get("cursor") or [None])[0]
        if not cursor:
            break
        if len(out) > 5000:
            break
    return sorted(list(set(out)))


def nearest_otm_strike(last: float, strikes: List[float], pct: float, cp: str) -> Optional[float]:
    """Choose strike ~ last*(1+/-pct). For call pick >= target; for put pick <= target."""
    if not strikes or last is None:
        return None
    target = last * (1.0 + pct if cp.upper() == "C" else 1.0 - pct)
    if cp.upper() == "C":
        # first strike >= target (closest OTM on the upside)
        cands = [s for s in strikes if s >= target]
        if cands:
            return min(cands, key=lambda s: abs(s - target))
        # fallback: nearest overall
        return min(strikes, key=lambda s: abs(s - target))
    else:
        # first strike <= target (closest OTM on the downside)
        cands = [s for s in strikes if s <= target]
        if cands:
            return min(cands, key=lambda s: abs(s - target))
        return min(strikes, key=lambda s: abs(s - target))


def opt_ticker(sym: str, exp_yyyy_mm_dd: str, cp: str, strike: float) -> str:
    # Polygon format: O:{SYM}{YYYYMMDD}{C/P}{strike*1000:08d}
    ymd = exp_yyyy_mm_dd.replace("-", "")
    k = int(round(strike * 1000))
    return f"O:{sym.upper()}{ymd}{cp.upper()}{k:08d}"


def last_nbbo(option: str, api: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (bid, ask) via v2 last/nbbo; fallback to v3 latest quote."""
    try:
        j = pget(f"/v2/last/nbbo/{option}", {"apiKey": api})
        q = j.get("results") or j.get("last") or {}
        b = q.get("bid") or q.get("P") or q.get("bP")
        a = q.get("ask") or q.get("p") or q.get("aP")
        b = float(b) if b is not None else None
        a = float(a) if a is not None else None
        if b is not None or a is not None:
            return b, a
    except Exception:
        pass

    try:
        j = pget(f"/v3/quotes/options/{option}", {"order": "desc", "limit": "1", "apiKey": api})
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
        if s is None:
            return f"  {cp}: (N/A)"
        t = opt_ticker(sym, exp, cp, s)
        b, a = last_nbbo(t, api)
        mid = None
        if b is not None and a is not None:
            mid = (b + a) / 2.0
        def f(x): return "NA" if x is None else f"{x:.2f}"
        return f"  {cp} {s:.0f}: bid {f(b)}  ask {f(a)}  mid {f(mid)}  ({t})"

    out.append(leg("CALL +10%", sc))
    out.append(leg("PUT  -10%", sp))
    return "\n".join(out) + "\n"


def symbols_from_picklist(picklist: str, topk: int) -> List[str]:
    try:
        df = pd.read_csv(picklist)
    except Exception:
        return []
    wk = None
    for c in ("week_start", "week"):
        if c in df.columns:
            wk = c; break
    if not wk:
        return []
    latest = pd.to_datetime(df[wk], errors="coerce").dropna().dt.date.max()
    sel = df[pd.to_datetime(df[wk], errors="coerce").dt.date == latest].copy()
    if "rank" in sel.columns:
        sel = sel.sort_values(["rank", "symbol"], ascending=[True, True])
    elif "score" in sel.columns:
        sel = sel.sort_values(["score", "symbol"], ascending=[False, True])
    syms = (sel["symbol"].dropna().astype(str).str.upper().str.strip().tolist())
    uniq: List[str] = []
    for s in syms:
        if s not in uniq:
            uniq.append(s)
    return uniq[: max(1, topk)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picklist", required=True, help="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default="backtests/options_snapshot.txt")
    ap.add_argument("--target-days", type=int, default=45)
    ap.add_argument("--pct", type=float, default=0.10, help="OTM percent (0.10=10%%)")
    args = ap.parse_args()

    api = must_env("POLYGON_API_KEY")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    syms = symbols_from_picklist(args.picklist, args.topk)
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
        time.sleep(0.35)  # be nice

    body = "\n".join(lines).rstrip() + "\n"
    open(args.out, "w", encoding="utf-8").write(body)
    print(f"[opts] wrote {args.out} for {len(syms)} symbols")


if __name__ == "__main__":
    main()

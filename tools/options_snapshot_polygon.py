#!/usr/bin/env python3
"""
ROBUST options snapshot (Polygon) for:
  --mode picklist  -> backtests/picklist_highrsi_trend.csv (Top-K latest week)
  --mode hi70      -> backtests/hi70_thisweek.csv        (Top-K by market cap)

It now:
  • tries multiple expiries near target-days: 0, ±7, ±14, ±21, ±28
  • for each expiry, tries a ladder of strikes:
      Calls: +10%, +5%, ATM
      Puts : -10%, -5%, ATM
    (for very low-priced underlyings also tries small absolute steps)
  • if NBBO is missing, falls back to last trade (label “last=”)

ENV: POLYGON_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import pandas as pd


BASE = "https://api.polygon.io"


# -------------- utilities --------------

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


def fmt_price(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.2f}"


def ts_age_ms(ms: Optional[int]) -> str:
    if not ms:
        return ""
    try:
        t = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        delta = datetime.now(timezone.utc) - t
        days = delta.days
        secs = delta.seconds
        hh = secs // 3600
        mm = (secs % 3600) // 60
        if days > 0:
            return f"{days}d {hh:02d}h"
        return f"{hh:02d}h{mm:02d}m"
    except Exception:
        return ""


# -------------- Polygon helpers --------------

def prev_close(sym: str, api: str) -> Optional[float]:
    try:
        j = pget(f"/v2/aggs/ticker/{sym}/prev", {"adjusted": "true", "apiKey": api})
        res = (j.get("results") or [])
        if res:
            return float(res[0].get("c", 0.0) or 0.0)
    except Exception:
        pass
    return None


def list_expiries(sym: str, api: str, horizon_days: int = 210) -> List[str]:
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
        if len(exps) > 4000: break
    return exps


def nearest_expiry_candidates(exps: List[str], target_days: int) -> List[str]:
    """Return expiries sorted by closeness to target ±(0, 7, 14, 21, 28)."""
    today = date.today()
    target = today + timedelta(days=target_days)
    scored: List[Tuple[int, str]] = []
    for e in exps:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        diff = abs((d - target).days)
        scored.append((diff, e))
    scored.sort(key=lambda x: x[0])
    # Now apply window widening by days
    # We just return the 10 closest expiries; the build step will probe until it finds quotes
    return [e for _, e in scored[:10]]


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


def nearest_with_fallback(last: float, strikes: List[float], pct: float, cp: str) -> List[float]:
    """
    Return a TRY-LIST of strikes:
      Calls: +pct, +pct/2, ATM
      Puts : -pct, -pct/2, ATM
    For very low underlyings (< $10), also add small absolute steps.
    """
    if not strikes:
        return []
    atm = min(strikes, key=lambda s: abs(s - last))
    if cp.upper() == "C":
        tgt1 = last * (1.0 + pct)
        tgt2 = last * (1.0 + pct / 2.0)
        cands = [min(strikes, key=lambda s: abs(s - tgt1)),
                 min(strikes, key=lambda s: abs(s - tgt2)),
                 atm]
    else:
        tgt1 = last * (1.0 - pct)
        tgt2 = last * (1.0 - pct / 2.0)
        cands = [min(strikes, key=lambda s: abs(s - tgt1)),
                 min(strikes, key=lambda s: abs(s - tgt2)),
                 atm]

    # for very low-priced names, also try small absolute steps
    if last < 10:
        step = 0.5 if last >= 5 else 0.25
        if cp.upper()=="C":
            tgt3 = last + step
        else:
            tgt3 = max(0.5, last - step)
        cands.append(min(strikes, key=lambda s: abs(s - tgt3)))

    # keep unique order
    seen = set(); ordered: List[float] = []
    for s in cands:
        if s not in seen:
            ordered.append(s); seen.add(s)
    return ordered


def opt_ticker(sym: str, exp_yyyy_mm_dd: str, cp: str, strike: float) -> str:
    ymd = exp_yyyy_mm_dd.replace("-", "")
    k = int(round(strike * 1000))
    return f"O:{sym.upper()}{ymd}{cp.upper()}{k:08d}"


def get_nbbo(option: str, api: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Return (bid, ask, ts_ms) from last NBBO; ts may be None."""
    try:
        j = pget(f"/v2/last/nbbo/{option}", {"apiKey": api})
        q = j.get("results") or j.get("last") or {}
        b = q.get("bid") or q.get("P") or q.get("bP")
        a = q.get("ask") or q.get("p") or q.get("aP")
        t = q.get("t") or q.get("sip_timestamp")
        b = float(b) if b is not None else None
        a = float(a) if a is not None else None
        t = int(t) if t is not None else None
        return b, a, t
    except Exception:
        return None, None, None


def get_last_trade(option: str, api: str) -> Tuple[Optional[float], Optional[int]]:
    """Return (last_price, ts_ms) from most recent trade if available."""
    try:
        j = pget(f"/v3/trades/options/{option}", {"order": "desc", "limit": "1", "apiKey": api})
        res = (j.get("results") or [])
        if res:
            r = res[0]
            px = r.get("price")
            ts = r.get("sip_timestamp") or r.get("t")
            return (float(px) if px is not None else None,
                    int(ts) if ts is not None else None)
    except Exception:
        pass
    return None, None


# -------------- symbol selection --------------

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


# -------------- builder --------------

def try_leg(sym: str, exp: str, cp: str, strike: float, api: str) -> str:
    opt = opt_ticker(sym, exp, cp, strike)
    b, a, qts = get_nbbo(opt, api)
    if b is not None or a is not None:
        mid = None if (b is None or a is None) else (b + a) / 2.0
        age = ts_age_ms(qts)
        age_s = f"  ({age})" if age else ""
        return f"{cp} {strike:.0f}: bid {fmt_price(b)} ask {fmt_price(a)} mid {fmt_price(mid)} {age_s} ({opt})"
    # fallback to last trade
    lp, lts = get_last_trade(opt, api)
    if lp is not None:
        age = ts_age_ms(lts)
        age_s = f"  ({age})" if age else ""
        return f"{cp} {strike:.0f}: last {fmt_price(lp)}{age_s} ({opt})"
    return f"{cp} {strike:.0f}: bid NA ask NA mid NA ({opt})"


def build_for_symbol(sym: str, pct: float, target_days: int, api: str) -> str:
    last = prev_close(sym, api)
    if not last:
        return f"{sym}: (underlying price unavailable)\n"

    exps = list_expiries(sym, api)
    if not exps:
        return f"{sym} (close {last:.2f}): no expiries found\n"

    exp_candidates = nearest_expiry_candidates(exps, target_days)

    chosen_exp = None
    out_call = out_put = None

    for exp in exp_candidates:
        strikes_c = list_strikes_for(sym, exp, "C", api)
        strikes_p = list_strikes_for(sym, exp, "P", api)
        if not strikes_c and not strikes_p:
            continue

        call_try = nearest_with_fallback(last, strikes_c, pct, "C") if strikes_c else []
        put_try  = nearest_with_fallback(last, strikes_p, pct, "P") if strikes_p else []

        # choose first strike that yields any informative quote (NBBO or last)
        def informative(s: str) -> bool:
            return ("last " in s) or ("bid " in s and "NA" not in s)

        c_line = p_line = None
        for s in call_try:
            txt = try_leg(sym, exp, "C", s, api)
            c_line = txt
            if informative(txt): break
        for s in put_try:
            txt = try_leg(sym, exp, "P", s, api)
            p_line = txt
            if informative(txt): break

        # accept this expiry if at least one leg is informative
        if (c_line and ("last " in c_line or "mid " in c_line or "bid " in c_line and "NA" not in c_line)) or \
           (p_line and ("last " in p_line or "mid " in p_line or "bid " in p_line and "NA" not in p_line)):
            chosen_exp = exp
            out_call, out_put = c_line, p_line
            break
        # otherwise keep last tried (so we still print something)
        if not chosen_exp:
            chosen_exp = exp
            out_call, out_put = c_line, p_line

    header = f"{sym}  close {last:.2f}  |  expiry {chosen_exp or 'N/A'}  (~{target_days}d)"
    lines = [header]
    if out_call: lines.append("  " + out_call)
    if out_put:  lines.append("  " + out_put)
    return "\n".join(lines) + "\n"


# -------------- main --------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["picklist","hi70"], default="picklist")
    ap.add_argument("--input", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-days", type=int, default=45)
    ap.add_argument("--pct", type=float, default=0.10)
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
        time.sleep(0.35)

    body = "\n".join(lines).rstrip() + "\n"
    open(args.out, "w", encoding="utf-8").write(body)
    print(f"[opts] wrote {args.out} for {len(syms)} symbols")


if __name__ == "__main__":
    main()

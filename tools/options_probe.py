#!/usr/bin/env python3
"""
Quick options snapshot for one underlying (default AAPL).
- Tries Polygon first (POLYGON_API_KEY); falls back to Alpaca (ALPACA_KEY/ALPACA_SECRET).
- Picks ~target_dte expiry (>= target), and strikes ~±otm_pct from last close.
- Prints a tiny two-line snapshot for CALL/PUT.

Runtime: ~2–10s (single underlying, few HTTP calls).
"""

import os, sys, math, time
from datetime import datetime, timedelta, timezone
import requests

UNDER = os.getenv("UNDER", "AAPL")
OTM_PCT = float(os.getenv("OTM_PCT", "0.10"))   # 10% OTM
TARGET_DTE = int(os.getenv("TARGET_DTE", "45")) # ~45 days

def pct(a,b): 
    try: return 100*(a/b-1)
    except: return float("nan")

def iso(d): return d.strftime("%Y-%m-%d")

# ----------------- Polygon helpers -----------------
def poly_prev_close(sym, key):
    r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev",
                     params={"adjusted":"true","apiKey":key}, timeout=15)
    if r.status_code!=200: return None
    j=r.json() or {}
    res=(j.get("results") or [{}])[0]
    return float(res.get("c") or 0) or None

def poly_contracts(sym, key):
    # active, unexpired contracts
    out=[]
    cursor=None
    base="https://api.polygon.io/v3/reference/options/contracts"
    while True:
        p={"underlying_ticker":sym,"expired":"false","limit":"1000","apiKey":key}
        if cursor: p["cursor"]=cursor
        r=requests.get(base,params=p,timeout=20)
        if r.status_code!=200: break
        j=r.json() or {}
        out.extend(j.get("results") or [])
        cursor=(j.get("next_url") or "").split("cursor=")[-1] if j.get("next_url") else None
        if not cursor: break
    return out

def poly_best_quote(opt, key):
    # Try last quote (faster); fallback to snapshot if needed
    for url in (f"https://api.polygon.io/v3/quotes/options/{opt}/last",
                f"https://api.polygon.io/v3/snapshot/options/{opt}"):
        r = requests.get(url, params={"apiKey":key}, timeout=15)
        if r.status_code==200:
            j=r.json() or {}
            if "last" in j and "ask" in j["last"] and "bid" in j["last"]:
                q=j["last"]; return float(q.get("bid") or 0), float(q.get("ask") or 0)
            if "results" in j and "last_quote" in j["results"]:
                q=j["results"]["last_quote"] or {}
                if "bid_price" in q and "ask_price" in q:
                    return float(q.get("bid_price") or 0), float(q.get("ask_price") or 0)
    return None, None

# ----------------- Alpaca helpers -----------------
def alp_base():
    return "https://paper-api.alpaca.markets"  # we only need market data endpoints host root

def alp_headers():
    return {"APCA-API-KEY-ID": os.getenv("ALPACA_KEY",""), "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET","")}

def alp_prev_close(sym):
    # options data lives on data API, but for prev close we can use stocks data v2 last quote/agg.
    r=requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/trades/latest", headers=alp_headers(), timeout=15)
    if r.status_code!=200: return None
    # If latest trade price missing, fallback to bars
    j=r.json() or {}
    p=j.get("trade",{}).get("p")
    if p: return float(p)
    r2=requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars?timeframe=1Day&limit=2", headers=alp_headers(), timeout=15)
    if r2.status_code!=200: return None
    jj=r2.json() or {}
    bars=jj.get("bars") or []
    if bars: return float(bars[-1].get("c") or 0) or None
    return None

def alp_options_sym(sym, expiry, strike, callput):
    # OCC-like root per Alpaca: e.g. AAPL 2025-11-21 190 C => AAPL231121C00190000 (but Alpaca also accepts human query)
    # We’ll use snapshot search by params instead of constructing exact OCC string.
    return {"underlying_symbol":sym, "expiration_date":expiry, "strike_price":strike, "contract_type":"call" if callput=="C" else "put"}

def alp_best_quote(sym, expiry, strike, cp):
    # options snapshots search (v1beta1) — requires data options entitlement
    params={"underlying_symbol": sym, "expiration_date": expiry, "strike_price": strike, "contract_type": "call" if cp=="C" else "put"}
    r=requests.get("https://data.alpaca.markets/v1beta1/options/snapshots", params=params, headers=alp_headers(), timeout=15)
    if r.status_code!=200: return None, None
    j=r.json() or {}
    snaps=j.get("snapshots") or []
    if not snaps: return None, None
    q=snaps[0].get("latest_quote") or {}
    bid=q.get("bp"); ask=q.get("ap")
    return (float(bid) if bid is not None else None, float(ask) if ask is not None else None)

# ----------------- Main -----------------
def main():
    sym=UNDER.upper()
    poly=os.getenv("POLYGON_API_KEY","").strip()
    has_poly=bool(poly)
    has_alp=bool(os.getenv("ALPACA_KEY") and os.getenv("ALPACA_SECRET"))

    last=None
    if has_poly:
        last=poly_prev_close(sym, poly)
    if last is None and has_alp:
        last=alp_prev_close(sym)
    if last is None:
        print(f"{sym}: couldn’t fetch last price (need POLYGON_API_KEY or Alpaca keys).")
        sys.exit(0)

    target_up = last*(1+OTM_PCT)
    target_dn = last*(1-OTM_PCT)
    today=datetime.now(timezone.utc).date()
    target_exp = today + timedelta(days=TARGET_DTE)

    if has_poly:
        contracts=poly_contracts(sym, poly)
        # group by expiry, pick first expiry >= target
        expirs=sorted({(c.get("expiration_date") or "") for c in contracts if c.get("expiration_date")}, key=str)
        pick_exp=None
        for e in expirs:
            try:
                if datetime.fromisoformat(e).date() >= target_exp:
                    pick_exp=e; break
            except: 
                pass
        if not pick_exp and expirs:
            pick_exp=expirs[-1]
        # find closest strikes to ±10%
        calls=[c for c in contracts if (c.get("expiration_date")==pick_exp and (c.get("contract_type") or "").upper()=="CALL")]
        puts =[c for c in contracts if (c.get("expiration_date")==pick_exp and (c.get("contract_type") or "").upper()=="PUT")]
        def closest_strike(lst, tgt, side):
            best=None; bd=1e18
            for c in lst:
                try:
                    k=float(c.get("strike_price") or 0)
                except: 
                    continue
                # enforce OTM direction
                if side=="C" and k < last: continue
                if side=="P" and k > last: continue
                d=abs(k-tgt)
                if d<bd: bd=d; best=c
            return best
        c=closest_strike(calls, target_up, "C")
        p=closest_strike(puts , target_dn, "P")

        print(f"{sym} close {last:.2f} | expiry {pick_exp or '-'} (~{TARGET_DTE}d)")
        if c:
            ot=c.get("ticker") or c.get("id") or "?"
            k=float(c.get("strike_price") or 0)
            b,a=poly_best_quote(ot, poly)
            mid = ((b or 0)+(a or 0))/2 if (b and a) else None
            print(f"CALL +{int(OTM_PCT*100)}% {int(round(k))}: bid {b if b is not None else 'NA'} ask {a if a is not None else 'NA'} mid {mid if mid else 'NA'} ({ot})")
        else:
            print("CALL leg: no contract found near +10% OTM")

        if p:
            ot=p.get("ticker") or p.get("id") or "?"
            k=float(p.get("strike_price") or 0)
            b,a=poly_best_quote(ot, poly)
            mid = ((b or 0)+(a or 0))/2 if (b and a) else None
            print(f"PUT  -{int(OTM_PCT*100)}% {int(round(k))}: bid {b if b is not None else 'NA'} ask {a if a is not None else 'NA'} mid {mid if mid else 'NA'} ({ot})")
        else:
            print("PUT leg: no contract found near -10% OTM")
        return

    # Fallback Alpaca only
    exp = (today + timedelta(days=TARGET_DTE)).isoformat()
    # Alpaca needs exact expiry; we’ll approximate with Fridays (+FWIW, this is just a probe)
    # round to next Friday ~TARGET_DTE
    while datetime.fromisoformat(exp).date().weekday()!=4:
        exp = (datetime.fromisoformat(exp).date()+timedelta(days=1)).isoformat()

    c_k = int(round(target_up))
    p_k = int(round(target_dn))
    cb,ca = alp_best_quote(sym, exp, c_k, "C")
    pb,pa = alp_best_quote(sym, exp, p_k, "P")
    print(f"{sym} close {last:.2f} | expiry {exp} (~{TARGET_DTE}d)")
    print(f"CALL +{int(OTM_PCT*100)}% {c_k}: bid {cb or 'NA'} ask {ca or 'NA'}")
    print(f"PUT  -{int(OTM_PCT*100)}% {p_k}: bid {pb or 'NA'} ask {pa or 'NA'}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)

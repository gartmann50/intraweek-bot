#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send the weekly summary email (weekly picks + 70D highs + options snapshot).

Finds (recursively):
  picks text: backtests/top6_preview.txt OR from picklist CSV
  70D digest: backtests/hi70_digest.txt
Then builds a short options snapshot for up to N liquid names:
  - union of Top-K picks + Top-10 70D
  - re-ranked by market cap (Polygon)
  - pick nearest monthly expiry around ~45 days (fallback to next up)
  - quote ±10% OTM strikes (robust fallbacks; 'NA' if no quote)

SMTP config:
  YAML (config_notify.yaml / .yml) or env:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO
    SMTP_TLS (true/false, default true), SUBJECT_PREFIX (optional)
Also needs POLYGON_API_KEY in env for options snapshot (if missing, snapshot is skipped).
"""

from __future__ import annotations
import os, sys, glob, smtplib, ssl, math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import glob
from email.mime.text import MIMEText

# optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# optional pandas just to read CSVs reliably
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # we can still send email without CSV reading

# -------------- file helpers --------------
def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def find_picks_file_text() -> Tuple[str, Optional[str]]:
    # explicit text first
    exacts = ["backtests/top6_preview.txt", "backtests/weekly-picks.txt"]
    for p in exacts:
        if os.path.isfile(p):
            return (read_text(p).strip() or "(picks file is empty)", p)
    # derive from CSV if present
    if pd is not None:
        pick_csvs = ["backtests/picklist_highrsi_trend.csv", *glob.glob("**/picklist_highrsi_trend.csv", recursive=True)]
        for pc in pick_csvs:
            if not os.path.isfile(pc): continue
            try:
                df = pd.read_csv(pc)
                wk = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
                if not wk: continue
                week = str(pd.to_datetime(df[wk], errors="coerce").dropna().dt.date.max())
                if "rank" in df.columns:
                    df = df.sort_values(["rank","symbol"], ascending=[True, True])
                elif "score" in df.columns:
                    df = df.sort_values(["score","symbol"], ascending=[False, True])
                picks = (df[pd.to_datetime(df[wk], errors="coerce").dt.date.astype(str)==week]["symbol"]
                            .dropna().astype(str).head(6).tolist())
                if picks:
                    return ("Top-6: " + ", ".join(picks), pc)
            except Exception:
                pass
    # finally wildcard text
    patterns = [
        "backtests/weekly-picks-*.txt","backtests/weekly_picks_*.txt",
        "**/weekly-picks-*.txt","**/weekly_picks_*.txt","**/top6_preview.txt",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        for p in hits:
            return (read_text(p).strip() or "(picks file is empty)", p)
    return ("(no weekly picks file found)", None)

def find_hi70_digest_text() -> Tuple[str, Optional[str]]:
    for p in ["backtests/hi70_digest.txt","hi70_digest.txt"]:
        if os.path.isfile(p):
            return (read_text(p).strip() or "(hi70 digest empty)", p)
    hits = sorted(glob.glob("**/hi70_digest.txt", recursive=True))
    for p in hits:
        return (read_text(p).strip() or "(hi70 digest empty)", p)
    return ("(no hi70 digest found)", None)

# -------------- email config --------------
def load_config() -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    for y in ("config_notify.yaml","config_notify.yml"):
        if os.path.isfile(y) and yaml is not None:
            try:
                data = yaml.safe_load(open(y, "r", encoding="utf-8")) or {}
                if isinstance(data, dict):
                    for k,v in data.items():
                        cfg[k] = ("true" if isinstance(v,bool) and v else "false") if isinstance(v,bool) else str(v)
            except Exception:
                pass
            break
    env_map = {
        "SMTP_HOST":"smtp_host","SMTP_PORT":"smtp_port","SMTP_USER":"smtp_user","SMTP_PASS":"smtp_pass",
        "SMTP_FROM":"smtp_from","SMTP_TO":"smtp_to","SMTP_TLS":"smtp_tls","SUBJECT_PREFIX":"subject_prefix",
    }
    for ek, ck in env_map.items():
        if ck not in cfg or not cfg[ck]:
            val = os.getenv(ek, "")
            if val: cfg[ck] = val
    cfg.setdefault("smtp_tls","true")
    cfg.setdefault("subject_prefix","[Weekly]")
    return cfg

def parse_bool(s: str) -> bool:
    return str(s or "").strip().lower() in {"1","true","yes","y","on"}

def split_recipients(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").replace(";",",").split(","):
        p = part.strip()
        if p: out.append(p)
    return out

# -------------- Polygon helpers --------------
def http_get_json(url: str, params: Dict[str, str], max_tries: int = 5) -> Dict:
    last = None
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time = min(0.5*(2**k), 4.0)
                import time as _t; _t.sleep(time)
                continue
            r.raise_for_status()
            return r.json() or {}
        except Exception as e:
            last = e
    raise RuntimeError(f"GET {url} failed ({last})")

def mc_for_symbols(key: str, syms: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in syms:
        try:
            j = http_get_json(f"https://api.polygon.io/v3/reference/tickers/{s}",
                              {"apiKey": key}, max_tries=3)
            out[s] = float((j.get("results") or {}).get("market_cap") or 0.0)
        except Exception:
            out[s] = 0.0
    return out

def prev_close(key: str, sym: str) -> Optional[float]:
    try:
        j = http_get_json(f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev",
                          {"adjusted":"true","apiKey": key}, max_tries=3)
        r = (j.get("results") or [{}])[0]
        return float(r.get("c") or 0.0) or None
    except Exception:
        return None

def choose_expiry(key: str, sym: str, target_days: int = 45) -> Optional[str]:
    """Pick the nearest expiry around ~target_days (prefer 30-60d, else next)."""
    try:
        j = http_get_json("https://api.polygon.io/v3/reference/options/contracts",
                          {"underlying_ticker": sym, "expired":"false", "limit":"1000", "apiKey": key}, 4)
        exps = sorted({(r.get("expiration_date") or "") for r in (j.get("results") or []) if r.get("expiration_date")},)
        if not exps: return None
        today = datetime.utcnow().date()
        candidates = []
        for e in exps:
            try:
                d = datetime.strptime(e,"%Y-%m-%d").date()
                delta = (d - today).days
                if delta >= 5: candidates.append((abs(delta-target_days), delta, e))
            except Exception:
                continue
        if not candidates: return exps[0]
        # prefer window 30..60 days if possible
        inwin = [c for c in candidates if 30 <= c[1] <= 60]
        chosen = min(inwin or candidates, key=lambda x: x[0])
        return chosen[2]
    except Exception:
        return None

def strike_increment(price: float) -> float:
    if price < 5: return 0.5
    if price < 25: return 1.0
    if price < 100: return 2.5
    if price < 200: return 5.0
    return 10.0

def round_to_grid(x: float, inc: float) -> float:
    return round(round(x / inc) * inc, 3)

def encode_strike_1c(x: float) -> str:
    # Polygon uses 8 digits, strike * 1000 (e.g., 100.00 -> 00100000)
    return f"{int(round(x * 1000)):08d}"

def last_quote_for_option(key: str, opt_ticker: str) -> Tuple[Optional[float], Optional[float]]:
    # Try v3 first
    try:
        j = http_get_json(f"https://api.polygon.io/v3/quotes/options/{opt_ticker}/last",
                          {"apiKey": key}, 3)
        q = (j.get("results") or {})
        bid = q.get("bid_price"); ask = q.get("ask_price")
        if bid is not None or ask is not None:
            return (float(bid) if bid is not None else None,
                    float(ask) if ask is not None else None)
    except Exception:
        pass
    # Fallback old endpoint
    try:
        j = http_get_json(f"https://api.polygon.io/v3/last_quote/options/{opt_ticker}",
                          {"apiKey": key}, 2)
        q = (j.get("results") or {})
        bid = q.get("bid price") or q.get("bid_price")
        ask = q.get("ask price") or q.get("ask_price")
        if bid is not None or ask is not None:
            return (float(bid) if bid is not None else None,
                    float(ask) if ask is not None else None)
    except Exception:
        pass
    return (None, None)

# -------------- build body --------------
def section_options_snapshot() -> str:
    key = (os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        return "(options snapshot unavailable — POLYGON_API_KEY missing)"

    # Gather candidates: picks + hi70 (limited)
    picks_syms: List[str] = []
    hi70_syms: List[str] = []

    # picks from CSV if available
    if pd is not None:
        try:
            dfp = pd.read_csv("backtests/picklist_highrsi_trend.csv")
            wk = "week_start" if "week_start" in dfp.columns else ("week" if "week" in dfp.columns else None)
            if wk:
                week = str(pd.to_datetime(dfp[wk], errors="coerce").dropna().dt.date.max())
                if "rank" in dfp.columns:
                    dfp = dfp.sort_values(["rank","symbol"], ascending=[True, True])
                elif "score" in dfp.columns:
                    dfp = dfp.sort_values(["score","symbol"], ascending=[False, True])
                picks_syms = (dfp[pd.to_datetime(dfp[wk], errors="coerce").dt.date.astype(str)==week]["symbol"]
                              .dropna().astype(str).head(10).tolist())
        except Exception:
            pass
        try:
            dfh = pd.read_csv("backtests/hi70_thisweek.csv")
            hi70_syms = dfh["symbol"].dropna().astype(str).head(10).tolist()
        except Exception:
            pass

    universe = list(dict.fromkeys(picks_syms + hi70_syms))  # dedup, keep order
    if not universe:
        return "(no picks/hi70 files to snapshot options)"

    # rank by market cap, keep top few
    mcs = mc_for_symbols(key, universe)
    ranked = sorted(universe, key=lambda s: mcs.get(s,0.0), reverse=True)
    under = ranked[:8]  # cap to 8 for speed

    lines: List[str] = []
    for sym in under:
        px = prev_close(key, sym)
        if px is None or px <= 0:
            lines.append(f"{sym}: no prev close / no data")
            continue

        expiry = choose_expiry(key, sym, target_days=45)
        if not expiry:
            lines.append(f"{sym} close {px:.2f}: no expiries found")
            continue

        inc = strike_increment(px)
        call_strike = round_to_grid(px * 1.10, inc)
        put_strike  = round_to_grid(px * 0.90, inc)

        yyyymmdd = expiry.replace("-", "")
        call_tkr = f"O:{sym}{yyyymmdd}C{encode_strike_1c(call_strike)}"
        put_tkr  = f"O:{sym}{yyyymmdd}P{encode_strike_1c(put_strike)}"

        cbid, cask = last_quote_for_option(key, call_tkr)
        pbid, pask = last_quote_for_option(key, put_tkr)
        cmid = None if (cbid is None and cask is None) else (None if (cbid is None or cask is None) else (cbid + cask)/2)
        pmid = None if (pbid is None and pask is None) else (None if (pbid is None or pask is None) else (pbid + pask)/2)

        def fmt(x): return "NA" if x is None else f"{x:.2f}"
        lines.append(
            f"{sym} close {px:.2f} | expiry {expiry} "
            f"CALL +10% {call_strike:g}: bid {fmt(cbid)} ask {fmt(cask)} mid {fmt(cmid)} ({call_tkr}) | "
            f"PUT -10% {put_strike:g}: bid {fmt(pbid)} ask {fmt(pask)} mid {fmt(pmid)} ({put_tkr})"
        )

    if not lines:
        return "(no option quotes available)"
    return "\n".join(lines)

def build_body() -> str:
    parts: List[str] = []

    txt, src = find_picks_file_text()
    parts.append("=== Weekly Picks ===")
    parts.append(txt)
    parts.append("")

    htxt, hsrc = find_hi70_digest_text()
    parts.append("=== 70D Highs (Top by market cap) ===")
    parts.append(htxt)
    parts.append("")

    parts.append("=== Options snapshot (±10% OTM, ~45d) ===")
    parts.append(section_options_snapshot())
    body = ("\n".join(parts)).rstrip() + "\n"

    try:
        with open("email_body.txt","w",encoding="utf-8") as f:
            f.write(body)
        print("[email] Wrote email_body.txt")
    except Exception as e:
        print("[email] WARN: could not write email_body.txt:", e)
    return body

# -------------- send --------------
def send_email(cfg, subject, body):
    """
    Sends a plain-text email via SMTP.
    Works out-of-the-box with Gmail when using an App Password.
    """
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]               # required
    pwd  = os.environ["SMTP_PASS"]               # required
    from_addr = os.environ.get("SMTP_FROM", user)
    to_raw = os.environ["SMTP_TO"]               # required
    to_addrs = [x.strip() for x in to_raw.split(",") if x.strip()]

    # Safety: Gmail rejects unknown From unless configured as an alias.
    if from_addr.strip().lower() != user.strip().lower():
        # You can remove this override if you've added the alias in Gmail (Send mail as)
        from_addr = user

    # Build message
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    # Helpful logging (no secrets)
    print(f"[email] host={host} port={port} from={from_addr} to={to_addrs}")

    # Port handling: 465 => SSL; 587 => STARTTLS
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30) as s:
            s.login(user, pwd)
            s.sendmail(from_addr, to_addrs, msg.as_string())
    else:
        # Default to 587 STARTTLS (Gmail standard)
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            s.starttls()      # <— IMPORTANT for 587
            s.ehlo()
            s.login(user, pwd)
            s.sendmail(from_addr, to_addrs, msg.as_string())


# -------------- main --------------
def main() -> None:
    cfg = load_config()
    body = build_body()
    prefix = cfg.get("subject_prefix","[Weekly]").strip()
    subject = (prefix + " " if prefix else "") + "Weekly picks + 70D highs"
    send_email(cfg, subject, body)

if __name__ == "__main__":
    main()

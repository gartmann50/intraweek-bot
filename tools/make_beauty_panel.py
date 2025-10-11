#!/usr/bin/env python3
"""
Beauty panel for IW Bot weekly email:
- Per-ticker metrics (momentum Top-K and breakouts Top-10)
- Portfolio stats on momentum basket

Writes: backtests/email_charts/beauty.html

Env: POLYGON_API_KEY
Inputs:
  --picklist backtests/picklist_highrsi_trend.csv
  --hi70     backtests/hi70_thisweek.csv  (optional; auto-discover)
  --topk     6
"""

from __future__ import annotations
import os, pathlib as p, argparse, datetime as dt, requests, numpy as np, pandas as pd

API = "https://api.polygon.io"
KEY = (os.getenv("POLYGON_API_KEY") or "").strip()
OUTDIR = p.Path("backtests/email_charts")
OUTDIR.mkdir(parents=True, exist_ok=True)

def fetch_ohlc(symbol: str, lookback_days: int = 300):
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    j = requests.get(url, params={"adjusted":"true","sort":"asc","limit":"50000","apiKey":KEY}, timeout=30).json()
    rows = j.get("results") or []
    if not rows: return None
    df = pd.DataFrame(rows)[["t","o","h","l","c","v"]].rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"vol"})
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    return df

def atr14_pct(df: pd.DataFrame) -> float:
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().iloc[-1]
    if not np.isfinite(atr): return float("nan")
    return float(100*atr/c[-1])

def sigma63_pct(df: pd.DataFrame) -> float:
    c = df["close"].astype(float).to_numpy()
    if len(c) < 65: return float("nan")
    r = np.diff(c)/c[:-1]
    s = np.std(r[-252:]) if len(r) >= 252 else np.std(r)
    return float(100*s*np.sqrt(63))

def ret63_pct(df: pd.DataFrame) -> float:
    c = df["close"].astype(float).to_numpy()
    if len(c) < 64: return float("nan")
    return float(100*(c[-1]/c[-63]-1))

def snr63(df: pd.DataFrame) -> float:
    r = ret63_pct(df); s = sigma63_pct(df)
    if not np.isfinite(r) or not np.isfinite(s) or s==0: return float("nan")
    return float(r/s)

def gap_vs_prior70(df: pd.DataFrame) -> float:
    # distance of last close vs max high of prior 70 trading days (exclude last bar)
    if len(df) < 80: return float("nan")
    prior70 = float(np.max(df["high"].astype(float).to_numpy()[-71:-1]))
    last = float(df["close"].iloc[-1])
    if prior70 <= 0: return float("nan")
    return float(100*(last/prior70-1))

def name_of(symbol: str) -> str:
    try:
        j = requests.get(f"{API}/v3/reference/tickers/{symbol}",
                         params={"apiKey": KEY}, timeout=20).json()
        return (j.get("results") or {}).get("name") or symbol
    except Exception:
        return symbol

def read_topk_from_picklist(path: str, topk: int) -> list[str]:
    df = pd.read_csv(path)
    wk = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
    if wk:
        latest = pd.to_datetime(df[wk], errors="coerce").dropna().dt.date
        if len(latest):
            last_week = str(latest.max())
            df = df[pd.to_datetime(df[wk], errors="coerce").dt.date.astype(str) == last_week].copy()
    if "rank" in df.columns:
        df = df.sort_values(["rank","symbol"], ascending=[True,True])
    elif "score" in df.columns:
        df = df.sort_values(["score","symbol"], ascending=[False,True])
    return df["symbol"].dropna().astype(str).head(int(topk)).tolist()

def find_hi70_csv(default: p.Path = p.Path("backtests/hi70_thisweek.csv")) -> p.Path|None:
    if default.exists(): return default
    root = p.Path("backtests")
    cands = list(root.glob("**/hi70_thisweek.csv")) if root.exists() else []
    if not cands: return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]

def read_breakouts(path: p.Path|None, topN=10) -> list[str]:
    if not path or not path.exists(): return []
    df = pd.read_csv(path)
    return df["symbol"].astype(str).str.upper().head(topN).tolist()

def momentum_portfolio_stats(dfs: dict[str,pd.DataFrame]) -> tuple[float,float,float]:
    # returns avg pairwise corr, est EW 5d vol %, avg ATR%
    syms = list(dfs.keys())
    rets = []
    atrs = []
    for s in syms:
        d = dfs[s]
        c = d["close"].astype(float).to_numpy()
        if len(c) < 66: continue
        r = np.diff(c)/c[:-1]
        rets.append(r[-252:] if len(r)>=252 else r)
        atrs.append(atr14_pct(d))
    if len(rets) < 2:
        return float("nan"), float("nan"), (np.nanmean(atrs) if atrs else float("nan"))
    # align by min length
    L = min(map(len, rets))
    R = np.vstack([r[-L:] for r in rets])
    corr = np.corrcoef(R)
    avg_offdiag = np.nan
    if corr.shape[0] > 1:
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg_offdiag = float(np.mean(corr[mask]))
    # EW vol 5d
    cov = np.cov(R)
    n = cov.shape[0]
    w = np.ones(n)/n
    daily_vol = float(np.sqrt(w @ cov @ w))
    vol5_pct = float(100*daily_vol*np.sqrt(5))
    avg_atr = float(np.nanmean(atrs)) if atrs else float("nan")
    return avg_offdiag, vol5_pct, avg_atr

def fmt(x, prec=2, pct=False):
    if not np.isfinite(x): return "—"
    return f"{x:.{prec}f}{'%' if pct else ''}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picklist", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--hi70",     default="backtests/hi70_thisweek.csv")
    ap.add_argument("--topk",     type=int, default=6)
    args = ap.parse_args()
    if not KEY: raise SystemExit("FATAL: POLYGON_API_KEY missing")

    # symbols
    mom_syms = []
    try: mom_syms = read_topk_from_picklist(args.picklist, args.topk)
    except Exception: pass
    hi70_path = p.Path(args.hi70) if args.hi70 else p.Path("backtests/hi70_thisweek.csv")
    if not hi70_path.exists(): hi70_path = find_hi70_csv(hi70_path)
    brk_syms = read_breakouts(hi70_path, 10)

    # fetch + metrics
    mom_rows = []
    brk_rows = []
    mom_dfs = {}
    for s in mom_syms:
        df = fetch_ohlc(s, 400)
        if df is None: continue
        mom_dfs[s] = df
        row = {
            "sym": s,
            "name": name_of(s),
            "price": df["close"].iloc[-1],
            "ret63": ret63_pct(df),
            "sig63": sigma63_pct(df),
            "snr":   snr63(df),
            "atrp":  atr14_pct(df)
        }
        mom_rows.append(row)
    for s in brk_syms:
        df = fetch_ohlc(s, 400)
        if df is None: continue
        row = {
            "sym": s,
            "name": name_of(s),
            "price": df["close"].iloc[-1],
            "ret63": ret63_pct(df),
            "sig63": sigma63_pct(df),
            "snr":   snr63(df),
            "atrp":  atr14_pct(df),
            "gap70": gap_vs_prior70(df)
        }
        brk_rows.append(row)

    avg_corr, vol5, avg_atr = momentum_portfolio_stats(mom_dfs)

    # HTML
    def table(rows, breakout=False):
        if not rows: return ""
        cols = ["Symbol","Name","Price","63d","σ₆₃","SNR","ATR₁₄%"] + (["Gap70%"] if breakout else [])
        head = "".join(f"<th style='padding:6px 8px;text-align:left;font-weight:600;border-bottom:1px solid #eee'>{c}</th>" for c in cols)
        body = []
        for r in rows:
            cells = [
                r["sym"],
                r["name"],
                fmt(r["price"],2,False),
                fmt(r["ret63"],2,True),
                fmt(r["sig63"],2,True),
                fmt(r["snr"],2,False),
                fmt(r["atrp"],2,True)
            ]
            if breakout: cells.append(fmt(r.get("gap70", float("nan")),2,True))
            tds = "".join(f"<td style='padding:6px 8px;border-bottom:1px solid #f6f6f6'>{c}</td>" for c in cells)
            body.append(f"<tr>{tds}</tr>")
        return (
            "<table style='border-collapse:collapse;width:100%;margin:6px 0 10px 0'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        )

    top = (
        "<div style='border:1px solid #e9ecef;border-radius:10px;padding:12px;margin:10px 0'>"
        "<div style='font-weight:700;margin-bottom:6px'>Beauty panel — signal & risk</div>"
        "<div style='color:#444;font-size:13px'>"
        f"Momentum basket: avg pairwise corr <b>{fmt(avg_corr,2,False)}</b> · "
        f"est EW vol (5d) <b>{fmt(vol5,2,True)}</b> · "
        f"avg ATR₁₄ <b>{fmt(avg_atr,2,True)}</b>"
        "</div>"
        "</div>"
    )
    html = [
        top,
        "<div style='margin-top:6px'><b>Momentum picks — metrics</b></div>",
        table(mom_rows, breakout=False),
        "<div style='margin-top:6px'><b>Breakouts — Top-10 metrics</b></div>",
        table(brk_rows, breakout=True)
    ]
    (OUTDIR/"beauty.html").write_text("\n".join(html), encoding="utf-8")
    print("[beauty] wrote:", OUTDIR/"beauty.html")

if __name__ == "__main__":
    main()

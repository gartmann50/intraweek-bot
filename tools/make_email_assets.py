#!/usr/bin/env python3
"""
Builds small charts + HTML for the weekly email.

Inputs (defaults match your repo):
  --picklist backtests/picklist_highrsi_trend.csv   (momentum Top-K table)
  --hi70     backtests/hi70_thisweek.csv            (breakouts table)
  --topk     6                                      (# momentum names to show)
  --outdir   backtests/email_charts                 (images live here)
Env:
  POLYGON_API_KEY  (required for names + bars)
"""
from __future__ import annotations
import os, argparse, datetime as dt, requests, pandas as pd, pathlib as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API = "https://api.polygon.io"
KEY = (os.getenv("POLYGON_API_KEY") or "").strip()

def bars(symbol: str, days: int) -> list[float]:
    # Fetch more than needed (120d) and slice from the end
    end = dt.date.today()
    start = end - dt.timedelta(days=max(180, days*3))
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted":"true","sort":"asc","limit":"50000","apiKey":KEY}
    try:
        j = requests.get(url, params=params, timeout=30).json()
        xs = [float(x.get("c",0) or 0) for x in j.get("results",[]) or []]
        xs = [x for x in xs if x>0]
        return xs[-days:] if len(xs) >= days else xs
    except Exception:
        return []

def name_of(symbol: str) -> str:
    # Try Polygon reference; fall back to symbol if missing
    try:
        j = requests.get(f"{API}/v3/reference/tickers/{symbol}",
                         params={"apiKey":KEY}, timeout=20).json()
        return (j.get("results") or {}).get("name") or symbol
    except Exception:
        return symbol

def sparkline(vals: list[float], out: p.Path):
    if not vals:
        return
    fig = plt.figure(figsize=(2.4, 0.9), dpi=100)  # ~240x90 px
    ax = fig.add_axes([0,0,1,1])
    ax.plot(vals, linewidth=1.6)
    # subtle baseline to make trends pop
    ax.axhline(vals[0], linewidth=0.6, alpha=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def read_topk_from_picklist(picklist: p.Path, topk: int) -> list[str]:
    df = pd.read_csv(picklist)
    wk = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
    if wk:
        week = str(pd.to_datetime(df[wk], errors="coerce").dropna().dt.date.max())
        df = df[pd.to_datetime(df[wk], errors="coerce").dt.date.astype(str) == week].copy()
    if "rank" in df.columns:
        df = df.sort_values(["rank","symbol"], ascending=[True,True])
    elif "score" in df.columns:
        df = df.sort_values(["score","symbol"], ascending=[False,True])
    return df["symbol"].dropna().astype(str).head(int(topk)).tolist()

def read_breakouts(hi70: p.Path, topN: int=10) -> list[tuple[str,str]]:
    if not hi70.exists(): return []
    df = pd.read_csv(hi70)
    # if hi70 has 'name' column, use it
    out=[]
    for _,r in df.head(topN).iterrows():
        sym=str(r.get("symbol") or "").upper()
        nm = str(r.get("name") or "")
        out.append((sym,nm))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picklist", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--hi70",     default="backtests/hi70_thisweek.csv")
    ap.add_argument("--topk",     type=int, default=6)
    ap.add_argument("--outdir",   default="backtests/email_charts")
    args = ap.parse_args()

    outdir = p.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Momentum list
    mom_syms = read_topk_from_picklist(p.Path(args.picklist), args.topk)

    # Breakouts (symbol, name) – fill name via API if missing
    brk_pairs = read_breakouts(p.Path(args.hi70), 10)

    # Build image assets + name map
    mom_rows = []
    for s in mom_syms:
        nm = name_of(s)
        img = outdir/f"MOM_{s}.png"
        sparkline(bars(s, 22), img)   # ~1 month
        mom_rows.append((s, nm, img.name))

    brk_rows = []
    for s, nm in brk_pairs:
        if not nm: nm = name_of(s)
        img = outdir/f"BO_{s}.png"
        sparkline(bars(s, 66), img)   # ~3 months
        brk_rows.append((s, nm, img.name))

    # HTML (inline by cid:filename)
    def section(title, rows):
        if not rows: return ""
        lines=[f"<h3 style='margin:12px 0'>{title}</h3>",
               "<table style='border-collapse:collapse;width:100%'>"]
        for s, nm, img in rows:
            lines.append(
                "<tr>"
                f"<td style='padding:6px 8px;width:260px'><img src='cid:{img}' width='240' height='90' style='display:block;border:0'></td>"
                f"<td style='padding:6px 8px;vertical-align:middle'><div style='font-weight:600'>{s}</div>"
                f"<div style='color:#666;font-size:13px'>{nm}</div></td>"
                "</tr>"
            )
        lines.append("</table>")
        return "\n".join(lines)

    html = [
      "<!doctype html><meta charset='utf-8'>",
      "<div style='font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif'>",
      "<h2 style='margin:0 0 8px'>IW Bot — Weekly Summary</h2>",
      section("Momentum picks (1-month mini-charts)", mom_rows),
      section("Breakouts — Top-10 (3-month mini-charts)", brk_rows),
      "<p style='color:#888;font-size:12px;margin-top:12px'>"
      "Mini-charts: daily closes only; for context, consult your platform."
      "</p></div>"
    ]
    (outdir/"email.html").write_text("\n".join(html), encoding="utf-8")

if __name__ == "__main__":
    if not KEY:
        raise SystemExit("POLYGON_API_KEY missing")
    main()

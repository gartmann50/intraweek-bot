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
import numpy as np
import matplotlib.dates as mdates


API = "https://api.polygon.io"
KEY = (os.getenv("POLYGON_API_KEY") or "").strip()

def bars(symbol: str, days: int) -> tuple[list[dt.date], list[float]]:
    """
    Return (dates, closes) for the last `days` trading days.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=max(200, int(days*4)))  # buffer
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted":"true","sort":"asc","limit":"50000","apiKey":KEY}
    try:
        j = requests.get(url, params=params, timeout=30).json()
        res = j.get("results") or []
        dts = [dt.datetime.utcfromtimestamp(int(r["t"])/1000).date()
               for r in res if r.get("c")]
        cls = [float(r["c"]) for r in res if r.get("c")]
        if len(cls) > days:
            dts, cls = dts[-days:], cls[-days:]
        return dts, cls
    except Exception:
        return [], []


def name_of(symbol: str) -> str:
    # Try Polygon reference; fall back to symbol if missing
    try:
        j = requests.get(f"{API}/v3/reference/tickers/{symbol}",
                         params={"apiKey":KEY}, timeout=20).json()
        return (j.get("results") or {}).get("name") or symbol
    except Exception:
        return symbol

def mini_chart(dts: list[dt.date], vals: list[float], out: p.Path, months=False):
    if not vals or not dts:  # nothing to draw
        return

    # ~half previous size; a bit higher DPI so labels stay crisp
    fig = plt.figure(figsize=(1.2, 0.6), dpi=130)  # ≈ 156×78 px
    ax  = fig.add_axes([0.10, 0.18, 0.86, 0.74])   # tight margins

    ax.plot(dts, vals, linewidth=1.2)
    ax.grid(alpha=0.18, linewidth=0.4)

    # X axis: concise dates; month ticks for 3-month view
    if months:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Jan→'Jan'
    else:
        loc = mdates.AutoDateLocator(minticks=2, maxticks=4)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    # Y axis: 3 ticks (min, mid, max) with small font
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        vmax = vmin * 1.01 + 0.01
    ax.set_ylim(vmin - (vmax - vmin) * 0.05, vmax + (vmax - vmin) * 0.05)
    ax.set_yticks(np.linspace(vmin, vmax, 3))
    ax.tick_params(axis='x', labelsize=7, pad=1)
    ax.tick_params(axis='y', labelsize=7, pad=1)

    # keep frame subtle
    for s in ax.spines.values():
        s.set_visible(False)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
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
        dts, cls = bars(s, 22)              # ~1 month
        mini_chart(dts, cls, img, months=False)
        mom_rows.append((s, nm, img.name))


    brk_rows = []
    for s, nm in brk_pairs:
        if not nm: nm = name_of(s)
        img = outdir/f"BO_{s}.png"
        dts, cls = bars(s, 63)              # ~3 months
        mini_chart(dts, cls, img, months=True)
        brk_rows.append((s, nm, img.name))


    # HTML (inline by cid:filename)
  def section(title, rows):
    if not rows: return ""
    lines=[f"<h3 style='margin:12px 0'>{title}</h3>",
           "<table style='border-collapse:collapse;width:100%'>"]
    for s, nm, img in rows:
        lines.append(
            "<tr>"
            f"<td style='padding:4px 6px;width:160px'>"
            f"<img src='cid:{img}' width='140' height='70' "
            f"style='display:block;border:0'></td>"
            f"<td style='padding:4px 6px;vertical-align:middle'>"
            f"<div style='font-weight:600'>{s}</div>"
            f"<div style='color:#666;font-size:13px'>{nm}</div>"
            "</td></tr>"
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

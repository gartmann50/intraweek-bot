#!/usr/bin/env python3
"""
Builds small charts + HTML for the weekly email.

Outputs in backtests/email_charts/:
  - PNG mini-charts (momentum & breakout)
  - email.html (embeds charts by cid:filename)

Inputs:
  --picklist backtests/picklist_highrsi_trend.csv
  --hi70     backtests/hi70_thisweek.csv
  --topk     6
  --outdir   backtests/email_charts

Env:
  POLYGON_API_KEY   (required)
"""

from __future__ import annotations
import os
import argparse
import datetime as dt
import pathlib as p
import requests
import pandas as pd
import numpy as np

# ---- plotting setup (headless) ----
import matplotlib
matplotlib.use("Agg")  # ensure headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

API = "https://api.polygon.io"
KEY = (os.getenv("POLYGON_API_KEY") or "").strip()


# ----------------- helpers -----------------

def bars(symbol: str, days: int) -> tuple[list[dt.date], list[float]]:
    """
    Fetch daily bars and return (dates, closes) for the last `days` trading days.
    Uses a buffer to ensure we have enough data.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=max(200, int(days * 4)))  # generous buffer
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": KEY}
    try:
        j = requests.get(url, params=params, timeout=30).json()
        res = j.get("results") or []
        dts = []
        cls = []
        for r in res:
            c = r.get("c")
            t = r.get("t")
            if c is None or t is None:
                continue
            try:
                dts.append(dt.datetime.utcfromtimestamp(int(t) / 1000).date())
                cls.append(float(c))
            except Exception:
                continue
        if len(cls) > days:
            dts, cls = dts[-days:], cls[-days:]
        return dts, cls
    except Exception:
        return [], []


def name_of(symbol: str) -> str:
    """Fetch company name from Polygon. Fallback to symbol on error."""
    try:
        j = requests.get(f"{API}/v3/reference/tickers/{symbol}",
                         params={"apiKey": KEY}, timeout=20).json()
        return (j.get("results") or {}).get("name") or symbol
    except Exception:
        return symbol


def mini_chart(dts: list[dt.date], vals: list[float], out: p.Path, months: bool = False):
    """
    Render a compact chart with axes:
      - X: concise dates (month ticks if months=True)
      - Y: price ticks (min/mid/max)
    Size ~ half of the previous version (~140x70 px).
    """
    if not dts or not vals:
        return

    # ~156x78 px at dpi=130
    fig = plt.figure(figsize=(1.2, 0.6), dpi=130)
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.74])

    ax.plot(dts, vals, linewidth=1.2)
    ax.grid(alpha=0.18, linewidth=0.4)

    # X axis formatting
    if months:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Jan, Feb, ...
    else:
        loc = mdates.AutoDateLocator(minticks=2, maxticks=4)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    # Y axis: min/mid/max
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmin == vmax:
        vmax = vmin * 1.01 + 0.01
    pad = (vmax - vmin) * 0.05
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_yticks(np.linspace(vmin, vmax, 3))

    ax.tick_params(axis='x', labelsize=7, pad=1)
    ax.tick_params(axis='y', labelsize=7, pad=1)

    # Subtle frame
    for s in ax.spines.values():
        s.set_visible(False)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def read_topk_from_picklist(picklist: p.Path, topk: int) -> list[str]:
    """From picklist CSV, take the latest week and return Top-K symbols."""
    df = pd.read_csv(picklist)
    wk = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
    if wk:
        latest = pd.to_datetime(df[wk], errors="coerce").dropna().dt.date
        if len(latest):
            last_week = str(latest.max())
            df = df[pd.to_datetime(df[wk], errors="coerce").dt.date.astype(str) == last_week].copy()

    if "rank" in df.columns:
        df = df.sort_values(["rank", "symbol"], ascending=[True, True])
    elif "score" in df.columns:
        df = df.sort_values(["score", "symbol"], ascending=[False, True])

    return df["symbol"].dropna().astype(str).head(int(topk)).tolist()


def read_breakouts(hi70: p.Path, topN: int = 10) -> list[tuple[str, str]]:
    """Read breakouts CSV: returns list of (symbol, name). Name may be blank."""
    if not hi70.exists():
        return []
    df = pd.read_csv(hi70)
    out = []
    for _, r in df.head(topN).iterrows():
        sym = str(r.get("symbol") or "").upper()
        nm = str(r.get("name") or "")
        if sym:
            out.append((sym, nm))
    return out


def section_html(title: str, rows: list[tuple[str, str, str]]) -> str:
    """HTML table for a set of rows (sym, name, img-filename)."""
    if not rows:
        return ""
    lines = [
        f"<h3 style='margin:12px 0'>{title}</h3>",
        "<table style='border-collapse:collapse;width:100%'>"
    ]
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


# ----------------- main -----------------

def main():
    if not KEY:
        raise SystemExit("FATAL: POLYGON_API_KEY is missing")

    ap = argparse.ArgumentParser()
    ap.add_argument("--picklist", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--hi70",     default="backtests/hi70_thisweek.csv")
    ap.add_argument("--topk",     type=int, default=6)
    ap.add_argument("--outdir",   default="backtests/email_charts")
    args = ap.parse_args()

    outdir = p.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Momentum symbols (Top-K, latest week)
    mom_syms = []
    try:
        mom_syms = read_topk_from_picklist(p.Path(args.picklist), args.topk)
    except Exception:
        mom_syms = []

    # Breakouts (symbol, name) — take 10
    try:
        brk_pairs = read_breakouts(p.Path(args.hi70), 10)
    except Exception:
        brk_pairs = []

    mom_rows: list[tuple[str, str, str]] = []
    for s in mom_syms:
        nm  = name_of(s)
        img = outdir / f"MOM_{s}.png"
        dts, cls = bars(s, 22)  # ~1 month
        mini_chart(dts, cls, img, months=False)
        mom_rows.append((s, nm, img.name))

    brk_rows: list[tuple[str, str, str]] = []
    for s, nm in brk_pairs:
        if not nm:
            nm = name_of(s)
        img = outdir / f"BO_{s}.png"
        dts, cls = bars(s, 63)  # ~3 months
        mini_chart(dts, cls, img, months=True)
        brk_rows.append((s, nm, img.name))

    # HTML
    html_parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<div style='font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif'>",
        "<h2 style='margin:0 0 8px'>IW Bot — Weekly Summary</h2>",
        section_html("Momentum picks (1-month mini-charts)", mom_rows),
        section_html("Breakouts — Top-10 (3-month mini-charts)", brk_rows),
        "<p style='color:#888;font-size:12px;margin-top:12px'>"
        "Mini-charts: daily closes; for context, consult your platform."
        "</p></div>"
    ]
    (outdir / "email.html").write_text("\n".join(html_parts), encoding="utf-8")

    # Console hints
    print(f"[email] momentum: {len(mom_rows)} charts")
    print(f"[email] breakouts: {len(brk_rows)} charts")
    print(f"[email] wrote: {outdir/'email.html'}")


if __name__ == "__main__":
    main()

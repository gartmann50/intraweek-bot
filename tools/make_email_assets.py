#!/usr/bin/env python3
"""
Builds candlestick mini-charts + HTML for the weekly email, with
WEEKENDS REMOVED (compressed to trading-day index).

Both sections (Momentum & Breakouts) use ~3 months of daily bars.

Outputs in backtests/email_charts/:
  - PNG mini-charts (momentum & breakout) as candlesticks
  - email.html (embeds charts by cid:filename)

Inputs:
  --picklist backtests/picklist_highrsi_trend.csv
  --hi70     backtests/hi70_thisweek.csv  (optional; auto-discover if missing)
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
from matplotlib.patches import Rectangle

API = "https://api.polygon.io"
KEY = (os.getenv("POLYGON_API_KEY") or "").strip()

UP_COLOR = "#2ca02c"
DN_COLOR = "#d62728"


# ----------------- data fetch -----------------

def ohlc(symbol: str, days: int) -> tuple[list[dt.date], list[float], list[float], list[float], list[float]]:
    """
    Fetch daily bars and return (dates, opens, highs, lows, closes)
    for the last `days` trading days. Uses a buffer to ensure enough data.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=max(200, int(days * 4)))  # generous buffer
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": KEY}
    try:
        j = requests.get(url, params=params, timeout=30).json()
        res = j.get("results") or []
        dts, opn, high, low, cls = [], [], [], [], []
        for r in res:
            o = r.get("o"); h = r.get("h"); l = r.get("l"); c = r.get("c"); t = r.get("t")
            if None in (o, h, l, c, t):
                continue
            dts.append(dt.datetime.utcfromtimestamp(int(t) / 1000).date())
            opn.append(float(o)); high.append(float(h)); low.append(float(l)); cls.append(float(c))
        if len(cls) > days:
            dts, opn, high, low, cls = dts[-days:], opn[-days:], high[-days:], low[-days:], cls[-days:]
        return dts, opn, high, low, cls
    except Exception:
        return [], [], [], [], []


def name_of(symbol: str) -> str:
    """Fetch company name from Polygon. Fallback to symbol on error."""
    try:
        j = requests.get(f"{API}/v3/reference/tickers/{symbol}",
                         params={"apiKey": KEY}, timeout=20).json()
        return (j.get("results") or {}).get("name") or symbol
    except Exception:
        return symbol


# ----------------- plotting (compressed trading axis) -----------------

def _tick_positions_and_labels(dts: list[dt.date], months: bool) -> tuple[list[int], list[str]]:
    """
    Build compact, human-friendly X tick positions/labels on a trading-day index.
    - For 3-month views (months=True): tick at first trading day of each month plus the last day.
    """
    n = len(dts)
    if n == 0:
        return [], []

    # month ticks (used for both sections now)
    pos = []
    lab = []
    seen = set()
    for i, d in enumerate(dts):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            pos.append(i)
            lab.append(d.strftime("%b"))  # Jan, Feb, ...
    if pos[-1] != n - 1:
        pos.append(n - 1); lab.append(dts[-1].strftime("%b"))
    return pos, lab


def mini_candles(
    dts: list[dt.date],
    o: list[float], h: list[float], l: list[float], c: list[float],
    out: p.Path,
    months: bool = True
):
    """
    Render a compact candlestick chart with axes using TRADING-DAY INDEX on X:
      - X: integer index (no weekend gaps) with month tick labels
      - Y: price ticks (min/mid/max)
    Size ~ 140x70 px.
    """
    if not dts or not c:
        return

    # ~156x78 px at dpi=130
    fig = plt.figure(figsize=(1.2, 0.6), dpi=130)
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.74])

    x = np.arange(len(dts), dtype=float)
    width = 0.8  # ~80% of spacing

    # Draw wicks + bodies
    for xi, oi, hi, lo, ci in zip(x, o, h, l, c):
        color = UP_COLOR if ci >= oi else DN_COLOR
        # wick
        ax.vlines(xi, lo, hi, colors=color, linewidth=0.6, alpha=0.9)
        # body
        y0 = min(oi, ci); bh = abs(ci - oi)
        if bh < max(1e-6, (hi - lo) * 0.02):  # doji-ish
            ax.hlines((oi + ci) / 2.0, xi - width/2, xi + width/2, colors=color, linewidth=0.8, alpha=0.9)
        else:
            rect = Rectangle((xi - width/2, y0), width, bh, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.9)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, x.max() + 0.5)
    ax.grid(alpha=0.18, linewidth=0.4)

    # X ticks: positions on index, labels from dates (months)
    pos, lab = _tick_positions_and_labels(dts, months=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(lab, fontsize=7)

    # Y axis: min/mid/max
    vmin = float(min(l)); vmax = float(max(h))
    if vmin == vmax:
        vmax = vmin * 1.01 + 0.01
    pad = (vmax - vmin) * 0.05
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_yticks(np.linspace(vmin, vmax, 3))
    ax.tick_params(axis='y', labelsize=7, pad=1)

    for s in ax.spines.values():
        s.set_visible(False)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ----------------- CSV helpers -----------------

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


def find_hi70_csv(default: p.Path = p.Path("backtests/hi70_thisweek.csv")) -> p.Path | None:
    """Locate hi70_thisweek.csv anywhere under backtests/ (newest first)."""
    if default.exists():
        return default
    root = p.Path("backtests")
    if not root.exists():
        return None
    candidates = list(root.glob("**/hi70_thisweek.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def read_breakouts(hi70_path: p.Path | None, topN: int = 10) -> list[tuple[str, str]]:
    """Read breakouts CSV: returns list of (symbol, name)."""
    if not hi70_path or not hi70_path.exists():
        return []
    df = pd.read_csv(hi70_path)
    rows = []
    for _, r in df.head(topN).iterrows():
        sym = str(r.get("symbol") or "").upper()
        nm  = str(r.get("name") or "")
        if sym:
            rows.append((sym, nm))
    return rows


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
            f"<img src='cid:{img}' width='140' height='70' style='display:block;border:0'></td>"
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
    try:
        mom_syms = read_topk_from_picklist(p.Path(args.picklist), args.topk)
    except Exception:
        mom_syms = []

    # Breakouts CSV: use given path if it exists, otherwise discover latest
    hi70_path = p.Path(args.hi70)
    if not hi70_path.exists():
        hi70_path = find_hi70_csv(hi70_path)
    print(f"[email] hi70 CSV: {hi70_path if hi70_path else '(not found)'}")

    # Breakouts (symbol, name)
    try:
        brk_pairs = read_breakouts(hi70_path, 10)
    except Exception:
        brk_pairs = []

    # Build images (3 months for BOTH sections)
    mom_rows: list[tuple[str, str, str]] = []
    for s in mom_syms:
        nm  = name_of(s)
        img = outdir / f"MOM_{s}.png"
        dts, op, hi, lo, cl = ohlc(s, 63)  # ~3 months
        mini_candles(dts, op, hi, lo, cl, img, months=True)
        mom_rows.append((s, nm, img.name))

    brk_rows: list[tuple[str, str, str]] = []
    for s, nm in brk_pairs:
        if not nm:
            nm = name_of(s)
        img = outdir / f"BO_{s}.png"
        dts, op, hi, lo, cl = ohlc(s, 63)  # ~3 months
        mini_candles(dts, op, hi, lo, cl, img, months=True)
        brk_rows.append((s, nm, img.name))

    # HTML
    html_parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<div style='font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif'>",
        "<h2 style='margin:0 0 8px'>IW Bot — Weekly Summary</h2>",
        section_html("Momentum picks (3-month mini-candles)", mom_rows),
        section_html("Breakouts — Top-10 (3-month mini-candles)", brk_rows),
        "<p style='color:#888;font-size:12px;margin-top:12px'>"
        "Mini-charts: daily candlesticks; weekends compressed out."
        "</p></div>"
    ]
    (outdir / "email.html").write_text("\n".join(html_parts), encoding="utf-8")

    # Console hints
    print(f"[email] momentum count: {len(mom_rows)}")
    print(f"[email] breakouts count: {len(brk_rows)}")
    if hi70_path and hi70_path.exists():
        print(f"[email] used: {hi70_path}")
    else:
        print("[email] WARNING: hi70 file not found, breakout section empty")


if __name__ == "__main__":
    main()

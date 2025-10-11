#!/usr/bin/env python3
"""
Builds the weekly HTML (Beauty Panel + prettier cards) + candlestick mini-charts.

What’s new (prettier):
- 3-month candlesticks (weekends compressed), now with EMA(20) overlay
- Price + 3-month % label on each chart (green/red badge)
- 2-column card layout with rounded corners
- Metric chips (63d%, σ63%, SNR, ATR14%) under each symbol
- Beauty Panel retained (portfolio stats + per-ticker tables)

Outputs
  backtests/email_charts/
    - email.html
    - PNGs: MOM_<sym>.png, BO_<sym>.png

Inputs
  --picklist backtests/picklist_highrsi_trend.csv
  --hi70     backtests/hi70_thisweek.csv  (optional; auto-discover if missing)
  --topk     6
  --outdir   backtests/email_charts

Env
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

API = "https://api.polygon.io"
KEY = (os.getenv("POLYGON_API_KEY") or "").strip()

UP_COLOR = "#2ca02c"
DN_COLOR = "#d62728"
EMA_COLOR = "#1f77b4"   # subtle blue line

# ===== Size & style =====
FIG_W_IN = 2.4
FIG_H_IN = 1.2
DPI      = 130
IMG_W    = 280
IMG_H    = 140
TICK_FONTSZ = 9
CARD_PAD   = "10px"
CELL_W     = 360  # width to fit chart + padding nicely


# ----------------- data fetch -----------------

def ohlc(symbol: str, days: int) -> tuple[list[dt.date], list[float], list[float], list[float], list[float]]:
    end = dt.date.today()
    start = end - dt.timedelta(days=max(200, int(days * 4)))
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


def fetch_ohlc_full(symbol: str, lookback_days: int = 400) -> pd.DataFrame | None:
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    j = requests.get(url, params={"adjusted":"true","sort":"asc","limit":"50000","apiKey":KEY}, timeout=30).json()
    rows = j.get("results") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)[["t","o","h","l","c","v"]].rename(
        columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"vol"}
    )
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    return df


def name_of(symbol: str) -> str:
    try:
        j = requests.get(f"{API}/v3/reference/tickers/{symbol}",
                         params={"apiKey": KEY}, timeout=20).json()
        return (j.get("results") or {}).get("name") or symbol
    except Exception:
        return symbol


# ----------------- plotting (compressed trading axis) -----------------

def _tick_positions_and_labels(dts: list[dt.date]) -> tuple[list[int], list[str]]:
    n = len(dts)
    if n == 0:
        return [], []
    pos, lab, seen = [], [], set()
    for i, d in enumerate(dts):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            pos.append(i)
            lab.append(d.strftime("%b"))
    if pos[-1] != n - 1:
        pos.append(n - 1); lab.append(dts[-1].strftime("%b"))
    return pos, lab


def _ema(arr: np.ndarray, win: int = 20) -> np.ndarray:
    if len(arr) == 0:
        return arr
    alpha = 2.0 / (win + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def mini_candles(
    dts: list[dt.date],
    o: list[float], h: list[float], l: list[float], c: list[float],
    out: p.Path
):
    if not dts or not c:
        return

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.74])

    x = np.arange(len(dts), dtype=float)
    width = 0.8

    # Draw wicks + bodies
    for xi, oi, hi, lo, ci in zip(x, o, h, l, c):
        color = UP_COLOR if ci >= oi else DN_COLOR
        ax.vlines(xi, lo, hi, colors=color, linewidth=0.8, alpha=0.95)
        y0 = min(oi, ci); bh = abs(ci - oi)
        if bh < max(1e-6, (hi - lo) * 0.02):
            ax.hlines((oi + ci) / 2.0, xi - width/2, xi + width/2, colors=color, linewidth=1.0, alpha=0.95)
        else:
            rect = Rectangle((xi - width/2, y0), width, bh,
                             facecolor=color, edgecolor=color, linewidth=0.9, alpha=0.95)
            ax.add_patch(rect)

    # EMA(20) overlay (on closes)
    c_arr = np.asarray(c, dtype=float)
    ema20 = _ema(c_arr, 20)
    ax.plot(x, ema20, linewidth=1.0, alpha=0.9, color=EMA_COLOR)

    ax.set_xlim(-0.5, x.max() + 0.5)
    ax.grid(alpha=0.20, linewidth=0.5)

    # X ticks
    pos, lab = _tick_positions_and_labels(dts)
    ax.set_xticks(pos)
    ax.set_xticklabels(lab, fontsize=TICK_FONTSZ)

    # Y axis
    vmin = float(min(l)); vmax = float(max(h))
    if vmin == vmax:
        vmax = vmin * 1.01 + 0.01
    pad = (vmax - vmin) * 0.05
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_yticks(np.linspace(vmin, vmax, 3))
    ax.tick_params(axis='y', labelsize=TICK_FONTSZ, pad=1)

    for s in ax.spines.values():
        s.set_visible(False)

    # Corner badge: price + window % change
    try:
        pct = 100.0 * (c_arr[-1] / c_arr[0] - 1.0)
        lbl = f"{c_arr[-1]:.2f}  ({pct:+.1f}%)"
        badge_color = UP_COLOR if pct >= 0 else DN_COLOR
        ax.text(
            0.98, 0.02, lbl,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color=badge_color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#ddd", linewidth=0.6, alpha=0.9)
        )
    except Exception:
        pass

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ----------------- CSV helpers -----------------

def read_topk_from_picklist(picklist: p.Path, topk: int) -> list[str]:
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


# ----------------- Beauty metrics -----------------

def atr14_pct(df: pd.DataFrame) -> float:
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().iloc[-1]
    if not np.isfinite(atr):
        return float("nan")
    return float(100 * atr / c[-1])

def sigma63_pct(df: pd.DataFrame) -> float:
    c = df["close"].astype(float).to_numpy()
    if len(c) < 65: return float("nan")
    r = np.diff(c) / c[:-1]
    s = np.std(r[-252:]) if len(r) >= 252 else np.std(r)
    return float(100 * s * np.sqrt(63))

def ret63_pct(df: pd.DataFrame) -> float:
    c = df["close"].astype(float).to_numpy()
    if len(c) < 64: return float("nan")
    return float(100 * (c[-1] / c[-63] - 1))

def snr63(df: pd.DataFrame) -> float:
    r = ret63_pct(df); s = sigma63_pct(df)
    if not np.isfinite(r) or not np.isfinite(s) or s == 0:
        return float("nan")
    return float(r / s)

def gap_vs_prior70(df: pd.DataFrame) -> float:
    if len(df) < 80: return float("nan")
    prior70 = float(np.max(df["high"].astype(float).to_numpy()[-71:-1]))
    last = float(df["close"].iloc[-1])
    if prior70 <= 0: return float("nan")
    return float(100 * (last / prior70 - 1))

def momentum_portfolio_stats(dfs: dict[str, pd.DataFrame]) -> tuple[float, float, float]:
    syms = list(dfs.keys())
    rets, atrs = [], []
    for s in syms:
        d = dfs[s]
        c = d["close"].astype(float).to_numpy()
        if len(c) < 66: continue
        r = np.diff(c) / c[:-1]
        rets.append(r[-252:] if len(r) >= 252 else r)
        atrs.append(atr14_pct(d))
    if len(rets) < 2:
        return float("nan"), float("nan"), (np.nanmean(atrs) if atrs else float("nan"))
    L = min(map(len, rets))
    R = np.vstack([r[-L:] for r in rets])
    corr = np.corrcoef(R)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    avg_offdiag = float(np.mean(corr[mask])) if corr.size else float("nan")
    cov = np.cov(R)
    n = cov.shape[0]
    w = np.ones(n) / n
    daily_vol = float(np.sqrt(w @ cov @ w))
    vol5_pct = float(100 * daily_vol * np.sqrt(5))
    avg_atr = float(np.nanmean(atrs)) if atrs else float("nan")
    return avg_offdiag, vol5_pct, avg_atr

def fmt(x, prec=2, pct=False):
    if not np.isfinite(x): return "—"
    return f"{x:.{prec}f}{'%' if pct else ''}"


# ----------------- HTML helpers -----------------

def chip(label: str) -> str:
    return (f"<span style='display:inline-block;background:#f3f5f7;border:1px solid #e7eaee;"
            f"border-radius:999px;padding:2px 8px;margin:0 6px 0 0;font-size:12px;color:#333'>{label}</span>")

def card(sym: str, name: str, img: str, met: dict | None) -> str:
    chips = ""
    if met:
        chips = "".join([
            chip(f"63d {fmt(met.get('ret63'),2,True)}"),
            chip(f"σ₆₃ {fmt(met.get('sig63'),2,True)}"),
            chip(f"SNR {fmt(met.get('snr'),2,False)}"),
            chip(f"ATR₁₄ {fmt(met.get('atrp'),2,True)}"),
        ])
    return (
        f"<table role='presentation' cellspacing='0' cellpadding='0' style='width:100%;"
        f"border:1px solid #e9ecef;border-radius:12px;overflow:hidden'>"
        f"<tr><td style='padding:{CARD_PAD}'>"
        f"<img src='cid:{img}' width='{IMG_W}' height='{IMG_H}' "
        f"style='display:block;border:0;border-radius:8px' alt='{sym} mini-chart'>"
        f"<div style='margin-top:8px;font-weight:700'>{sym}</div>"
        f"<div style='color:#666;font-size:13px;margin:2px 0 6px 0'>{name}</div>"
        f"<div>{chips}</div>"
        f"</td></tr></table>"
    )

def section_cards(title: str, rows: list[tuple[str, str, str]], metrics: dict[str,dict]) -> str:
    if not rows:
        return ""
    # two columns (email-safe via table layout)
    html = [f"<h3 style='margin:12px 0'>{title}</h3>",
            "<table role='presentation' cellspacing='0' cellpadding='0' style='width:100%'>"]
    for i in range(0, len(rows), 2):
        html.append("<tr>")
        for j in range(2):
            if i+j < len(rows):
                s, nm, img = rows[i+j]
                met = metrics.get(s, {})
                html.append(
                    f"<td valign='top' style='padding:6px;width:{CELL_W}px'>{card(s, nm, img, met)}</td>"
                )
            else:
                html.append("<td></td>")
        html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)

def table_metrics(rows, breakout=False) -> str:
    if not rows: return ""
    cols = ["Symbol","Name","Price","63d","σ₆₃","SNR","ATR₁₄%"] + (["Gap70%"] if breakout else [])
    head = "".join(f"<th style='padding:6px 8px;text-align:left;font-weight:600;border-bottom:1px solid #eee'>{c}</th>" for c in cols)
    body = []
    for r in rows:
        cells = [
            r["sym"], r["name"], fmt(r["price"],2,False),
            fmt(r["ret63"],2,True), fmt(r["sig63"],2,True),
            fmt(r["snr"],2,False), fmt(r["atrp"],2,True)
        ]
        if breakout: cells.append(fmt(r.get("gap70", float("nan")),2,True))
        tds = "".join(f"<td style='padding:6px 8px;border-bottom:1px solid #f6f6f6'>{c}</td>" for c in cells)
        body.append(f"<tr>{tds}</tr>")
    return ("<table style='border-collapse:collapse;width:100%;margin:6px 0 10px 0'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


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

    # ---------- Build images ----------
    mom_rows: list[tuple[str, str, str]] = []
    for s in mom_syms:
        nm  = name_of(s)
        img = outdir / f"MOM_{s}.png"
        dts, op, hi, lo, cl = ohlc(s, 63)
        mini_candles(dts, op, hi, lo, cl, img)
        mom_rows.append((s, nm, img.name))

    brk_rows: list[tuple[str, str, str]] = []
    for s, nm in brk_pairs:
        if not nm:
            nm = name_of(s)
        img = outdir / f"BO_{s}.png"
        dts, op, hi, lo, cl = ohlc(s, 63)
        mini_candles(dts, op, hi, lo, cl, img)
        brk_rows.append((s, nm, img.name))

    # ---------- Beauty metrics (and per-card chips) ----------
    mom_metrics_list, brk_metrics_list = [], []
    mom_metrics_map: dict[str,dict] = {}
    for s in mom_syms:
        df = fetch_ohlc_full(s, 400)
        if df is None: continue
        met = {
            "sym": s, "name": name_of(s),
            "price": df["close"].iloc[-1],
            "ret63": ret63_pct(df),
            "sig63": sigma63_pct(df),
            "snr":   snr63(df),
            "atrp":  atr14_pct(df)
        }
        mom_metrics_list.append(met)
        mom_metrics_map[s] = met
    brk_metrics_map: dict[str,dict] = {}
    for s, nm in brk_pairs:
        df = fetch_ohlc_full(s, 400)
        if df is None: continue
        met = {
            "sym": s, "name": nm or name_of(s),
            "price": df["close"].iloc[-1],
            "ret63": ret63_pct(df),
            "sig63": sigma63_pct(df),
            "snr":   snr63(df),
            "atrp":  atr14_pct(df),
            "gap70": gap_vs_prior70(df)
        }
        brk_metrics_list.append(met)
        brk_metrics_map[s] = met

    avg_corr, vol5, avg_atr = momentum_portfolio_stats(
        {s: fetch_ohlc_full(s, 400) for s in mom_syms if fetch_ohlc_full(s, 400) is not None}
    )

    beauty_block = (
        "<div style='border:1px solid #e9ecef;border-radius:12px;padding:12px;margin:10px 0'>"
        "<div style='font-weight:700;margin-bottom:6px'>Beauty panel — signal & risk</div>"
        "<div style='color:#444;font-size:13px'>"
        f"Momentum basket: avg pairwise corr <b>{fmt(avg_corr,2,False)}</b> · "
        f"est EW vol (5d) <b>{fmt(vol5,2,True)}</b> · "
        f"avg ATR₁₄ <b>{fmt(avg_atr,2,True)}</b>"
        "</div>"
        "</div>"
        "<div style='margin-top:6px'><b>Momentum picks — metrics</b></div>"
        f"{table_metrics(mom_metrics_list, breakout=False)}"
        "<div style='margin-top:6px'><b>Breakouts — Top-10 metrics</b></div>"
        f"{table_metrics(brk_metrics_list, breakout=True)}"
    )

    # ---------- HTML ----------
    html_parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<div style='font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif'>",
        "<h2 style='margin:0 0 8px'>IW Bot — Weekly Summary</h2>",
        beauty_block,
        section_cards("Momentum picks (3-month mini-candles)", mom_rows, mom_metrics_map),
        section_cards("Breakouts — Top-10 (3-month mini-candles)", brk_rows, brk_metrics_map),
        "<p style='color:#888;font-size:12px;margin-top:12px'>"
        "Mini-charts: daily candlesticks with EMA(20); weekends compressed out."
        "</p></div>"
    ]
    out = p.Path(args.outdir) / "email.html"
    out.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"[email] momentum charts: {len(mom_rows)} @ ~{IMG_W}x{IMG_H}px")
    print(f"[email] breakout charts: {len(brk_rows)} @ ~{IMG_W}x{IMG_H}px")
    print(f"[beauty] momentum metrics: {len(mom_metrics_list)} | breakout metrics: {len(brk_metrics_list)}")
    if hi70_path and hi70_path.exists():
        print(f"[email] used hi70: {hi70_path}")
    else:
        print("[email] WARNING: hi70 file not found, breakout section may be empty")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
API-only weekly RSI picklist builder (Polygon).

- Reads symbol list from --symbols-file (e.g. backtests/universe_topvol.txt)
- Fetches daily OHLCV from Polygon GROUPED DAILY endpoint for a lookback window
- Computes RSI(14) + extension vs SMA(20), applies same trend guards as before
- Writes: week_start,symbol,rank,score,filters

Requires:
  POLYGON_API_KEY in env

No local data dirs. No CSV scanning.
"""

import argparse, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import requests


POLYGON_KEY = os.getenv("POLYGON_API_KEY", "").strip()
POLYGON_GROUPED = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day}"


def next_monday(d: date) -> date:
    days = (7 - d.weekday()) % 7
    if days == 0:
        days = 7
    return d + timedelta(days=days)


def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing (EMA alpha=1/14)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def two_consecutive_lower_fridays(sub: pd.DataFrame) -> bool:
    if sub.empty:
        return False
    wk = sub.set_index("Date")["Close"].resample("W-FRI").last().dropna()
    if len(wk) < 3:
        return False
    return (wk.iloc[-1] < wk.iloc[-2]) and (wk.iloc[-2] < wk.iloc[-3])


def read_symbols_file(p: Path) -> list[str]:
    if not p.exists():
        raise SystemExit(f"ERROR: symbols-file not found: {p}")
    syms = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            syms.append(s)
    if not syms:
        raise SystemExit(f"ERROR: symbols-file is empty: {p}")
    return syms


def trading_days_range(until_d: date, lookback_trading_days: int) -> list[date]:
    # Approx: 5 trading days per 7 calendar days => multiply by ~1.6–2.0 for safety
    cal_lookback = int(max(30, lookback_trading_days * 2))
    start = until_d - timedelta(days=cal_lookback)
    days = []
    d = start
    while d <= until_d:
        days.append(d)
        d += timedelta(days=1)
    return days


def fetch_grouped_day(day: date, session: requests.Session) -> pd.DataFrame:
    url = POLYGON_GROUPED.format(day=day.isoformat())
    params = {"adjusted": "true", "apiKey": POLYGON_KEY}
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Polygon grouped {day} HTTP {r.status_code}: {r.text[:200]}")
    j = r.json() or {}
    res = j.get("results") or []
    if not res:
        return pd.DataFrame(columns=["Date", "symbol", "Open", "High", "Low", "Close", "Volume"])

    # Polygon grouped results keys:
    # T=ticker, o/h/l/c/v
    df = pd.DataFrame(res)
    if "T" not in df.columns:
        return pd.DataFrame(columns=["Date", "symbol", "Open", "High", "Low", "Close", "Volume"])

    out = pd.DataFrame({
        "Date": pd.to_datetime([day] * len(df)),
        "symbol": df["T"].astype(str).str.upper(),
        "Open": pd.to_numeric(df.get("o"), errors="coerce"),
        "High": pd.to_numeric(df.get("h"), errors="coerce"),
        "Low": pd.to_numeric(df.get("l"), errors="coerce"),
        "Close": pd.to_numeric(df.get("c"), errors="coerce"),
        "Volume": pd.to_numeric(df.get("v"), errors="coerce"),
    })
    out = out.dropna(subset=["Close"])
    return out


def fetch_history_for_symbols(symbols: list[str], until_d: date, lookback_trading_days: int) -> pd.DataFrame:
    if not POLYGON_KEY:
        raise SystemExit("ERROR: missing POLYGON_API_KEY env var")

    wanted = set(symbols)
    days = trading_days_range(until_d, lookback_trading_days)

    frames = []
    with requests.Session() as sess:
        for i, d in enumerate(days):
            # Weekend skip (Polygon grouped may return empty anyway, but this saves calls)
            if d.weekday() >= 5:
                continue
            try:
                day_df = fetch_grouped_day(d, sess)
            except Exception as e:
                # One transient failure shouldn't kill everything; but keep it loud
                raise RuntimeError(f"Failed grouped fetch for {d}: {e}") from e

            if not day_df.empty:
                day_df = day_df[day_df["symbol"].isin(wanted)]
                if not day_df.empty:
                    frames.append(day_df)

            # be polite; grouped endpoint is heavier
            if (i % 5) == 0:
                time.sleep(0.15)

    if not frames:
        return pd.DataFrame(columns=["Date","symbol","Open","High","Low","Close","Volume"])
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "Date"])
    return df


def build_once_api(sym_hist: pd.DataFrame, until_d: date, rsi_min: float, max_ext20: float, top_per_week: int) -> pd.DataFrame:
    rows = []
    if sym_hist.empty:
        return pd.DataFrame()

    # Group by symbol
    for sym, df in sym_hist.groupby("symbol", sort=False):
        sub = df[df["Date"].dt.date <= until_d].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("Date")
        if len(sub) < 30:
            continue

        close = sub["Close"].astype(float)
        sma5  = close.rolling(5,  min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        slope20 = sma20 - sma20.shift(5)

        rsi = rsi14(close)

        last_rsi   = float(rsi.iloc[-1])   if pd.notna(rsi.iloc[-1])   else np.nan
        last_sma20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else np.nan
        last_sma5  = float(sma5.iloc[-1])  if pd.notna(sma5.iloc[-1])  else np.nan
        last_close = float(close.iloc[-1])

        if not (np.isfinite(last_rsi) and np.isfinite(last_sma20) and np.isfinite(last_sma5)):
            continue

        # trend guards (same as your original)
        if last_close <= last_sma20:
            continue
        if last_sma5 <= last_sma20:
            continue
        if not (float(slope20.iloc[-1]) > 0):
            continue
        if two_consecutive_lower_fridays(sub.rename(columns={"symbol":"Symbol"})[["Date","Close"]]):
            # two_consecutive_lower_fridays expects Date/Close; symbol not needed
            continue

        ext20 = abs((last_close / last_sma20) - 1.0)

        if last_rsi >= rsi_min and ext20 <= max_ext20:
            rows.append({"symbol": sym, "rsi": last_rsi, "ext20": ext20})

    if not rows:
        return pd.DataFrame()

    dfc = pd.DataFrame(rows).sort_values("rsi", ascending=False)
    return dfc.head(top_per_week).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="Text file with one ticker per line (e.g. backtests/universe_topvol.txt)")
    ap.add_argument("--out", default=r"backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--rsi-min", type=float, default=68.0)
    ap.add_argument("--max-ext20", type=float, default=0.15)
    ap.add_argument("--top-per-week", type=int, default=40)
    ap.add_argument("--until", default="")
    ap.add_argument("--lookback-days", type=int, default=80, help="Approx trading-day lookback window (default 80)")
    args = ap.parse_args()

    if args.until:
        until_d = datetime.strptime(args.until, "%Y-%m-%d").date()
    else:
        until_d = date.today()

    week_start = next_monday(until_d)

    symbols = read_symbols_file(Path(args.symbols_file))
    hist = fetch_history_for_symbols(symbols, until_d, args.lookback_days)

    picks = build_once_api(hist, until_d, args.rsi_min, args.max_ext20, args.top_per_week)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        f.write("week_start,symbol,rank,score,filters\n")
        for i, r in picks.iterrows():
            filt = (
                f"source=polygon_grouped; rsi_min={args.rsi_min}; ext20<={args.max_ext20}; rank=rsi; "
                f"trend=Close>SMA20 & SMA5>SMA20 & SMA20_slope>0 & !two_down_fridays"
            )
            f.write(f"{week_start},{r['symbol']},{i+1},{float(r['rsi']):.2f},{filt}\n")

    print(f"Wrote {outp} | week_start={week_start} | rows={len(picks)} | symbols={len(symbols)} | until={until_d}")


if __name__ == "__main__":
    main()

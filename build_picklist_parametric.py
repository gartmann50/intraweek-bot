#!/usr/bin/env python3
"""
Minimal weekly RSI picklist builder for CI.

- Reads daily OHLCV csv files from --data-dir (default: stock_data_400/)
  (files like AAPL.csv, MSFT.csv, etc.; requires Date, Open, High, Low, Close, Volume).
- Computes RSI(14) and a simple extension filter vs 20-day SMA.
- Keeps rows where RSI >= --rsi-min and |Close/SMA20 - 1| <= --max-ext20
- Ranks by RSI desc, writes one-week picklist:
    week_start,symbol,rank,score,filters
  to --out (default: backtests/picklist_highrsi_trend.csv)
"""

import argparse, os, sys
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

def next_monday(d: date) -> date:
    days = (7 - d.weekday()) % 7
    if days == 0:
        days = 7
    return d + timedelta(days=days)

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # find Date col name, case-insensitive
    date_col = None
    for c in df.columns:
        if str(c).lower() == "date":
            date_col = c
            break
    if not date_col:
        raise ValueError(f"No Date column in {p.name}")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    # normalize OHLCV case
    for k in ["Open", "High", "Low", "Close", "Volume"]:
        if k not in df.columns:
            matches = [c for c in df.columns if str(c).lower() == k.lower()]
            if matches:
                df[k] = df[matches[0]]
            else:
                raise ValueError(f"Missing {k} in {p.name}")
    return df[["Date","Open","High","Low","Close","Volume"]]

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing (EMA with alpha=1/14)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def build_once(data_dir: Path, until_d: date, rsi_min: float, max_ext20: float, top_per_week: int):
    rows = []
    files = sorted(list(data_dir.glob("*.csv")))
    for i, p in enumerate(files):
        sym = p.stem.upper().split("_")[0]  # strip possible _5YEAR_DATA
        try:
            df = load_csv(p)
        except Exception:
            continue
        if df.empty or df["Date"].iloc[-1].date() > until_d:
            # still ok; we’ll use the latest row we have
            pass
        # compute RSI & ext vs SMA20
        close = df["Close"].astype(float)
        sma20 = close.rolling(20, min_periods=20).mean()
        rsi = rsi14(close)
        latest = df.iloc[-1]
        rsi_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
        sma_last = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else np.nan
        if not np.isfinite(rsi_last) or not np.isfinite(sma_last):
            continue
        ext20 = abs((latest["Close"] / sma_last) - 1.0)
        if rsi_last >= rsi_min and ext20 <= max_ext20:
            rows.append(
                {"symbol": sym, "rsi": rsi_last, "ext20": ext20}
            )
    if not rows:
        return []
    dfc = pd.DataFrame(rows).sort_values("rsi", ascending=False)
    return dfc.head(top_per_week).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="stock_data_400")
    ap.add_argument("--out", default=r"backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--rsi-min", type=float, default=68.0)
    ap.add_argument("--max-ext20", type=float, default=0.15)
    ap.add_argument("--top-per-week", type=int, default=40)
    ap.add_argument("--until", default="")
    args = ap.parse_args()

    # UNTIL date (Europe/Oslo already handled by workflow; here just parse yyyy-mm-dd or use today)
    if args.until:
        until_d = datetime.strptime(args.until, "%Y-%m-%d").date()
    else:
        until_d = date.today()
    week_start = next_monday(until_d)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(2)

    picks = build_once(data_dir, until_d, args.rsi_min, args.max_ext20, args.top_per_week)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        f.write("week_start,symbol,rank,score,filters\n")
        if len(picks) == 0:
            pass
        else:
            for i, r in picks.iterrows():
                filt = f"preset=classic; rsi_min={args.rsi_min}; ext20<={args.max_ext20}; rank=rsi"
                f.write(f"{week_start},{r['symbol']},{i+1},{r['rsi']:.2f},{filt}\n")

    print(f"Wrote {outp} | week_start={week_start} | rows={len(picks)}")

if __name__ == "__main__":
    main()

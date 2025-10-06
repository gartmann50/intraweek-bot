#!/usr/bin/env python3
"""
Minimal weekly RSI picklist builder for CI with light trend guards.

Reads daily OHLCV CSV files from --data-dir (e.g. stock_data_400/),
computes RSI(14) + extension vs SMA(20), then applies *trend filters*:

  ✓ Allow at most ONE down week — reject if the last two Fridays are
    consecutively lower than the Friday before them.
  ✓ Require Close > SMA(20) on the 'until' date.
  ✓ Require SMA(5) > SMA(20) on the 'until' date.
  ✓ Require SMA(20) slope > 0 (today's SMA20 > SMA20 5 bars ago).

Keeps rows where: RSI >= --rsi-min AND |Close/SMA20 - 1| <= --max-ext20 AND trend_ok
Ranks by RSI desc and writes one-week picklist:
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

def two_consecutive_lower_fridays(sub: pd.DataFrame) -> bool:
    """
    Return True if the last TWO weekly (Fri) closes are each lower than the prior Friday close.
    (i.e., week[-1] < week[-2] AND week[-2] < week[-3])
    """
    if sub.empty:
        return False
    # end-of-week (Friday) close; if Fri is holiday, W-FRI takes the last value in that week
    wk = sub.set_index("Date")["Close"].resample("W-FRI").last().dropna()
    if len(wk) < 3:
        return False
    return (wk.iloc[-1] < wk.iloc[-2]) and (wk.iloc[-2] < wk.iloc[-3])

def build_once(data_dir: Path, until_d: date, rsi_min: float, max_ext20: float, top_per_week: int):
    rows = []
    files = sorted(list(data_dir.glob("*.csv")))
    for i, p in enumerate(files):
        sym = p.stem.upper().split("_")[0]  # strip possible _SUFFIX
        try:
            df = load_csv(p)
        except Exception:
            continue

        # clip to 'until' date so we don't peek beyond the anchor
        sub = df[df["Date"].dt.date <= until_d].copy()
        if sub.empty:
            continue

        # need enough bars for SMA20 & slope
        if len(sub) < 30:
            continue

        close = sub["Close"].astype(float)
        sma5  = close.rolling(5,  min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        slope20 = sma20 - sma20.shift(5)  # ~1 week slope

        rsi = rsi14(close)

        last_rsi   = float(rsi.iloc[-1])   if pd.notna(rsi.iloc[-1])   else np.nan
        last_sma20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else np.nan
        last_sma5  = float(sma5.iloc[-1])  if pd.notna(sma5.iloc[-1])  else np.nan
        last_close = float(close.iloc[-1])

        if not (np.isfinite(last_rsi) and np.isfinite(last_sma20) and np.isfinite(last_sma5)):
            continue

        # --- trend guards ---
        if last_close <= last_sma20:            # Close > SMA20
            continue
        if last_sma5 <= last_sma20:             # SMA5 > SMA20
            continue
        if not (float(slope20.iloc[-1]) > 0):   # SMA20 slope > 0 vs ~1 week ago
            continue
        if two_consecutive_lower_fridays(sub):  # reject 2 down Fridays in a row
            continue

        # extension vs SMA20
        ext20 = abs((last_close / last_sma20) - 1.0)

        # base signal filters
        if last_rsi >= rsi_min and ext20 <= max_ext20:
            rows.append(
                {
                    "symbol": sym,
                    "rsi": last_rsi,
                    "ext20": ext20,
                    # optional debug columns (not written to CSV, but handy if you print dfc)
                    # "close": last_close, "sma20": last_sma20, "sma5": last_sma5, "slope20": float(slope20.iloc[-1])
                }
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

    # UNTIL date: parse yyyy-mm-dd or default to today
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
                filt = (
                    f"preset=classic; rsi_min={args.rsi_min}; ext20<={args.max_ext20}; rank=rsi; "
                    f"trend=Close>SMA20 & SMA5>SMA20 & SMA20_slope>0 & !two_down_fridays"
                )
                f.write(f"{week_start},{r['symbol']},{i+1},{r['rsi']:.2f},{filt}\n")

    print(f"Wrote {outp} | week_start={week_start} | rows={len(picks)}")

if __name__ == "__main__":
    main()

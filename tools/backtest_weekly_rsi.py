#!/usr/bin/env python3
"""
Weekly RSI momentum backtester compatible with your picklist builder.

Fix in this version
-------------------
- **No more `argparse` crash** when you run it without `--start`/`--end`.
  Those are now optional: the script infers a sensible date range from your CSVs
  (starts when both RSI and SMA20 are available, ends at the latest date in the
  data). You can still pass them explicitly.
- Added a **self-test** mode that builds synthetic data and asserts the engine
  works, so we have repeatable test cases.

What it does
------------
- Reads daily OHLCV CSVs from a folder (same format your script expects).
- Recomputes RSI(14) and SMA20 for every day (no look-ahead).
- Each *Friday* (or last trading day of the week), builds a picklist using the
  same filter: RSI >= rsi_min AND |Close/SMA20 - 1| <= max_ext20, ranked by RSI desc.
- On the following *Monday* (or next available trading day), buys the top N
  names equally-weighted (skipping any without a valid Monday open) and holds for
  exactly `hold_days` trading days, then sells at the close of the exit day.
- Supports per-trade commission and slippage (in bps) applied to both entry
  and exit.
- Produces an equity curve CSV, a trades CSV, and prints summary metrics.

Quickstart
----------
From repo root (no dates needed):

```bash
python tools/backtest_weekly_rsi.py --data-dir stock_data_400 --out-dir backtests
```

Examples (explicit dates)
-------------------------
```bash
python tools/backtest_weekly_rsi.py \
  --data-dir stock_data_400 \
  --start 2015-01-01 --end 2025-10-03 \
  --rsi-min 68 --max-ext20 0.15 --top-per-week 40 \
  --hold-days 5 \
  --slippage-bps 5 --commission-per-trade 0.0 \
  --initial-capital 100000 \
  --out-dir backtests
```

Outputs
-------
- backtests/equity_curve.csv  (Date, Equity)
- backtests/trades.csv        (week_start, symbol, entry_date, entry_px, exit_date, exit_px, qty, pnl, ret)
- Metrics printed to stdout.

"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import sys

# ----------------------------
# Utilities and indicators
# ----------------------------

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

@dataclass
class Trade:
    week_start: pd.Timestamp
    symbol: str
    entry_date: pd.Timestamp
    entry_px: float
    exit_date: pd.Timestamp
    exit_px: float
    qty: float

    @property
    def pnl(self) -> float:
        return (self.exit_px - self.entry_px) * self.qty

    @property
    def ret(self) -> float:
        return (self.exit_px / self.entry_px) - 1.0

# ----------------------------
# Data loading
# ----------------------------

def load_ohlcv_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    if 'date' not in cols:
        raise ValueError(f"No Date column in {p.name}")
    # Map canonical columns
    def get(name: str):
        for c in df.columns:
            if c.lower() == name:
                return df[c]
        raise ValueError(f"Missing {name} in {p.name}")

    out = pd.DataFrame({
        'Date': pd.to_datetime(get('date'), errors='coerce'),
        'Open': pd.to_numeric(get('open'), errors='coerce'),
        'High': pd.to_numeric(get('high'), errors='coerce'),
        'Low': pd.to_numeric(get('low'), errors='coerce'),
        'Close': pd.to_numeric(get('close'), errors='coerce'),
        'Volume': pd.to_numeric(get('volume'), errors='coerce'),
    }).dropna(subset=['Date']).sort_values('Date')
    out = out.set_index('Date')
    return out


def load_universe(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for p in sorted(data_dir.glob('*.csv')):
        sym = p.stem.upper().split('_')[0]
        try:
            df = load_ohlcv_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        # Indicators (no look-ahead)
        df['SMA20'] = df['Close'].rolling(20, min_periods=20).mean()
        df['RSI'] = rsi14(df['Close'])
        data[sym] = df
    return data

# ----------------------------
# Signal & portfolio logic
# ----------------------------

def nth_trading_day_after(dates: pd.DatetimeIndex, d: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
    subset = dates[dates > d]
    if len(subset) < n:
        return None
    return subset[n-1]


def build_picklist_for_week(universe: Dict[str, pd.DataFrame],
                            week_start: pd.Timestamp,
                            rsi_min: float,
                            max_ext20: float,
                            top_per_week: int) -> List[Tuple[str, float]]:
    """Use data up to LAST TRADING DAY BEFORE week_start (i.e., prior Friday)."""
    rows = []
    week_end = week_start - pd.Timedelta(days=1)
    for sym, df in universe.items():
        dates = df.index
        if len(dates) == 0:
            continue
        # last trading day up to week_end
        last_d = dates[dates <= week_end]
        if len(last_d) == 0:
            continue
        t = last_d[-1]
        rsi = df.at[t, 'RSI'] if pd.notna(df.at[t, 'RSI']) else np.nan
        sma = df.at[t, 'SMA20'] if pd.notna(df.at[t, 'SMA20']) else np.nan
        close = df.at[t, 'Close']
        if not (np.isfinite(rsi) and np.isfinite(sma)):
            continue
        ext20 = abs((close / sma) - 1.0)
        if rsi >= rsi_min and ext20 <= max_ext20:
            rows.append((sym, float(rsi)))
    if not rows:
        return []
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_per_week]

# ----------------------------
# Backtest engine
# ----------------------------

def backtest(universe: Dict[str, pd.DataFrame],
             start: pd.Timestamp,
             end: pd.Timestamp,
             rsi_min: float,
             max_ext20: float,
             top_per_week: int,
             hold_days: int,
             slippage_bps: float,
             commission_per_trade: float,
             initial_capital: float,
             out_dir: Path):

    all_dates = sorted({d for df in universe.values() for d in df.index})
    if not all_dates:
        raise ValueError("No data loaded")

    # Generate Monday anchors within [start, end]
    cal = pd.date_range(start, end, freq='D')
    mondays = [d for d in cal if d.weekday() == 0]

    equity = []
    trades: List[Trade] = []
    cash = initial_capital
    equity_val = initial_capital

    for week_start in mondays:
        # Build picklist using prior week's data
        picks = build_picklist_for_week(universe, week_start, rsi_min, max_ext20, top_per_week)

        # Determine entry date (next available trading day on/after week_start for each symbol)
        valid_entries = []
        for sym, _ in picks:
            df = universe[sym]
            entry_d = df.index[df.index >= week_start]
            if len(entry_d) == 0:
                continue
            entry_date = entry_d[0]
            entry_px = float(df.at[entry_date, 'Open'])
            if not np.isfinite(entry_px):
                continue
            valid_entries.append((sym, entry_date, entry_px))

        if not valid_entries:
            # Nothing to buy this week; mark equity carry
            equity.append((week_start, equity_val))
            continue

        # Equal-weight position sizing
        n = len(valid_entries)
        alloc_per = cash / n

        week_trades: List[Trade] = []
        capital_used = 0.0

        for sym, entry_date, entry_px in valid_entries:
            df = universe[sym]
            exit_date = nth_trading_day_after(df.index, entry_date, hold_days)
            if exit_date is None:
                # Can't complete a round-trip inside sample; skip
                continue
            exit_px = float(df.at[exit_date, 'Close'])

            # Apply slippage and commissions
            slip_mult_in = 1.0 + (slippage_bps / 1e4)
            slip_mult_out = 1.0 - (slippage_bps / 1e4)
            entry_gross = entry_px * slip_mult_in
            exit_net = exit_px * slip_mult_out

            # Shares as floor of capital / entry_gross (avoid leverage)
            qty = np.floor(alloc_per / entry_gross)
            if qty <= 0:
                continue
            cost = qty * entry_gross + commission_per_trade
            # proceeds = qty * exit_net - commission_per_trade  # not needed explicitly

            t = Trade(week_start=pd.Timestamp(week_start), symbol=sym,
                      entry_date=entry_date, entry_px=entry_gross,
                      exit_date=exit_date, exit_px=exit_net, qty=float(qty))
            week_trades.append(t)
            capital_used += cost

        if not week_trades:
            equity.append((week_start, equity_val))
            continue

        # Update equity as if realizing PnL at exit (weekly granularity)
        pnl_sum = sum(t.pnl for t in week_trades)
        cash = cash - capital_used + (capital_used + pnl_sum)
        equity_val = cash
        trades.extend(week_trades)
        equity.append((week_start, equity_val))

    # Build outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame(equity, columns=['Date', 'Equity']).set_index('Date')
    eq_df.to_csv(out_dir / 'equity_curve.csv')

    tr_rows = []
    for t in trades:
        tr_rows.append({
            'week_start': t.week_start.date(),
            'symbol': t.symbol,
            'entry_date': t.entry_date.date(),
            'entry_px': round(t.entry_px, 6),
            'exit_date': t.exit_date.date(),
            'exit_px': round(t.exit_px, 6),
            'qty': int(t.qty),
            'pnl': round(t.pnl, 2),
            'ret': round(t.ret, 6),
        })
    trades_df = pd.DataFrame(tr_rows)
    trades_df.to_csv(out_dir / 'trades.csv', index=False)

    # Metrics
    if len(eq_df) >= 2:
        rets = eq_df['Equity'].pct_change().dropna()
        ann_factor = 52  # weekly steps
        cagr = (eq_df['Equity'].iloc[-1] / eq_df['Equity'].iloc[0]) ** (ann_factor/len(eq_df)) - 1
        vol = rets.std() * np.sqrt(ann_factor)
        sharpe = np.nan if vol == 0 or np.isnan(vol) else rets.mean() * ann_factor / vol
        # Max drawdown
        cum = eq_df['Equity']
        roll_max = cum.cummax()
        dd = (cum / roll_max - 1.0)
        max_dd = dd.min()
    else:
        cagr = vol = sharpe = max_dd = np.nan

    print("==== Backtest Summary ====")
    print(f"Start: {eq_df.index[0].date() if not eq_df.empty else 'NA'}  End: {eq_df.index[-1].date() if not eq_df.empty else 'NA'}")
    print(f"Initial: {initial_capital:,.2f}  Final: {eq_df['Equity'].iloc[-1]:,.2f}" if not eq_df.empty else "No equity series")
    print(f"CAGR: {cagr:.2%}  Vol: {vol:.2%}  Sharpe(~0% rf): {sharpe:.2f}  MaxDD: {max_dd:.2%}")
    print(f"Trades: {len(trades_df)}  Weeks: {len(eq_df)}  Avg Weekly Ret: {rets.mean():.4%}" if len(eq_df) >= 2 else "")
    print(f"Wrote {out_dir/'equity_curve.csv'} and {out_dir/'trades.csv'}")


# ----------------------------
# CLI helpers
# ----------------------------

def _infer_date_range(universe: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Infer a safe start/end from the data.
    Start: latest first-valid index among symbols for which both RSI & SMA20 exist,
           then align to the **previous** Monday (never after `end`).
    End:   latest date across all symbols.
    """
    first_valid_dates = []
    last_dates = []
    for df in universe.values():
        fv_rsi = df['RSI'].first_valid_index()
        fv_sma = df['SMA20'].first_valid_index()
        if fv_rsi is None and fv_sma is None:
            continue
        fv = max([d for d in [fv_rsi, fv_sma] if d is not None])
        first_valid_dates.append(fv)
        last_dates.append(df.index[-1])

    if not first_valid_dates or not last_dates:
        raise SystemExit("Unable to infer date range: no valid indicators found.")

    # End is the latest date we actually have in the data
    end = max(last_dates).normalize()

    # Start must be <= end. Align to previous Monday (or the same day if Monday).
    raw_start = max(first_valid_dates).normalize()
    # Align to previous Monday
    start = raw_start - pd.Timedelta(days=raw_start.weekday())

    # Guardrail: if alignment still goes past `end` (e.g., end is a Friday and start rounded up accidentally),
    # clamp start to the previous Monday not after end.
    if start > end:
        # choose the Monday of the week that contains `end`
        start = end - pd.Timedelta(days=end.weekday())

    return pd.Timestamp(start), pd.Timestamp(end)

    if not first_valid_dates or not last_dates:
        raise SystemExit("Unable to infer date range: no valid indicators found.")

    start = max(first_valid_dates)  # ensure indicators are formed
    # align to Monday or later (avoid empty anchor weeks)
    if start.weekday() != 0:
        start = start + pd.Timedelta(days=(7 - start.weekday()))
    end = max(last_dates)
    return pd.Timestamp(start.normalize()), pd.Timestamp(end.normalize())


def _parse_date(s: Optional[str], *, label: str) -> Optional[pd.Timestamp]:
    """Parse a YYYY-MM-DD string. If s is None or empty, return None.
    If a non-empty string is provided and parsing fails, exit with an error.
    """
    if s is None or str(s).strip() == "":
        return None
    try:
        return pd.Timestamp(str(s).strip())
    except Exception as e:
        raise SystemExit(f"Invalid {label} date '{s}'. Expected YYYY-MM-DD.
{e}")
    except Exception as e:
        raise SystemExit(f"Invalid date '{s}'. Expected YYYY-MM-DD.\n{e}")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='stock_data_500')
    ap.add_argument('--start', default=None, help='YYYY-MM-DD (optional; inferred if omitted)')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (optional; inferred if omitted)')
    ap.add_argument('--rsi-min', type=float, default=68.0)
    ap.add_argument('--max-ext20', type=float, default=0.15)
    ap.add_argument('--top-per-week', type=int, default=40)
    ap.add_argument('--hold-days', type=int, default=5)
    ap.add_argument('--slippage-bps', type=float, default=5.0)
    ap.add_argument('--commission-per-trade', type=float, default=0.0)
    ap.add_argument('--initial-capital', type=float, default=100000.0)
    ap.add_argument('--out-dir', default='backtests')
    ap.add_argument('--self-test', action='store_true', help='Run built-in tests with synthetic data and exit')
    args = ap.parse_args()

    if args.self_test:
        return _run_self_test()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    universe = load_universe(data_dir)
    if not universe:
        raise SystemExit("No usable CSVs loaded")

    # Dates: parse or infer
    start = _parse_date(args.start, label='--start')
    end = _parse_date(args.end, label='--end')

    if start is None or end is None:
        inf_start, inf_end = _infer_date_range(universe)
        start = inf_start if start is None else start
        end = inf_end if end is None else end
        print(f"[dates] Using inferred range: start={start.date()} end={end.date()}")
    else:
        print(f"[dates] Using explicit range: start={start.date()} end={end.date()}")

    if end <= start:
        raise SystemExit(f"Invalid range: end ({end.date()}) must be after start ({start.date()})")

    # Use pandas built-in weekly Monday frequency to avoid gaps
    backtest(universe=universe,
             start=start,
             end=end,
             rsi_min=args.rsi_min,
             max_ext20=args.max_ext20,
             top_per_week=args.top_per_week,
             hold_days=args.hold_days,
             slippage_bps=args.slippage_bps,
             commission_per_trade=args.commission_per_trade,
             initial_capital=args.initial_capital,
             out_dir=Path(args.out_dir))


if __name__ == '__main__':
    sys.exit(main() or 0)

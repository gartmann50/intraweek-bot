#!/usr/bin/env python3
"""
Weekly RSI momentum backtester with in-week hard & trailing stops.

Implements your live flow:
- Friday: build picklist (RSI + extension + optional trend/weekly-up filters), take Top-K (default 6).
- Monday (next trading day): enter all picks at Open.
- During the week:
  * Hard stop: exit if Close <= entry * (1 - abs_stop_pct); exit next trading day's Open.
  * Trailing stop (optional): stop ratchets with new closing highs: stop = max(hard_stop, peak_close * (1 - trail_pct)).
- End of week: any survivors exit at that week's last trading day's Close.
- Equal-weight across the week's picks; long-only; no re-entries mid-week.

CLI toggles (easy to iterate):
  --top-per-week 6
  --rsi-min 68 --max-ext20 0.15
  --trend-sma 100 --weekly-up-only
  --use-stops --abs-stop-pct 0.02 --trail-pct 0.02
  --slippage-bps 5 --commission-per-trade 0

Usage example
-------------
python tools/backtest_weekly_rsi_stops.py \
  --data-dir stock_data_500 \
  --start 2021-01-01 --end 2025-10-05 \
  --rsi-min 68 --max-ext20 0.15 \
  --trend-sma 100 --weekly-up-only \
  --top-per-week 6 \
  --use-stops --abs-stop-pct 0.02 --trail-pct 0.02 \
  --out-dir backtests
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import sys

# ----------------------------
# Indicators & data loading
# ----------------------------

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

@dataclass
class Trade:
    week_start: pd.Timestamp
    symbol: str
    entry_date: pd.Timestamp
    entry_px: float
    exit_date: pd.Timestamp
    exit_px: float
    qty: float
    exit_reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_px - self.entry_px) * self.qty

    @property
    def ret(self) -> float:
        return (self.exit_px / self.entry_px) - 1.0


def load_ohlcv_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if 'date' not in cols:
        raise ValueError(f"No Date column in {p.name}")
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
    out['SMA20'] = out['Close'].rolling(20, min_periods=20).mean()
    out['RSI'] = rsi14(out['Close'])
    return out


def load_universe(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for p in sorted(data_dir.glob('*.csv')):
        sym = p.stem.upper().split('_')[0]
        try:
            df = load_ohlcv_csv(p)
        except Exception:
            continue
        if not df.empty:
            data[sym] = df
    return data

# ----------------------------
# Helpers
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
                            top_per_week: int,
                            trend_sma: int = 0,
                            weekly_up_only: bool = False) -> List[Tuple[str, float]]:
    """Signals from last trading day before week_start (prior Friday)."""
    rows = []
    week_end = week_start - pd.Timedelta(days=1)
    for sym, df in universe.items():
        dates = df.index
        last_d = dates[dates <= week_end]
        if len(last_d) == 0:
            continue
        t = last_d[-1]
        rsi = df.at[t, 'RSI'] if pd.notna(df.at[t, 'RSI']) else np.nan
        sma20 = df.at[t, 'SMA20'] if pd.notna(df.at[t, 'SMA20']) else np.nan
        c = df.at[t, 'Close']
        if not (np.isfinite(rsi) and np.isfinite(sma20)):
            continue
        if abs((c / sma20) - 1.0) > max_ext20 or rsi < rsi_min:
            continue
        if trend_sma and trend_sma > 0:
            col = f'SMA{trend_sma}'
            if col not in df.columns:
                df[col] = df['Close'].rolling(trend_sma, min_periods=trend_sma).mean()
            smaN = df.at[t, col] if pd.notna(df.at[t, col]) else np.nan
            if not np.isfinite(smaN) or c < smaN:
                continue
        if weekly_up_only:
            prev_candidates = dates[dates <= (t - pd.Timedelta(days=7))]
            if len(prev_candidates) == 0:
                continue
            prev_t = prev_candidates[-1]
            if c < df.at[prev_t, 'Close']:
                continue
        rows.append((sym, float(rsi)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_per_week]

# ----------------------------
# Backtest
# ----------------------------

def backtest(universe: Dict[str, pd.DataFrame],
             start: pd.Timestamp,
             end: pd.Timestamp,
             rsi_min: float,
             max_ext20: float,
             top_per_week: int,
             slippage_bps: float,
             commission_per_trade: float,
             initial_capital: float,
             out_dir: Path,
             trend_sma: int = 0,
             weekly_up_only: bool = False,
             use_stops: bool = True,
             abs_stop_pct: float = 0.02,
             trail_pct: float = 0.02):

    all_dates = sorted({d for df in universe.values() for d in df.index})
    if not all_dates:
        raise ValueError("No data loaded")

    cal = pd.date_range(start, end, freq='D')
    mondays = [d for d in cal if d.weekday() == 0]

    equity = []
    trades: List[Trade] = []
    cash = initial_capital
    equity_val = initial_capital

    for week_start in mondays:
        # signals from prior Friday
        picks = build_picklist_for_week(universe, week_start, rsi_min, max_ext20, top_per_week,
                                        trend_sma=trend_sma, weekly_up_only=weekly_up_only)
        valid_entries = []
        for sym, _ in picks:
            df = universe[sym]
            entry_d = df.index[df.index >= week_start]
            if len(entry_d) == 0:
                continue
            entry_date = entry_d[0]
            entry_px = float(df.at[entry_date, 'Open'])
            if np.isfinite(entry_px):
                valid_entries.append((sym, entry_date, entry_px))
        if not valid_entries:
            equity.append((week_start, equity_val)); continue

        n = len(valid_entries)
        alloc_per = cash / n
        week_trades: List[Trade] = []
        capital_used = 0.0

        for sym, entry_date, entry_px in valid_entries:
            df = universe[sym]
            next_week_start = week_start + pd.Timedelta(days=7)
            week_days = df.index[(df.index >= entry_date) & (df.index < next_week_start)]
            if len(week_days) == 0:
                continue
            # prices & slippage
            in_mult = 1.0 + (slippage_bps/1e4)
            out_mult = 1.0 - (slippage_bps/1e4)
            entry_gross = entry_px * in_mult
            hard_stop = entry_gross * (1.0 - abs_stop_pct) if use_stops else -np.inf

            exit_date = None
            exit_px = None
            peak_close = -np.inf

            for d in week_days:
                close_d = float(df.at[d, 'Close'])
                if use_stops:
                    peak_close = max(peak_close, close_d)
                    trail_stop = peak_close * (1.0 - trail_pct) if trail_pct > 0 else -np.inf
                    stop_lvl = max(hard_stop, trail_stop)
                    if close_d <= stop_lvl:
                        nd = nth_trading_day_after(df.index, d, 1)
                        if nd is None:
                            exit_date = d; exit_px = close_d * out_mult
                        else:
                            exit_date = nd; exit_px = float(df.at[nd, 'Open']) * out_mult
                        reason = 'stop'
                        break
            if exit_date is None:
                last_d = week_days[-1]
                exit_date = last_d
                exit_px = float(df.at[last_d, 'Close']) * out_mult
                reason = 'week_close'

            qty = np.floor(alloc_per / entry_gross)
            if qty <= 0:
                continue
            cost = qty * entry_gross + commission_per_trade
            t = Trade(week_start=pd.Timestamp(week_start), symbol=sym,
                      entry_date=entry_date, entry_px=entry_gross,
                      exit_date=exit_date, exit_px=exit_px, qty=float(qty),
                      exit_reason=reason)
            week_trades.append(t)
            capital_used += cost

        if not week_trades:
            equity.append((week_start, equity_val)); continue

        pnl_sum = sum(t.pnl for t in week_trades)
        cash = cash - capital_used + (capital_used + pnl_sum)
        equity_val = cash
        trades.extend(week_trades)
        equity.append((week_start, equity_val))

    out_dir.mkdir(parents=True, exist_ok=True)
    eq_df = pd.DataFrame(equity, columns=['Date','Equity']).set_index('Date')
    eq_df.to_csv(out_dir / 'equity_curve.csv')

    tr_rows = [{
        'week_start': t.week_start.date(),
        'symbol': t.symbol,
        'entry_date': t.entry_date.date(),
        'entry_px': round(t.entry_px,6),
        'exit_date': t.exit_date.date(),
        'exit_px': round(t.exit_px,6),
        'qty': int(t.qty),
        'pnl': round(t.pnl,2),
        'ret': round(t.ret,6),
        'exit_reason': t.exit_reason,
    } for t in trades]
    pd.DataFrame(tr_rows).to_csv(out_dir / 'trades.csv', index=False)

    if len(eq_df) >= 2:
        rets = eq_df['Equity'].pct_change().dropna()
        ann_factor = 52
        cagr = (eq_df['Equity'].iloc[-1] / eq_df['Equity'].iloc[0]) ** (ann_factor/len(eq_df)) - 1
        vol = rets.std() * np.sqrt(ann_factor)
        sharpe = np.nan if vol == 0 or np.isnan(vol) else rets.mean() * ann_factor / vol
        roll_max = eq_df['Equity'].cummax()
        max_dd = (eq_df['Equity']/roll_max - 1.0).min()
    else:
        cagr = vol = sharpe = max_dd = np.nan
        rets = pd.Series(dtype=float)

    print("==== Backtest Summary ====")
    print(f"Start: {eq_df.index[0].date() if not eq_df.empty else 'NA'}  End: {eq_df.index[-1].date() if not eq_df.empty else 'NA'}")
    print(f"Initial: {initial_capital:,.2f}  Final: {eq_df['Equity'].iloc[-1]:,.2f}" if not eq_df.empty else "No equity series")
    print(f"CAGR: {cagr:.2%}  Vol: {vol:.2%}  Sharpe(~0% rf): {sharpe:.2f}  MaxDD: {max_dd:.2%}")
    if len(eq_df) >= 2:
        print(f"Trades: {len(tr_rows)}  Weeks: {len(eq_df)}  Avg Weekly Ret: {rets.mean():.4%}")
    print(f"Wrote {out_dir/'equity_curve.csv'} and {out_dir/'trades.csv'}")

# ----------------------------
# CLI
# ----------------------------

def _infer_dates_from_data(universe: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    first_valid = []
    last_dates = []
    for df in universe.values():
        fv = max([x for x in [df['RSI'].first_valid_index(), df['SMA20'].first_valid_index()] if x is not None], default=None)
        if fv is not None:
            first_valid.append(fv)
            last_dates.append(df.index[-1])
    if not first_valid or not last_dates:
        raise SystemExit("Unable to infer dates from data")
    end = max(last_dates).normalize()
    raw_start = max(first_valid).normalize()
    start = raw_start - pd.Timedelta(days=raw_start.weekday())
    if start > end:
        start = end - pd.Timedelta(days=end.weekday())
    return pd.Timestamp(start), pd.Timestamp(end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='stock_data_500')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--rsi-min', type=float, default=68.0)
    ap.add_argument('--max-ext20', type=float, default=0.15)
    ap.add_argument('--top-per-week', type=int, default=6)
    ap.add_argument('--trend-sma', type=int, default=0)
    ap.add_argument('--weekly-up-only', action='store_true')
    ap.add_argument('--use-stops', action='store_true', help='Enable in-week stops')
    ap.add_argument('--abs-stop-pct', type=float, default=0.02)
    ap.add_argument('--trail-pct', type=float, default=0.02)
    ap.add_argument('--slippage-bps', type=float, default=5.0)
    ap.add_argument('--commission-per-trade', type=float, default=0.0)
    ap.add_argument('--initial-capital', type=float, default=100000.0)
    ap.add_argument('--out-dir', default='backtests')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    universe = load_universe(data_dir)
    if not universe:
        raise SystemExit("No usable CSVs loaded")

    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    if start is None or end is None:
        start, end = _infer_dates_from_data(universe)
        print(f"[dates] Using inferred range: start={start.date()} end={end.date()}")
    else:
        print(f"[dates] Using explicit range: start={start.date()} end={end.date()}")

    backtest(universe=universe,
             start=start,
             end=end,
             rsi_min=args.rsi_min,
             max_ext20=args.max_ext20,
             top_per_week=args.top_per_week,
             slippage_bps=args.slippage_bps,
             commission_per_trade=args.commission_per_trade,
             initial_capital=args.initial_capital,
             out_dir=Path(args.out_dir),
             trend_sma=args.trend_sma,
             weekly_up_only=args.weekly_up_only,
             use_stops=args.use_stops,
             abs_stop_pct=args.abs_stop_pct,
             trail_pct=args.trail_pct)

if __name__ == '__main__':
    sys.exit(main() or 0)

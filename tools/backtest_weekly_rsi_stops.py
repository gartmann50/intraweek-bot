#!/usr/bin/env python3
"""
Weekly RSI momentum backtester with in-week hard & trailing stops.

Key features
------------
- Signals each **Friday** (last trading day before Monday) with filters:
  * RSI(14) >= rsi_min
  * |Close/SMA20 - 1| <= max_ext20
  * optional trend filter: Close >= SMA(trend_sma)
  * optional weekly-up filter: Friday Close >= prior Friday Close
- Pick **Top-K** by RSI (default 6). Enter **next trading day (Mon) Open**.
- **Stops inside the week** (choose kind):
  * Percent: hard = entry * (1 - abs_stop_pct); trailing = peak_high * (1 - trail_pct)
  * ATR:     hard = entry - atr_mult * ATR;     trailing = peak_high - atr_mult * ATR
  Active stop = max(hard, trailing) when trailing is enabled.
- **Execution model**: 'intraday' (default) exits **same day** if Low <= stop
  at price = min(Open, stop) with slippage; 'next_open' exits next day open.
- Survivors exit on **week close**. Long-only, equal-weight, no mid-week re-entries.
- Outputs: backtests/equity_curve.csv and backtests/trades.csv (with exit_reason).

Default ATR multiple is **1.75** per your request. Switch between percent and ATR
with `--stop-kind pct|atr` (default atr). Toggle trailing by setting `--trail-pct`
(for pct) or `--atr-mult` (for atr). Set `--stop-model intraday|next_open`.
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

def compute_atr(df: pd.DataFrame, win: int) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        (df['High'] - df['Low']).abs(),
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=win).mean()

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
    def pnl(self) -> float: return (self.exit_px - self.entry_px) * self.qty
    @property
    def ret(self) -> float: return (self.exit_px / self.entry_px) - 1.0


def load_ohlcv_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if 'date' not in cols:
        raise ValueError('No Date column in {}'.format(p.name))
    def get(name: str):
        for c in df.columns:
            if c.lower() == name:
                return df[c]
        raise ValueError('Missing {} in {}'.format(name, p.name))
    out = pd.DataFrame({
        'Date':  pd.to_datetime(get('date'), errors='coerce'),
        'Open':  pd.to_numeric(get('open'),  errors='coerce'),
        'High':  pd.to_numeric(get('high'),  errors='coerce'),
        'Low':   pd.to_numeric(get('low'),   errors='coerce'),
        'Close': pd.to_numeric(get('close'), errors='coerce'),
        'Volume':pd.to_numeric(get('volume'),errors='coerce'),
    }).dropna(subset=['Date']).sort_values('Date').set_index('Date')
    # indicators
    out['SMA20'] = out['Close'].rolling(20, min_periods=20).mean()
    out['RSI']   = rsi14(out['Close'])
    return out


def load_universe(data_dir: Path) -> Dict[str, pd.DataFrame]:
    uni: Dict[str, pd.DataFrame] = {}
    for p in sorted(data_dir.glob('*.csv')):
        sym = p.stem.upper().split('_')[0]
        try:
            df = load_ohlcv_csv(p)
        except Exception:
            continue
        if not df.empty:
            uni[sym] = df
    return uni

# ----------------------------
# Helpers & signals
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
    rows = []
    week_end = week_start - pd.Timedelta(days=1)
    for sym, df in universe.items():
        dates = df.index
        last_d = dates[dates <= week_end]
        if len(last_d) == 0:
            continue
        t = last_d[-1]
        rsi = df.at[t, 'RSI']
        sma20 = df.at[t, 'SMA20']
        c = df.at[t, 'Close']
        if not (pd.notna(rsi) and pd.notna(sma20)):
            continue
        if abs((c / sma20) - 1.0) > max_ext20 or rsi < rsi_min:
            continue
        if trend_sma and trend_sma > 0:
            col = 'SMA{}'.format(trend_sma)
            if col not in df.columns:
                df[col] = df['Close'].rolling(trend_sma, min_periods=trend_sma).mean()
            smaN = df.at[t, col]
            if not pd.notna(smaN) or c < smaN:
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
             stop_kind: str = 'atr',
             abs_stop_pct: float = 0.02,
             trail_pct: float = 0.02,
             atr_win: int = 14,
             atr_mult: float = 1.75,
             stop_model: str = 'intraday'):

    cal = pd.date_range(start, end, freq='D')
    mondays = [d for d in cal if d.weekday() == 0]

    equity: List[Tuple[pd.Timestamp,float]] = []
    trades: List[Trade] = []
    cash = initial_capital
    equity_val = initial_capital

    for week_start in mondays:
        picks = build_picklist_for_week(universe, week_start, rsi_min, max_ext20, top_per_week,
                                        trend_sma=trend_sma, weekly_up_only=weekly_up_only)

        entries: List[Tuple[str,pd.Timestamp,float]] = []
        for sym, _ in picks:
            df = universe[sym]
            entry_days = df.index[df.index >= week_start]
            if len(entry_days) == 0: continue
            d0 = entry_days[0]
            px0 = float(df.at[d0, 'Open'])
            if np.isfinite(px0): entries.append((sym, d0, px0))

        if not entries:
            equity.append((week_start, equity_val))
            continue

        n = len(entries)
        alloc_per = cash / n
        in_mult  = 1.0 + (slippage_bps / 1e4)
        out_mult = 1.0 - (slippage_bps / 1e4)

        week_trades: List[Trade] = []
        capital_used = 0.0

        for sym, entry_date, entry_px in entries:
            df = universe[sym]
            entry_gross = entry_px * in_mult

            # Precompute ATR if needed
            atr_series = compute_atr(df, atr_win) if stop_kind == 'atr' else None
            atr_entry = float(atr_series.at[entry_date]) if atr_series is not None else np.nan

            # Hard stop
            if use_stops:
                if stop_kind == 'atr':
                    hard_stop = entry_gross - atr_mult * (atr_entry if np.isfinite(atr_entry) else 0.0)
                else:
                    hard_stop = entry_gross * (1.0 - abs_stop_pct)
            else:
                hard_stop = -np.inf

            next_week_start = week_start + pd.Timedelta(days=7)
            week_days = df.index[(df.index >= entry_date) & (df.index < next_week_start)]
            if len(week_days) == 0:
                continue

            exit_date = None
            exit_px   = None
            reason    = 'week_close'
            peak_high = -np.inf

            for d in week_days:
                o = float(df.at[d, 'Open'])
                h = float(df.at[d, 'High'])
                l = float(df.at[d, 'Low'])
                c = float(df.at[d, 'Close'])

                if use_stops:
                    peak_high = max(peak_high, h)
                    if stop_kind == 'atr':
                        atr_d = float(atr_series.at[d]) if atr_series is not None else 0.0
                        trail_stop = peak_high - atr_mult * atr_d
                    else:
                        trail_stop = peak_high * (1.0 - trail_pct) if trail_pct > 0 else -np.inf
                    stop_lvl = max(hard_stop, trail_stop)

                    if stop_model == 'intraday':
                        if l <= stop_lvl:
                            exit_date = d
                            exit_px   = min(o, stop_lvl) * out_mult
                            reason    = 'stop_intraday'
                            break
                    else:  # next_open
                        if c <= stop_lvl:
                            nd = nth_trading_day_after(df.index, d, 1)
                            if nd is None:
                                exit_date = d; exit_px = c * out_mult
                            else:
                                exit_date = nd; exit_px = float(df.at[nd, 'Open']) * out_mult
                            reason = 'stop_next_open'
                            break

            if exit_date is None:
                last_d = week_days[-1]
                exit_date = last_d
                exit_px   = float(df.at[last_d, 'Close']) * out_mult

            qty = np.floor(alloc_per / entry_gross)
            if qty <= 0:
                continue
            capital_used += qty * entry_gross + commission_per_trade
            week_trades.append(Trade(week_start=week_start, symbol=sym,
                                     entry_date=entry_date, entry_px=entry_gross,
                                     exit_date=exit_date, exit_px=exit_px,
                                     qty=float(qty), exit_reason=reason))

        if not week_trades:
            equity.append((week_start, equity_val))
            continue

        pnl = sum(t.pnl for t in week_trades)
        cash = cash - capital_used + (capital_used + pnl)
        equity_val = cash
        equity.append((week_start, equity_val))
        trades.extend(week_trades)

    out_dir.mkdir(parents=True, exist_ok=True)
    eq = pd.DataFrame(equity, columns=['Date','Equity']).set_index('Date')
    eq.to_csv(out_dir / 'equity_curve.csv')

    pd.DataFrame([{
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
    } for t in trades]).to_csv(out_dir / 'trades.csv', index=False)

    # Summary
    if len(eq) >= 2:
        rets = eq['Equity'].pct_change().dropna()
        ann = 52
        cagr = (eq['Equity'].iloc[-1] / eq['Equity'].iloc[0]) ** (ann/len(eq)) - 1
        vol  = rets.std() * np.sqrt(ann)
        sharpe = np.nan if (vol == 0 or np.isnan(vol)) else rets.mean() * ann / vol
        roll_max = eq['Equity'].cummax()
        maxdd = (eq['Equity']/roll_max - 1.0).min()
    else:
        cagr = vol = sharpe = maxdd = np.nan

    print('==== Backtest Summary ====')
    print('Start: {}  End: {}'.format(eq.index[0].date() if not eq.empty else 'NA',
                                     eq.index[-1].date() if not eq.empty else 'NA'))
    if not eq.empty:
        print('Initial: {:,.2f}  Final: {:,.2f}'.format(initial_capital, eq['Equity'].iloc[-1]))
    print('CAGR: {:.2%}  Vol: {:.2%}  Sharpe(~0% rf): {:.2f}  MaxDD: {:.2%}'.format(cagr, vol, sharpe, maxdd))
    if len(eq) >= 2:
        print('Trades: {}  Weeks: {}  Avg Weekly Ret: {:.4%}'.format(len(trades), len(eq), rets.mean()))
    print('Wrote {} and {}'.format(out_dir/'equity_curve.csv', out_dir/'trades.csv'))

# ----------------------------
# CLI helpers
# ----------------------------

def _infer_date_range(universe: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    first_valid, last_dates = [], []
    for df in universe.values():
        fv = max([x for x in [df['RSI'].first_valid_index(), df['SMA20'].first_valid_index()] if x is not None], default=None)
        if fv is None: continue
        first_valid.append(fv)
        last_dates.append(df.index[-1])
    if not first_valid or not last_dates:
        raise SystemExit('Unable to infer date range from data.')
    end = max(last_dates).normalize()
    raw = max(first_valid).normalize()
    start = raw - pd.Timedelta(days=raw.weekday())
    if start > end:
        start = end - pd.Timedelta(days=end.weekday())
    return pd.Timestamp(start), pd.Timestamp(end)


def _parse_date(s: Optional[str], label: str) -> Optional[pd.Timestamp]:
    if s is None or str(s).strip() == '':
        return None
    try:
        return pd.Timestamp(str(s).strip())
    except Exception as e:
        raise SystemExit("Invalid {} date '{}'. Expected YYYY-MM-DD. Error: {}".format(label, s, e))

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='stock_data_500')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end',   default=None)

    # selection
    ap.add_argument('--rsi-min', type=float, default=68.0)
    ap.add_argument('--max-ext20', type=float, default=0.15)
    ap.add_argument('--trend-sma', type=int, default=100)
    ap.add_argument('--weekly-up-only', action='store_true')

    # portfolio / costs
    ap.add_argument('--top-per-week', type=int, default=6)
    ap.add_argument('--slippage-bps', type=float, default=5.0)
    ap.add_argument('--commission-per-trade', type=float, default=0.0)
    ap.add_argument('--initial-capital', type=float, default=100000.0)

    # stops
    ap.add_argument('--use-stops', action='store_true')
    ap.add_argument('--stop-kind', choices=['pct','atr'], default='atr')
    ap.add_argument('--abs-stop-pct', type=float, default=0.02)
    ap.add_argument('--trail-pct', type=float, default=0.02)
    ap.add_argument('--atr-win', type=int, default=14)
    ap.add_argument('--atr-mult', type=float, default=1.75)
    ap.add_argument('--stop-model', choices=['intraday','next_open'], default='intraday')

    ap.add_argument('--out-dir', default='backtests')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit('Data dir not found: {}'.format(data_dir))

    universe = load_universe(data_dir)
    if not universe:
        raise SystemExit('No usable CSVs loaded')

    start = _parse_date(args.start, '--start')
    end   = _parse_date(args.end, '--end')
    if start is None or end is None:
        start, end = _infer_date_range(universe)
        print('[dates] Using inferred range: start={} end={}'.format(start.date(), end.date()))
    else:
        print('[dates] Using explicit range: start={} end={}'.format(start.date(), end.date()))
    if end <= start:
        raise SystemExit('Invalid range: end ({}) must be after start ({})'.format(end.date(), start.date()))

    backtest(universe=universe,
             start=start, end=end,
             rsi_min=args.rsi_min, max_ext20=args.max_ext20,
             top_per_week=args.top_per_week,
             slippage_bps=args.slippage_bps,
             commission_per_trade=args.commission_per_trade,
             initial_capital=args.initial_capital,
             out_dir=Path(args.out_dir),
             trend_sma=args.trend_sma,
             weekly_up_only=args.weekly_up_only,
             use_stops=args.use_stops,
             stop_kind=args.stop_kind,
             abs_stop_pct=args.abs_stop_pct,
             trail_pct=args.trail_pct,
             atr_win=args.atr_win,
             atr_mult=args.atr_mult,
             stop_model=args.stop_model)

if __name__ == '__main__':
    sys.exit(main() or 0)

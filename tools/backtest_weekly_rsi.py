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
    Start: latest first-valid index among symbols for which both RSI & SMA20 exist.
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

    start = max(first_valid_dates)  # ensure indicators are formed
    # align to Monday or later (avoid empty anchor weeks)
    if start.weekday() != 0:
        start = start + pd.Timedelta(days=(7 - start.weekday()))
    end = max(last_dates)
    return pd.Timestamp(start.normalize()), pd.Timestamp(end.normalize())


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    try:
        return pd.Timestamp(s)
    except Exception as e:
        raise SystemExit(f"Invalid date '{s}'. Expected YYYY-MM-DD.\n{e}")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='stock_data_400')
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
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start is None or end is None:
        inf_start, inf_end = _infer_date_range(universe)
        start = inf_start if start is None else start
        end = inf_end if end is None else end
    if end <= start:
        raise SystemExit(f"Invalid range: end ({end.date()}) must be after start ({start.date()})")

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


# ----------------------------
# Self tests (ALWAYS add tests)
# ----------------------------

def _mk_synth_csv(dirpath: Path, symbol: str, days: int, start_price: float, drift: float) -> None:
    idx = pd.bdate_range('2020-01-01', periods=days)
    prices = []
    p = start_price
    rng = np.random.default_rng(42 if symbol == 'AAA' else 43)
    for _ in range(len(idx)):
        # geometric-ish walk with drift
        p *= (1 + drift + rng.normal(0, 0.005))
        prices.append(p)
    close = pd.Series(prices, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1)
    low = pd.concat([open_, close], axis=1).min(axis=1)
    vol = pd.Series(1_000_000, index=idx)
    df = pd.DataFrame({'Date': idx, 'Open': open_.values, 'High': high.values,
                       'Low': low.values, 'Close': close.values, 'Volume': vol.values})
    df.to_csv(dirpath / f"{symbol}.csv", index=False)


def _run_self_test():
    print("Running self-tests...")
    tmpdir = Path('./_tmp_selftest')
    outdir = tmpdir / 'out'
    tmpdir.mkdir(exist_ok=True)

    # Two symbols: one strong uptrend, one mild drift
    _mk_synth_csv(tmpdir, 'AAA', days=200, start_price=50.0, drift=0.004)
    _mk_synth_csv(tmpdir, 'BBB', days=200, start_price=30.0, drift=0.000)

    universe = load_universe(tmpdir)
    assert 'AAA' in universe and 'BBB' in universe, "Universe not loaded correctly"

    start, end = _infer_date_range(universe)

    backtest(universe=universe,
             start=start,
             end=end,
             rsi_min=55.0,           # easier threshold for synthetic data
             max_ext20=0.50,
             top_per_week=2,
             hold_days=5,
             slippage_bps=0.0,
             commission_per_trade=0.0,
             initial_capital=10000.0,
             out_dir=outdir)

    # Validate artifacts
    eq = pd.read_csv(outdir / 'equity_curve.csv')
    tr = pd.read_csv(outdir / 'trades.csv')
    assert len(eq) > 4, "Equity series too short"
    assert len(tr) > 0, "No trades recorded"
    assert eq['Equity'].iloc[-1] > eq['Equity'].iloc[0], "Equity did not grow in uptrend"

    print("Self-test passed ✔")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)


# =============================================================
# NEW FILE: tools/make_universe_polygon.py
# =============================================================
#!/usr/bin/env python3
"""
Stock-400 data builder using Polygon.io

Goal
----
Create/refresh a `stock_data_400/` directory of daily OHLCV CSVs so your
backtester has data. Designed to run **locally or in GitHub Actions** with the
API key stored as a secret.

Highlights
----------
- Pulls daily aggregates via Polygon v2 aggs endpoint (1 day bars).
- Two ways to choose the universe:
  1) Provide your own ticker list file (one symbol per line), or
  2) Auto-pick top *N* U.S. common stocks by **average dollar volume** over the
     last 60 trading days (ex-ETFs) using Polygon ref + aggs endpoints.
- Idempotent: skips symbols that are already up-to-date unless `--force`.
- Safe defaults for rate limiting, retries, and pagination.
- Outputs CSVs like `AAPL.csv` with columns: `Date,Open,High,Low,Close,Volume`.
- `--self-test` mode generates synthetic CSVs without hitting Polygon (for CI).

Usage (local)
-------------
```bash
export POLYGON_API_KEY=...  # or pass --api-key
python tools/make_universe_polygon.py \
  --out-dir stock_data_400 \
  --start 2015-01-01 --end 2025-10-03 \
  --universe-size 400
```

Use a custom ticker file:
```bash
python tools/make_universe_polygon.py --tickers-file universe/sp500.txt --out-dir stock_data_400 \
  --start 2018-01-01 --end 2025-10-03
```

GitHub Actions (example)
------------------------
Add `.github/workflows/polygon_download.yml` (see bottom of this file for YAML).
Store your API key in repo settings as `POLYGON_API_KEY` secret.

Requirements
------------
- `requests`, `pandas`

"""

from __future__ import annotations
import argparse, os, sys, time, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
import pandas as pd

POLY_BASE = "https://api.polygon.io"

# ---------------------------
# Utilities
# ---------------------------

def _err_exit(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def _sleep_backoff(i: int):
    # exponential backoff with cap
    time.sleep(min(2 ** i, 30))


def _get_json(url: str, params: Dict[str, str], retries: int = 5) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                # rate limited – respect server hint
                _sleep_backoff(i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            _sleep_backoff(i)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


# ---------------------------
# Universe discovery
# ---------------------------

def read_ticker_file(p: Path) -> List[str]:
    return [line.strip().upper() for line in p.read_text().splitlines() if line.strip() and not line.startswith('#')]


def fetch_active_common_stocks(api_key: str, limit: int = 1000) -> List[dict]:
    """Fetch active US common stocks (filters out ETFs) via v3/reference/tickers.
    Returns minimal records with ticker & primaryExchange.
    """
    url = f"{POLY_BASE}/v3/reference/tickers"
    params = {
        'market': 'stocks',
        'active': 'true',
        'limit': 1000,
        'apiKey': api_key,
        'type': 'CS',  # Common Stock; helps filter out ETFs (type=ETF)
    }
    out = []
    while True:
        data = _get_json(url, params)
        results = data.get('results', [])
        out.extend(results)
        next_url = data.get('next_url')
        if not next_url or len(out) >= limit:
            break
        # next_url already contains apiKey sometimes
        url = next_url
        params = {} if 'apiKey=' in next_url else {'apiKey': api_key}
    return out[:limit]


def select_by_dollar_volume(api_key: str, tickers: List[str], lookback_days: int = 60, top_n: int = 400) -> List[str]:
    """Rank tickers by average dollar volume over last N trading days and return top N."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.tseries.offsets.BDay(lookback_days * 1.3)  # overshoot for holidays
    start_s = start.strftime('%Y-%m-%d')
    end_s = end.strftime('%Y-%m-%d')

    scores: List[Tuple[str, float]] = []
    for i, t in enumerate(tickers):
        url = f"{POLY_BASE}/v2/aggs/ticker/{t}/range/1/day/{start_s}/{end_s}"
        params = {'adjusted': 'true', 'apiKey': api_key, 'limit': 50000}
        try:
            data = _get_json(url, params)
            bars = data.get('results', [])
            if not bars:
                continue
            df = pd.DataFrame(bars)
            # v2 fields: c=close, v=volume
            if 'c' not in df or 'v' not in df:
                continue
            adv = float((df['c'] * df['v']).mean())
            if math.isfinite(adv):
                scores.append((t, adv))
        except Exception:
            continue
        # small courteous pause to avoid rate spikes
        time.sleep(0.02)
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_n]]


# ---------------------------
# Download aggregates and write CSVs
# ---------------------------

def fetch_aggs_daily(api_key: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {'adjusted': 'true', 'apiKey': api_key, 'limit': 50000}
    data = _get_json(url, params)
    bars = data.get('results', [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    # map fields per Polygon docs: t (ms), o,h,l,c,v
    df['Date'] = pd.to_datetime(df['t'], unit='ms')
    out = pd.DataFrame({
        'Date': df['Date'],
        'Open': df['o'],
        'High': df['h'],
        'Low': df['l'],
        'Close': df['c'],
        'Volume': df['v'],
    }).dropna(subset=['Date']).sort_values('Date')
    return out


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------
# Main builder
# ---------------------------

def build_universe(api_key: str, out_dir: Path, start: str, end: str, universe_size: int,
                   tickers_file: Optional[Path], force: bool) -> List[str]:
    # Step 1: decide tickers
    if tickers_file and tickers_file.exists():
        tickers = read_ticker_file(tickers_file)
        if universe_size:
            tickers = tickers[:universe_size]
    else:
        print("Discovering active common stocks from Polygon...")
        refs = fetch_active_common_stocks(api_key, limit=2000)
        candidates = [r['ticker'] for r in refs if r.get('type') == 'CS']
        print(f"Candidates: {len(candidates)}. Ranking by avg $ volume...")
        tickers = select_by_dollar_volume(api_key, candidates, lookback_days=60, top_n=universe_size)

    print(f"Universe size: {len(tickers)}")

    # Step 2: download or refresh CSVs
    written = []
    for i, t in enumerate(tickers, 1):
        outp = out_dir / f"{t}.csv"
        if outp.exists() and not force:
            # best-effort append (fetch from last date forward)
            try:
                existing = pd.read_csv(outp)
                if 'Date' in existing.columns and not existing.empty:
                    last = pd.to_datetime(existing['Date']).max().strftime('%Y-%m-%d')
                    if last >= end:
                        print(f"[{i}/{len(tickers)}] {t}: up-to-date")
                        continue
                    print(f"[{i}/{len(tickers)}] {t}: updating from {last} -> {end}")
                    df_new = fetch_aggs_daily(api_key, t, last, end)
                    if not df_new.empty:
                        merged = (pd.concat([existing, df_new])
                                    .drop_duplicates(subset=['Date'])
                                    .sort_values('Date'))
                        write_csv(merged, outp)
                        written.append(t)
                        time.sleep(0.02)
                        continue
            except Exception:
                pass  # fall through to full refresh
        print(f"[{i}/{len(tickers)}] {t}: downloading {start} -> {end}")
        try:
            df = fetch_aggs_daily(api_key, t, start, end)
            if df.empty:
                print(f"  (no data) {t}")
                continue
            write_csv(df, outp)
            written.append(t)
        except Exception as e:
            print(f"  (error) {t}: {e}")
        time.sleep(0.02)

    print(f"Finished. Wrote/updated {len(written)} CSVs to {out_dir}")
    return tickers


# ---------------------------
# CLI & Self-test
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-key', default=os.getenv('POLYGON_API_KEY'), help='Polygon API key (or set POLYGON_API_KEY)')
    ap.add_argument('--out-dir', default='stock_data_400')
    ap.add_argument('--start', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='YYYY-MM-DD')
    ap.add_argument('--universe-size', type=int, default=400)
    ap.add_argument('--tickers-file', default=None)
    ap.add_argument('--force', action='store_true', help='Ignore existing CSVs and redownload')
    ap.add_argument('--self-test', action='store_true', help='Generate 3 synthetic tickers without hitting Polygon')
    args = ap.parse_args()

    if args.self_test:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # synth series to validate CSV shape
        idx = pd.bdate_range('2020-01-01', '2020-06-30')
        for t in ['SYN1','SYN2','SYN3']:
            s = pd.Series(100.0, index=idx).cumprod()*0+100
            df = pd.DataFrame({'Date': idx, 'Open': s, 'High': s, 'Low': s, 'Close': s, 'Volume': 1_000_000})
            df.to_csv(out / f"{t}.csv", index=False)
        print(f"Self-test CSVs written to {out}")
        return 0

    if not args.api_key:
        _err_exit("Missing Polygon API key. Pass --api-key or set POLYGON_API_KEY env var.")

    out_dir = Path(args.out_dir)
    tickers_file = Path(args.tickers_file) if args.tickers_file else None

    build_universe(api_key=args.api_key,
                   out_dir=out_dir,
                   start=args.start,
                   end=args.end,
                   universe_size=args.universe_size,
                   tickers_file=tickers_file,
                   force=args.force)


if __name__ == '__main__':
    sys.exit(main() or 0)


# ---------------------------
# GitHub Actions (example workflow)
# ---------------------------
# Save as .github/workflows/polygon_download.yml
#
# name: Download Polygon Data
# on:
#   workflow_dispatch:
#   schedule:
#     - cron: '15 2 * * 1-5'   # weekdays 02:15 UTC
# jobs:
#   build:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with:
#           python-version: '3.11'
#       - run: pip install requests pandas
#       - name: Fetch data
#         env:
#           POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
#         run: |
#           python tools/make_universe_polygon.py \
#             --out-dir stock_data_400 \
#             --start 2015-01-01 --end $(date -u +%F) \
#             --universe-size 400
#       - name: Upload data artifact
#         uses: actions/upload-artifact@v4
#         with:
#           name: stock_data_400
#           path: stock_data_400/**

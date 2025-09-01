#!/usr/bin/env python3
# trailing_stops.py — compact trailing stop engine (percent or ATR) on daily closes
#
# Inputs:
#   --data-dir stock_data_400
#   --picklist backtests/picklist_highrsi_trend.csv
#   --week 2025-09-05         # Friday of the target week (your WEEK var)
#   --monday 2025-09-08       # next Monday (planned entry date)
#   --topk 6
#   --positions backtests/positions.json
#   --method pct|atr          # default: pct
#   --trail-pct 0.10          # 10% trailing on the running max Close since entry
#   --atr-win 14 --atr-mult 3 # (only if method=atr) trailing = peak_close - 3*ATR14
# Behavior:
#   - Opens/rolls Top-K picks for the given week if not already active.
#   - Updates trailing stop (ratchets up, never down).
#   - If latest close <= stop: marks EXIT (at next open) and deactivates the position.
# Outputs:
#   - backtests/positions.json (persisted state)
#   - backtests/stops.csv      (one row per active position)
#   - backtests/trailing_report.txt (human-readable summary)

import argparse, json, math, pathlib, sys
from datetime import date, datetime
import pandas as pd

def read_csv_prices(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns like: Date, Open, High, Low, Close, Volume
    # Make everything predictable:
    cols = {c.lower(): c for c in df.columns}
    for need in ("date","close","high","low"):
        if need not in cols:
            raise ValueError(f"{csv_path.name}: missing column like '{need}'")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[cols["date"]], errors="coerce").dt.date,
        "open": pd.to_numeric(df.get("Open", df.get("open")), errors="coerce"),
        "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
        "low":  pd.to_numeric(df[cols["low"]],  errors="coerce"),
        "close":pd.to_numeric(df[cols["close"]],errors="coerce"),
    }).dropna(subset=["date","close"])
    out = out.sort_values("date").reset_index(drop=True)
    return out

def compute_atr14(df: pd.DataFrame) -> pd.Series:
    # df: columns date, open, high, low, close
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    return atr

def today_or_last(dates: pd.Series) -> date:
    # just use the last available day in the file
    return dates.max()

def load_positions(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"positions": []}

def save_positions(path: pathlib.Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def upsert_position(state: dict, sym: str, **fields):
    for p in state["positions"]:
        if p["symbol"] == sym and p.get("active", True):
            p.update(fields)
            return
    state["positions"].append({"symbol": sym, "active": True, **fields})

def deactivate(state: dict, sym: str, exit_date: str, exit_price: float, reason: str):
    for p in state["positions"]:
        if p["symbol"] == sym and p.get("active", True):
            p["active"] = False
            p["exit_date"] = exit_date
            p["exit_price"] = round(float(exit_price), 4)
            p["exit_reason"] = reason
            return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--week", required=True)
    ap.add_argument("--monday", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--positions", default="backtests/positions.json")
    ap.add_argument("--method", choices=["pct","atr"], default="pct")
    ap.add_argument("--trail-pct", type=float, default=0.10)
    ap.add_argument("--atr-win", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=3.0)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    pos_path = pathlib.Path(args.positions)
    pick = pd.read_csv(args.picklist)

    # pick week column and top-K
    wkcol = "week_start" if "week_start" in pick.columns else ("week" if "week" in pick.columns else None)
    if not wkcol:
        print("ERROR: picklist missing week column", file=sys.stderr); sys.exit(2)

    pick[wkcol] = pd.to_datetime(pick[wkcol], errors="coerce").dt.date
    week = datetime.fromisoformat(args.week).date()
    picks_week = pick[pick[wkcol] == week].copy()
    if "rank" in picks_week.columns:
        picks_week = picks_week.sort_values(["rank","symbol"], ascending=[True, True])
    elif "score" in picks_week.columns:
        picks_week = picks_week.sort_values(["score","symbol"], ascending=[False,True])
    picks = [s.upper() for s in picks_week["symbol"].astype(str).head(args.topk).tolist()]

    # load/prepare positions state
    state = load_positions(pos_path)
    report_lines = []
    stops_rows = []

    for sym in picks:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            report_lines.append(f"{sym}: SKIP (no CSV at {csv_path})")
            continue

        df = read_csv_prices(csv_path)
        if df.empty:
            report_lines.append(f"{sym}: SKIP (no rows)")
            continue

        # entry is the upcoming Monday (planned); initial stop based on last close
        monday = datetime.fromisoformat(args.monday).date()
        last_close = float(df.iloc[-1]["close"])
        entry_date = monday.isoformat()
        entry_price = last_close  # approximation for planning

        # reconstruct running peak and stops since entry
        # filter from first date >= (week Friday) inclusive, so peak begins at first bar after signal
        df_after = df[df["date"] >= week].copy()
        if df_after.empty:
            # no bars yet after the signal; seed a stop from last close
            peak = last_close
            if args.method == "pct":
                stop = peak * (1 - args.trail_pct)
            else:
                atr = compute_atr14(df).iloc[-1]
                stop = peak - args.atr_mult * float(atr if not math.isnan(atr) else 0.0)
        else:
            if args.method == "pct":
                df_after["peak"] = df_after["close"].cummax()
                df_after["stop"] = df_after["peak"] * (1 - args.trail_pct)
            else:
                atr = compute_atr14(df_after)
                df_after["peak"] = df_after["close"].cummax()
                df_after["stop"] = df_after["peak"] - args.atr_mult * atr
            # latest values
            peak = float(df_after["peak"].iloc[-1]) if "peak" in df_after else last_close
            stop = float(df_after["stop"].iloc[-1]) if "stop" in df_after else last_close*(1-args.trail_pct)

            # if ever breached: first date where close <= stop
            breached = df_after[df_after["close"] <= df_after["stop"]]
            if not breached.empty:
                br = breached.iloc[0]
                # mark as deactivated (exit at next open; we don't have intraday here)
                deactivate(state, sym, exit_date=str(br["date"]), exit_price=float(br["close"]), reason="trailing_stop")
                report_lines.append(f"{sym}: EXIT on {br['date']} close={br['close']:.2f} stop={br['stop']:.2f}")
                # still write one row with last known stop:
                stops_rows.append({"symbol": sym, "entry_date": entry_date, "entry_price": entry_price,
                                   "peak_close": peak, "stop": stop, "last_close": last_close, "active": False})
                continue

        # upsert/refresh active position
        upsert_position(state, sym,
                        entry_date=entry_date,
                        entry_price=round(entry_price,4),
                        method=args.method,
                        trail_pct=args.trail_pct,
                        atr_win=args.atr_win,
                        atr_mult=args.atr_mult,
                        peak_close=round(float(peak),4),
                        current_stop=round(float(stop),4),
                        last_close=round(float(last_close),4),
                        active=True)
        report_lines.append(f"{sym}: ACTIVE peak={peak:.2f} stop={stop:.2f} last={last_close:.2f}")
        stops_rows.append({"symbol": sym, "entry_date": entry_date, "entry_price": entry_price,
                           "peak_close": peak, "stop": stop, "last_close": last_close, "active": True})

    # save artifacts
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    save_positions(pos_path, state)
    pd.DataFrame(stops_rows).sort_values("symbol").to_csv("backtests/stops.csv", index=False)
    pathlib.Path("backtests/trailing_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print("Wrote backtests/positions.json, backtests/stops.csv, backtests/trailing_report.txt")

if __name__ == "__main__":
    main()

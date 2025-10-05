#!/usr/bin/env python3
# trailing_stops.py — compact trailing stop engine (percent or ATR) on daily bars

import argparse, json, math, pathlib, sys
from datetime import date, datetime, timedelta
import pandas as pd

def read_csv_prices(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    need = ("date","close","high","low","open")
    for k in need:
        if k not in cols:
            # try to map by case-insensitive name
            m = [c for c in df.columns if c.lower()==k]
            if not m:
                raise ValueError(f"{csv_path.name}: missing column like '{k}'")
            cols[k] = m[0]
    out = pd.DataFrame({
        "date":  pd.to_datetime(df[cols["date"]], errors="coerce").dt.date,
        "open":  pd.to_numeric(df[cols["open"]],  errors="coerce"),
        "high":  pd.to_numeric(df[cols["high"]],  errors="coerce"),
        "low":   pd.to_numeric(df[cols["low"]],   errors="coerce"),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
    }).dropna(subset=["date","close"])
    out = out.sort_values("date").reset_index(drop=True)
    return out

def compute_atr14(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()

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

def next_monday(d: date) -> date:
    days = (7 - d.weekday()) % 7
    return d + timedelta(days=days or 7)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--picklist", required=True, help="CSV with columns: week_start/symbol/(rank|score)")
    ap.add_argument("--week", default="", help="Friday of the signal week (YYYY-MM-DD). If blank, use latest in picklist.")
    ap.add_argument("--monday", default="", help="Entry Monday (YYYY-MM-DD). If blank, computed from --week.")
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

    # detect week column
    wkcol = "week_start" if "week_start" in pick.columns else ("week" if "week" in pick.columns else None)
    if not wkcol:
        print("ERROR: picklist missing week column", file=sys.stderr); sys.exit(2)
    pick[wkcol] = pd.to_datetime(pick[wkcol], errors="coerce").dt.date

    # auto-select latest week/monday if not provided
    if not args.week.strip():
        latest_week = pick[wkcol].dropna().max()
        if pd.isna(latest_week):
            print("ERROR: no parsable dates in picklist", file=sys.stderr); sys.exit(2)
        args.week = latest_week.isoformat()
        print(f"[trailing_stops] --week not supplied; using latest: {args.week}")
    try:
        week = datetime.fromisoformat(args.week).date()
    except Exception:
        print(f"ERROR: bad --week '{args.week}'", file=sys.stderr); sys.exit(2)

    if not args.monday.strip():
        args.monday = next_monday(week).isoformat()
        print(f"[trailing_stops] --monday not supplied; using next Monday: {args.monday}")
    try:
        monday = datetime.fromisoformat(args.monday).date()
    except Exception:
        print(f"ERROR: bad --monday '{args.monday}'", file=sys.stderr); sys.exit(2)

    # pick week subset and Top-K
    sub = pick[pick[wkcol] == week].copy()
    if "rank" in sub.columns:
        sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce")
        sub = sub.sort_values(["rank", "symbol"], ascending=[True, True])
    elif "score" in sub.columns:
        sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
        sub = sub.sort_values(["score", "symbol"], ascending=[False, True])
    picks = [str(s).upper() for s in sub["symbol"].head(args.topk)]

    # state & outputs
    state = load_positions(pos_path)
    report_lines, stops_rows = [], []

    for sym in picks:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            report_lines.append(f"{sym}: SKIP (no CSV at {csv_path})"); continue
        df = read_csv_prices(csv_path)
        if df.empty:
            report_lines.append(f"{sym}: SKIP (no rows)"); continue

        # Entry price = Monday open if available; else fallback to last close
        entry_price = float(df.loc[df["date"] >= monday, "open"].iloc[0]) if (df["date"] >= monday).any() else float(df.iloc[-1]["close"])
        entry_date = monday.isoformat()

        # Consider bars strictly AFTER Friday signal day (start of trade is next session)
        df_after = df[df["date"] > week].copy()

        if df_after.empty:
            # no bars yet after signal; seed stop from last close
            last_close = float(df.iloc[-1]["close"])
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

            peak = float(df_after["peak"].iloc[-1])
            stop = float(df_after["stop"].iloc[-1])

            # Breach on close
            breached = df_after[df_after["close"] <= df_after["stop"]]
            if not breached.empty:
                br = breached.iloc[0]
                deactivate(state, sym, exit_date=str(br["date"]), exit_price=float(br["close"]), reason="trailing_stop")
                report_lines.append(f"{sym}: EXIT on {br['date']} close={br['close']:.2f} stop={br['stop']:.2f}")
                stops_rows.append({"symbol": sym, "entry_date": entry_date, "entry_price": entry_price,
                                   "peak_close": peak, "stop": stop,
                                   "last_close": float(df_after['close'].iloc[-1]), "active": False})
                continue

        last_close = float(df[df["date"] <= date.today()].iloc[-1]["close"])
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
    save_positions(pos_path, state)

    cols = ["symbol","entry_date","entry_price","peak_close","stop","last_close","active"]
    stops_df = pd.DataFrame(stops_rows, columns=cols)
    stops_df = stops_df.sort_values("symbol") if not stops_df.empty else stops_df
    stops_df.to_csv("backtests/stops.csv", index=False)

    report = "\n".join(report_lines) if report_lines else "[trailing_stops] no symbols processed"
    pathlib.Path("backtests/trailing_report.txt").write_text(report, encoding="utf-8")

    print(report)
    msg = f"{len(stops_df)} rows" if not stops_df.empty else "no rows"
    print(f"Wrote backtests/positions.json, backtests/stops.csv ({msg}), backtests/trailing_report.txt")

if __name__ == "__main__":
    main()

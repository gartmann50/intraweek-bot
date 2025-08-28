#!/usr/bin/env python3
import argparse, pathlib, pandas as pd

CANDIDATE_SCORE_COLS = ["score","f_rsi14","rsi14","rsi_14","RSI","rsi","rsi14_score","f_rsi_14"]

def pick_numeric_column(df, names):
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        col = lower.get(name.lower())
        if not col: 
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return col, s
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--exclude-file", default="universe/exclude.txt")
    args = ap.parse_args()

    pick = pathlib.Path(args.pick)
    df = pd.read_csv(pick)

    # defaults + repo exclude
    defaults = {"SPLK","SGEN","SPY","QQQ","IWM","VOO","VTI","DIA","XLK"}
    exfile = pathlib.Path(args.exclude_file)
    exfile.parent.mkdir(parents=True, exist_ok=True)
    extra = set()
    if exfile.exists():
        extra = {s.strip().upper() for s in exfile.read_text(encoding="utf-8").splitlines()
                 if s.strip() and not s.strip().startswith("#")}
    excl = defaults | extra

    if "symbol" not in df.columns:
        raise SystemExit("picklist missing 'symbol'")

    # drop excluded
    df["__UP"] = df["symbol"].str.upper()
    before = len(df)
    df = df[~df["__UP"].isin(excl)].drop(columns="__UP")
    removed = before - len(df)

    # find week column
    wkcol = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
    if not wkcol:
        raise SystemExit("picklist missing week column")
    df[wkcol] = pd.to_datetime(df[wkcol], errors="coerce")

    # prefer RSI-like score, else rank, else symbol; rebuild ranks
    score_col, _ = pick_numeric_column(df, CANDIDATE_SCORE_COLS)
    if score_col:
        df["__score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(-1e9)
        df = df.sort_values([wkcol,"__score","symbol"], ascending=[True,False,True], kind="stable").drop(columns="__score")
        method = f"score:{score_col}"
    elif "rank" in df.columns:
        df["__rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(1e9)
        df = df.sort_values([wkcol,"__rank","symbol"], ascending=[True,True,True], kind="stable").drop(columns="__rank")
        method = "rank"
    else:
        df = df.sort_values([wkcol,"symbol"], ascending=[True,True], kind="stable")
        method = "symbol-only"

    df["rank"] = df.groupby(wkcol).cumcount() + 1
    df.to_csv(pick, index=False)

    last = df[wkcol].max()
    sub = df[df[wkcol]==last].copy()
    cols = [c for c in ["symbol","rank"] + CANDIDATE_SCORE_COLS if c in sub.columns]
    print(f"Re-ranked by {method}. Columns: {list(df.columns)}")
    try:
        print(sub.head(10)[cols].to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()

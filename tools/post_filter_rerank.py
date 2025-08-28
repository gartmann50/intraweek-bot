#!/usr/bin/env python3
import argparse, pathlib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--exclude-file", default="universe/exclude.txt")
    args = ap.parse_args()

    pick = pathlib.Path(args.pick)
    df = pd.read_csv(pick)

    defaults = {"SPLK","SGEN","SPY","QQQ","IWM","VOO","VTI","DIA","XLK"}
    exfile = pathlib.Path(args.exclude_file)
    exfile.parent.mkdir(parents=True, exist_ok=True)
    file_syms = set()
    if exfile.exists():
        file_syms = {s.strip().upper() for s in exfile.read_text(encoding="utf-8").splitlines()
                     if s.strip() and not s.strip().startswith("#")}
    excl = defaults | file_syms

    if "symbol" not in df.columns:
        raise SystemExit("picklist missing 'symbol'")

    df["__UP"] = df["symbol"].str.upper()
    before = len(df)
    df = df[~df["__UP"].isin(excl)].drop(columns="__UP")
    removed = before - len(df)

    wkcol = "week_start" if "week_start" in df.columns else ("week" if "week" in df.columns else None)
    if not wkcol:
        raise SystemExit("picklist missing week column")

    # Prefer RSI score for order; fallback to rank; rebuild ranks
    if "score" in df.columns:
        df["__score"] = pd.to_numeric(df["score"], errors="coerce").fillna(-1e9)
        df = df.sort_values([wkcol, "__score", "symbol"], ascending=[True, False, True], kind="stable").drop(columns="__score")
        df["rank"] = df.groupby(wkcol).cumcount() + 1
        method = "score"
    elif "rank" in df.columns:
        df["__rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(1e9)
        df = df.sort_values([wkcol, "__rank", "symbol"], ascending=[True, True, True], kind="stable").drop(columns="__rank")
        df["rank"] = df.groupby(wkcol).cumcount() + 1
        method = "rank"
    else:
        df = df.sort_values([wkcol, "symbol"], ascending=[True, True], kind="stable")
        method = "symbol-only"

    df.to_csv(pick, index=False)
    print(f"Excluded {removed} rows; kept {len(df)}. Re-ranked by {method}.")

if __name__ == "__main__":
    main()

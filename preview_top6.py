#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import pandas as pd
import sys
import re

def clean_symbol(s: str) -> str:
    s = str(s).strip().upper()
    # strip suffixes like _5YEAR_DATA
    return re.sub(r"_\d+YEAR.*$", "", s)

def to_date(obj) -> date | None:
    try:
        return pd.to_datetime(obj, errors="coerce").date()
    except Exception:
        return None

def next_monday(today: date | None = None) -> date:
    d = today or datetime.now(timezone.utc).astimezone().date()
    return d + timedelta(days=(7 - d.weekday()) % 7)

def choose_week(week_series: pd.Series, prefer: date | None = None) -> date | None:
    """Pick the week to show: prefer (if present), else next Monday (if present), else last available."""
    weeks = sorted(set([d for d in week_series if isinstance(d, date)]))
    if not weeks:
        return None
    if prefer and prefer in weeks:
        return prefer
    nm = next_monday()
    if nm in weeks:
        return nm
    return weeks[-1]  # most recent

def load_symbol_names(path: Path) -> dict[str, str]:
    """
    Optional: load ticker->company name map from symbol_names.csv (columns: symbol,name).
    Returns {} if file missing or invalid.
    """
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        sym_col = None
        for c in df.columns:
            if c.lower() in ("symbol", "ticker"):
                sym_col = c
                break
        name_col = None
        for c in df.columns:
            if c.lower() in ("name", "company", "company_name"):
                name_col = c
                break
        if not sym_col or not name_col:
            return {}
        df = df[[sym_col, name_col]].dropna()
        return {str(r[sym_col]).strip().upper(): str(r[name_col]).strip() for _, r in df.iterrows()}
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser(description="Preview Top-6 picks from a picklist CSV")
    ap.add_argument("--picklist", default=r"backtests\picklist_highrsi_trend.csv",
                    help="CSV with columns: week_start, symbol (rank/score optional)")
    ap.add_argument("--topk", type=int, default=6, help="How many symbols to show")
    ap.add_argument("--week", default=None,
                    help="Force a specific week_start (YYYY-MM-DD). If absent, choose next Monday if present, else last.")
    ap.add_argument("--names", default="symbol_names.csv",
                    help="Optional CSV mapping (symbol,name) to show full company names")
    args = ap.parse_args()

    p = Path(args.picklist)
    if not p.exists():
        print(f"[WARN] Picklist not found: {p}")
        sys.exit(0)

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] Could not read picklist: {e}")
        sys.exit(0)

    # Basic columns
    if "week_start" not in df.columns:
        print("[WARN] Picklist missing 'week_start' column")
        sys.exit(0)

    # Normalize symbol column name
    sym_col = None
    for c in df.columns:
        if c.lower() in ("symbol", "ticker"):
            sym_col = c
            break
    if not sym_col:
        print("[WARN] Picklist needs a 'symbol' or 'ticker' column")
        sys.exit(0)

    # Parse weeks
    df["week_start_date"] = pd.to_datetime(df["week_start"], errors="coerce").dt.date
    if df["week_start_date"].isna().all():
        print("[WARN] No valid dates in week_start")
        sys.exit(0)

    # Resolve target week
    prefer = to_date(args.week) if args.week else None
    target_week = choose_week(df["week_start_date"].dropna(), prefer)
    if target_week is None:
        print("[WARN] Could not determine a valid week to show.")
        sys.exit(0)

    block = df[df["week_start_date"] == target_week].copy()
    if block.empty:
        print(f"[WARN] No rows for week_start={target_week}")
        sys.exit(0)

    # Sort preference: rank asc, else score desc, else symbol asc
    if "rank" in block.columns:
        block = block.sort_values("rank", ascending=True)
    elif "score" in block.columns:
        block = block.sort_values("score", ascending=False)
    else:
        block = block.sort_values(sym_col)

    # Clean symbols and take Top-K
    block[sym_col] = block[sym_col].map(clean_symbol)
    picks = [str(x) for x in block[sym_col].head(max(1, int(args.topk))).tolist() if str(x).strip()]

    # Optional company names
    names_map = load_symbol_names(Path(args.names))
    def label(sym: str) -> str:
        nm = names_map.get(sym)
        return f"{sym} — {nm}" if nm else sym

    print(f"\n=== Weekly picks — week of {target_week} ===")
    if not picks:
        print("[WARN] Top-K list is empty after filtering.")
        print()
        sys.exit(0)

    for i, s in enumerate(picks, 1):
        print(f"{i:>2}. {label(s)}")
    print(f"\nCSV: {','.join(picks)}\n")

if __name__ == "__main__":
    main()

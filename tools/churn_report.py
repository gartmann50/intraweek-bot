#!/usr/bin/env python3
"""
churn_report.py — quick “are we scanning fresh?” report

What it does
- Reads THIS week’s picklist CSV and LAST week’s picklist CSV
- Computes overlap (“churn”) for Top-6 / Top-10 / Top-20 / Top-40
- Prints entrants/exits for each cutoff
- Also prints file hashes (so you can tell if files are identical week-to-week)

Typical usage (locally):
  python tools/churn_report.py \
    --current backtests/picklist_highrsi_trend.csv \
    --previous backtests_prev/picklist_highrsi_trend.csv

Typical usage (in Actions) after you download last week’s artifact to backtests_prev/:
  python tools/churn_report.py --current backtests/picklist_highrsi_trend.csv --previous backtests_prev/picklist_highrsi_trend.csv

Exit code:
  0 always (it’s a report tool, not a gate).
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def sha12(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b).hexdigest()[:12]


def detect_week_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("week_start", "week", "friday", "date"):
        if c in df.columns:
            return c
    return None


def detect_rank_sort(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Returns (sorted_df, mode_string).
    Priority:
      - rank asc
      - score desc
      - else: keep file order
    """
    if "rank" in df.columns:
        return df.sort_values(["rank", "symbol"], ascending=[True, True]).copy(), "rank_asc"
    if "score" in df.columns:
        return df.sort_values(["score", "symbol"], ascending=[False, True]).copy(), "score_desc"
    return df.copy(), "file_order"


def normalize_symbols(series: pd.Series) -> List[str]:
    return (
        series.dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )


def pick_top(df: pd.DataFrame, n: int) -> List[str]:
    if "symbol" not in df.columns:
        raise SystemExit("ERROR: CSV missing required column: symbol")
    return normalize_symbols(df["symbol"]).copy()[:n]


def overlap_report(curr: List[str], prev: List[str], n: int) -> str:
    a = set(curr[:n])
    b = set(prev[:n])
    inter = a & b
    only_a = [s for s in curr[:n] if s in (a - b)]
    only_b = [s for s in prev[:n] if s in (b - a)]
    pct = (len(inter) / n * 100.0) if n else 0.0
    lines = []
    lines.append(f"Top-{n}: overlap {len(inter)}/{n} = {pct:.1f}%")
    if only_a:
        lines.append(f"  + entered: {', '.join(only_a)}")
    if only_b:
        lines.append(f"  - exited : {', '.join(only_b)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="Path to current week picklist CSV")
    ap.add_argument("--previous", required=True, help="Path to previous week picklist CSV")
    ap.add_argument("--cutoffs", default="6,10,20,40", help="Comma-separated cutoffs (default 6,10,20,40)")
    ap.add_argument("--week", default="", help="Optional: force week label (YYYY-MM-DD). If empty, uses max week in file if present.")
    args = ap.parse_args()

    cur_path = Path(args.current)
    prev_path = Path(args.previous)
    if not cur_path.exists():
        raise SystemExit(f"ERROR: current picklist not found: {cur_path}")
    if not prev_path.exists():
        raise SystemExit(f"ERROR: previous picklist not found: {prev_path}")

    dfc = pd.read_csv(cur_path)
    dfp = pd.read_csv(prev_path)

    # Week selection: if a week column exists, use max date unless --week is provided.
    wkcol_c = detect_week_col(dfc)
    wkcol_p = detect_week_col(dfp)

    def select_week(df: pd.DataFrame, wkcol: Optional[str], forced_week: str) -> pd.DataFrame:
        if not wkcol:
            return df
        s = pd.to_datetime(df[wkcol], errors="coerce").dt.date
        if forced_week:
            wanted = str(pd.to_datetime(forced_week).date())
        else:
            wanted = str(s.dropna().max())
        return df.loc[s.astype(str) == wanted].copy()

    dfc = select_week(dfc, wkcol_c, args.week)
    dfp = select_week(dfp, wkcol_p, args.week)

    dfc, mode_c = detect_rank_sort(dfc)
    dfp, mode_p = detect_rank_sort(dfp)

    # Build ordered lists
    # (Keep as list so "entered/exited" preserves order)
    cur_syms = pick_top(dfc, 10_000)  # big number to get all symbols
    prev_syms = pick_top(dfp, 10_000)

    cutoffs = []
    for x in args.cutoffs.split(","):
        x = x.strip()
        if not x:
            continue
        cutoffs.append(int(x))
    cutoffs = sorted(set(cutoffs))

    # Header
    print("=== IW Bot — Churn / Freshness Report ===")
    print(f"current : {cur_path}  sha={sha12(cur_path)}  rows={len(dfc)}  sort={mode_c}")
    print(f"previous: {prev_path}  sha={sha12(prev_path)}  rows={len(dfp)}  sort={mode_p}")
    if wkcol_c:
        wc = str(pd.to_datetime(dfc[wkcol_c], errors="coerce").dropna().dt.date.max()) if len(dfc) else "n/a"
        print(f"current week column: {wkcol_c}  max={wc}")
    if wkcol_p:
        wp = str(pd.to_datetime(dfp[wkcol_p], errors="coerce").dropna().dt.date.max()) if len(dfp) else "n/a"
        print(f"previous week column: {wkcol_p}  max={wp}")

    print()
    print(f"Total unique symbols: current={len(set(cur_syms))} previous={len(set(prev_syms))}")
    print()

    for n in cutoffs:
        # Ensure we don't divide by a cutoff larger than list length (still report)
        if len(cur_syms) < n or len(prev_syms) < n:
            print(f"Top-{n}: cannot compute cleanly (current has {len(cur_syms)}, previous has {len(prev_syms)})")
            # still compute on available length if you want; for now keep it strict & clear
            print()
            continue
        print(overlap_report(cur_syms, prev_syms, n))
        print()

    # Quick “freshness smell test”
    # If hashes are identical, you are almost certainly reusing the same file.
    if sha12(cur_path) == sha12(prev_path):
        print("⚠️  WARNING: current and previous picklist SHA are identical.")
        print("    That strongly suggests you are reusing the same file (not scanning fresh).")


if __name__ == "__main__":
    main()

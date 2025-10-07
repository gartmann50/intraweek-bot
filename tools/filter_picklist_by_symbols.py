#!/usr/bin/env python3
import sys, csv, re
from typing import Set, List

"""
Robustly filter a picklist CSV to a symbol allow-list.

- Detects the symbol column by header name (case-insensitive). Tries:
  ['symbol', 'ticker', 'sym', 'ticker_symbol'] — falls back to column 0.
- Normalizes tickers for matching:
    * uppercases
    * trims spaces
    * compares s, s.replace('-', '.'), s.replace('.', '-')
- Preserves header and original row order/ranking.

Usage:
  python tools/filter_picklist_by_symbols.py \
    backtests/universe_topvol.txt \
    backtests/picklist_highrsi_trend.csv \
    backtests/picklist_highrsi_trend.csv   # (in-place overwrite)
"""

CANDIDATE_HEADERS = ["symbol", "ticker", "sym", "ticker_symbol"]

def load_allow(path: str) -> Set[str]:
    allow = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                allow.add(s)
                # allow both dot/hyphen share-class notations
                allow.add(s.replace("-", "."))
                allow.add(s.replace(".", "-"))
    return allow

def find_symbol_col(header: List[str]) -> int:
    # case-insensitive match against common names
    lower = [h.strip().lower() for h in header]
    for name in CANDIDATE_HEADERS:
        if name in lower:
            return lower.index(name)
    # fallback: first column
    return 0

def norm(sym: str) -> List[str]:
    s = (sym or "").strip().upper()
    if not s:
        return []
    return [s, s.replace("-", "."), s.replace(".", "-")]

def main():
    if len(sys.argv) != 4:
        print("args: SYMBOLS_TXT IN_CSV OUT_CSV", file=sys.stderr)
        sys.exit(2)

    symfile, incsv, outcsv = sys.argv[1:]
    allow = load_allow(symfile)

    with open(incsv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("[filter] empty input CSV; nothing to do.", file=sys.stderr)
        sys.exit(0)

    header, body = rows[0], rows[1:]
    sym_col = find_symbol_col(header)

    kept = []
    for r in body:
        if not r or sym_col >= len(r):
            continue
        syms = norm(r[sym_col])
        if any(s in allow for s in syms):
            kept.append(r)

    with open(outcsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    print(f"[filter] kept {len(kept)} rows from {len(body)} based on universe ({len(allow)} symbols).")
    # small diagnostic to help if it ever drops to 0 again
    print(f"[filter] detected symbol column: {header[sym_col] if sym_col < len(header) else 'col0'}")

if __name__ == "__main__":
    main()

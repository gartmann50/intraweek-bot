#!/usr/bin/env python3
import sys, csv

"""
Usage:
  python tools/filter_picklist_by_symbols.py \
    backtests/universe_topvol.txt \
    backtests/picklist_highrsi_trend.csv \
    backtests/picklist_highrsi_trend.csv   # (in-place overwrite)
"""

def read_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip()}

def main():
    if len(sys.argv) != 4:
        print("args: SYMBOLS_TXT IN_CSV OUT_CSV", file=sys.stderr)
        sys.exit(2)
    symfile, incsv, outcsv = sys.argv[1:]
    allow = read_symbols(symfile)
    with open(incsv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        sys.exit(0)
    header, body = rows[0], rows[1:]
    # assume first column is symbol; adjust if your file differs
    out = [header] + [r for r in body if r and r[0].strip().upper() in allow]
    with open(outcsv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"[filter] kept {len(out)-1} rows from {len(body)} based on universe ({len(allow)} symbols).")

if __name__ == "__main__":
    main()

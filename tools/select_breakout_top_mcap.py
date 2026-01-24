#!/usr/bin/env python3
import os, argparse, requests
import pandas as pd
from pathlib import Path

POLY = "https://api.polygon.io"

def get_mcap(ticker: str, key: str):
    # Polygon ticker details endpoint usually includes market_cap
    url = f"{POLY}/v3/reference/tickers/{ticker}"
    r = requests.get(url, params={"apiKey": key}, timeout=25)
    if not r.ok:
        return None
    j = r.json() or {}
    res = j.get("results") or {}
    mc = res.get("market_cap")
    try:
        return float(mc) if mc is not None else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="CSV containing at least a symbol column")
    ap.add_argument("--symbol-col", default="symbol")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", default="backtests/breakout_symbols.txt")
    args = ap.parse_args()

    key = os.environ.get("POLYGON_API_KEY","").strip()
    if not key:
        raise SystemExit("Missing POLYGON_API_KEY")

    df = pd.read_csv(args.in_csv)
    if args.symbol_col not in df.columns:
        raise SystemExit(f"Missing column '{args.symbol_col}' in {args.in_csv}")

    syms = df[args.symbol_col].dropna().astype(str).str.upper().unique().tolist()
    if not syms:
        Path(args.out).write_text("", encoding="utf-8")
        print("No symbols in input; wrote empty breakout list.")
        return 0

    rows=[]
    for s in syms:
        mc = get_mcap(s, key)
        rows.append({"symbol": s, "market_cap": mc if mc is not None else -1})

    outdf = pd.DataFrame(rows).sort_values(["market_cap","symbol"], ascending=[False, True])
    top = outdf.head(args.top)["symbol"].tolist()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(top)+("\n" if top else ""), encoding="utf-8")
    print(f"Wrote {args.out} ({len(top)} symbols). Top by market cap:")
    print(top)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

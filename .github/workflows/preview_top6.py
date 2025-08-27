#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path
from datetime import date, datetime, timedelta

def clean_sym(s: str) -> str:
    s = (s or "").strip().upper()
    return re.sub(r"_\d+YEAR.*$", "", s)

def parse_date(s: str):
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def next_monday(d: date | None = None) -> date:
    d = d or date.today()
    return d + timedelta(days=(7 - d.weekday()) % 7)

def read_symbol_names(path: Path) -> dict[str, str]:
    m = {}
    if not path or not path.exists(): 
        return m
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = clean_sym(row.get("symbol",""))
            name = (row.get("name") or "").strip()
            if sym and name:
                m[sym] = name
    return m

def main():
    ap = argparse.ArgumentParser(description="Preview weekly Top-K from picklist")
    ap.add_argument("--picklist", required=True, help="CSV with week_start,symbol,(rank|score)")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", default=None, help="YYYY-MM-DD for week_start; default=next Monday if present else latest")
    ap.add_argument("--names", default=None, help="Optional symbol_names.csv with columns: symbol,name")
    args = ap.parse_args()

    pkl = Path(args.picklist)
    if not pkl.exists():
        raise SystemExit(f"Picklist not found: {pkl}")

    rows = []
    with pkl.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not {"week_start","symbol"}.issubset({c.lower() for c in r.fieldnames}):
            raise SystemExit("Picklist must have columns: week_start, symbol")
        for row in r:
            ws = parse_date(row.get("week_start",""))
            sym = clean_sym(row.get("symbol",""))
            rank_s = (row.get("rank") or "").strip()
            score_s = (row.get("score") or "").strip()
            # coerce numbers if present
            rank = None
            try: rank = int(float(rank_s)) if rank_s else None
            except: pass
            score = None
            try: score = float(score_s) if score_s else None
            except: pass
            if ws and sym:
                rows.append({"week_start":ws, "symbol":sym, "rank":rank, "score":score})

    if not rows:
        raise SystemExit("Picklist is empty or unreadable rows.")

    # choose the week:
    if args.week:
        want = parse_date(args.week)
        use = want if any(r["week_start"]==want for r in rows) else max(r["week_start"] for r in rows)
    else:
        nm = next_monday()
        use = nm if any(r["week_start"]==nm for r in rows) else max(r["week_start"] for r in rows)

    wk = [r for r in rows if r["week_start"] == use]

    # sort: rank asc if present, else score desc if present, else by symbol
    if any(r["rank"] is not None for r in wk):
        wk.sort(key=lambda x: (x["rank"] if x["rank"] is not None else 1_000_000))
    elif any(r["score"] is not None for r in wk):
        wk.sort(key=lambda x: (-(x["score"] if x["score"] is not None else -1e9), x["symbol"]))
    else:
        wk.sort(key=lambda x: x["symbol"])

    wk = wk[: max(1, args.topk)]

    # names mapping
    names_map = read_symbol_names(Path(args.names)) if args.names else {}

    print(f"Week start: {use.isoformat()}")
    if not wk:
        print("No picks for that week.")
        return

    # Pretty list
    for i, r in enumerate(wk, start=1):
        sym = r["symbol"]
        name = names_map.get(sym, "")
        label = f"{sym} — {name}" if name else sym
        print(f"{i:>2}. {label}")

    # CSV line useful for other scripts
    print("\nCSV:", ",".join([r["symbol"] for r in wk]))

if __name__ == "__main__":
    main()

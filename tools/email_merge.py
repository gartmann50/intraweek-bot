#!/usr/bin/env python3
"""
Combine weekly picks (text or CSV) + 70D-highs (digest or CSV) into email_body.txt.

Looks for (in this order):
- Picks text: backtests/top6_preview.txt, backtests/top6_preview.md, backtests/weekly-picks-*.txt, backtests/weekly_picks_*.txt
- Picks CSV fallback: backtests/picklist_highrsi_trend.csv or backtests/weekly-picks-*.csv
- 70D highs digest: backtests/hi70_digest.txt
- 70D highs CSV fallback: backtests/hi70_thisweek.csv
"""

import os
import glob
import csv

OUT_PATH = "email_body.txt"
parts = []

# -------- Picks section --------
picks_txt = None
for pat in (
    "backtests/top6_preview.txt",
    "backtests/top6_preview.md",
    "backtests/weekly-picks-*.txt",
    "backtests/weekly_picks_*.txt",
):
    matches = glob.glob(pat)
    if matches:
        picks_txt = matches[0]
        break

if picks_txt and os.path.getsize(picks_txt) > 0:
    with open(picks_txt, "r", encoding="utf-8", errors="ignore") as f:
        parts.append("WEEKLY PICKS\n=============\n" + f.read().strip())
else:
    # Try to build a tiny text section from a CSV
    picks_csv = None
    for pat in ("backtests/picklist_highrsi_trend.csv", "backtests/weekly-picks-*.csv"):
        matches = glob.glob(pat)
        if matches:
            picks_csv = matches[0]
            break

    if picks_csv and os.path.getsize(picks_csv) > 0:
        rows = []
        try:
            with open(picks_csv, "r", newline="", encoding="utf-8", errors="ignore") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= 10:
                        break
                    sym = (row.get("symbol") or row.get("ticker") or "").strip()
                    score = row.get("score") or row.get("rank") or ""
                    rows.append(f"{sym:<6}  score={score}")
        except Exception:
            rows = []
        if rows:
            parts.append("WEEKLY PICKS (top)\n===================\n" + "\n".join(rows))
        else:
            parts.append("WEEKLY PICKS\n=============\n(no picks section found)")

# -------- 70D highs section --------
hi70_txt = "backtests/hi70_digest.txt"
if os.path.exists(hi70_txt) and os.path.getsize(hi70_txt) > 0:
    with open(hi70_txt, "r", encoding="utf-8", errors="ignore") as f:
        parts.append("\n" + f.read().strip())
else:
    hi70_csv = "backtests/hi70_thisweek.csv"
    if os.path.exists(hi70_csv) and os.path.getsize(hi70_csv) > 0:
        lines = []
        try:
            with open(hi70_csv, "r", newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 10:
                        break
                    sym = (row.get("symbol") or "").strip()
                    gap = row.get("gap_high_pct") or row.get("gap_close_pct") or ""
                    cap = row.get("market_cap") or ""
                    lines.append(f"{sym:<6}  cap={cap}  gap≈{gap}%")
        except Exception:
            lines = []
        if lines:
            parts.append("70D HIGHS — top\n================\n" + "\n".join(lines))
        else:
            parts.append("70D HIGHS\n=========\n(none)")

# -------- Write out --------
body = "\n\n".join([p for p in parts if p and p.strip()])
if not body.strip():
    body = "No picks or 70D-highs sections were found."

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(body + "\n")

print(f"Wrote {OUT_PATH} ({len(body.encode('utf-8'))} bytes)")

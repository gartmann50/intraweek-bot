#!/usr/bin/env python3
"""
Send 'Weekly Picks' email.

- Reads a picklist CSV with columns like: week_start, symbol, (rank|score optional)
- Chooses the latest available week (or a --week you pass)
- Sends an email with Top-K tickers (and full company names when available)
- SMTP settings are read from config YAML, with fallbacks to environment:
    SMTP_USER, SMTP_PASS, SMTP_TO, SMTP_FROM
- Recipients are Bcc’d (privacy); "To" shows the sender only.
"""

from __future__ import annotations

import argparse
import os
import sys
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, List

import pandas as pd
import yaml
from datetime import datetime, date


# ------------------------------
# Helpers
# ------------------------------

def load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_date_any(s: str) -> Optional[date]:
    """Parse YYYY-MM-DD (or anything pandas can coerce) -> date."""
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        # If it’s a series we take the first valid
        if hasattr(ts, "iloc"):
            ts = ts.iloc[0]
        return ts.date()
    except Exception:
        return None


def load_picklist(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        print(f"Picklist not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    # Ensure 'week_start' column
    wk_col = None
    for c in df.columns:
        if c.lower().strip() in ("week_start", "week", "weekstart", "start"):
            wk_col = c
            break
    if not wk_col:
        raise ValueError("Picklist must contain a 'week_start' column (or similar).")

    # Parse dates robustly
    df["week_start"] = pd.to_datetime(df[wk_col], errors="coerce").dt.date

    # Symbol column
    sym_col = None
    for c in df.columns:
        if c.lower().strip() in ("symbol", "ticker", "sym"):
            sym_col = c
            break
    if not sym_col:
        # fallback: first column that looks string-ish
        sym_col = df.columns[0]

    df = df.rename(columns={sym_col: "symbol"})

    # Ranking column (optional)
    rank_col = None
    for c in df.columns:
        if c.lower().strip() in ("rank", "score", "position"):
            rank_col = c
            break

    # Sort by week then by rank/score (if present)
    if rank_col:
        df = df.sort_values(["week_start", rank_col], ascending=[True, True])
    else:
        df = df.sort_values(["week_start"]).reset_index(drop=True)

    return df[["week_start", "symbol"] + ([rank_col] if rank_col else [])]


def choose_week(df: pd.DataFrame, explicit: Optional[str]) -> date:
    if explicit:
        d = parse_date_any(explicit)
        if not d:
            raise ValueError(f"Could not parse --week {explicit}")
        return d
    # choose latest available
    latest = df["week_start"].dropna().max()
    if pd.isna(latest):
        raise ValueError("Could not parse any valid dates from week_start in picklist.")
    return latest


def load_name_map() -> dict:
    """
    Optional: map tickers to full company names if a file exists in repo:
      - symbol_names.csv  (columns: symbol,name)
      - symbol_names.xlsx (columns: symbol,name)
    If not found, return {} and we’ll print tickers only.
    """
    for fname in ("symbol_names.csv", "symbol_names.xlsx"):
        p = Path(fname)
        if p.exists():
            try:
                if p.suffix == ".csv":
                    df = pd.read_csv(p)
                else:
                    df = pd.read_excel(p)
                m = {}
                if "symbol" in df.columns and "name" in df.columns:
                    for _, r in df.iterrows():
                        sym = str(r["symbol"]).strip().upper()
                        nm = str(r["name"]).strip()
                        if sym and nm:
                            m[sym] = nm
                if m:
                    return m
            except Exception:
                pass
    return {}


def build_email_text(week: date, rows: List[tuple], name_map: dict, topk: int) -> str:
    lines = []
    lines.append(f"Weekly Picks — week start {week:%Y-%m-%d}")
    lines.append("")
    lines.append(f"Top-{topk}")
    for i, (sym,) in enumerate(rows, start=1):
        sym_u = str(sym).upper()
        nm = name_map.get(sym_u, "")
        if nm:
            lines.append(f"  {i:>2}  {sym_u} — {nm}")
        else:
            lines.append(f"  {i:>2}  {sym_u}")
    lines.append("")
    lines.append("Good trading, very good.")
    lines.append("CEO of Kanute")
    return "\n".join(lines)


def send_email(
    smtp_user: str,
    smtp_pass: str,
    smtp_to_list: List[str],
    smtp_from: str,
    subject: str,
    body: str,
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = smtp_from  # keep recipients hidden
    msg["Reply-To"] = smtp_user
    msg.set_content(body)

    # BCC for privacy
    if smtp_to_list:
        msg["Bcc"] = ", ".join(smtp_to_list)

    # Gmail over SSL (465). Fallback to STARTTLS (587).
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=25) as s:
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
            return
    except Exception as e:
        print(f"SSL send failed ({e}); trying STARTTLS…")

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=25) as s:
        s.ehlo()
        s.starttls(context=context)
        s.ehlo()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Send Weekly Picks email.")
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", help="YYYY-MM-DD (optional). If omitted, latest week in picklist is used.")
    ap.add_argument("--subject", help="Override subject")
    args = ap.parse_args()

    # SMTP config (file + env fallbacks)
    cfg = load_yaml(args.config)
    smtp_user = (cfg.get("smtp_user") or os.getenv("SMTP_USER", "")).strip()
    smtp_pass = (cfg.get("smtp_pass") or os.getenv("SMTP_PASS", "")).strip()
    smtp_to   = (cfg.get("smtp_to")   or os.getenv("SMTP_TO",   "")).strip()
    smtp_from = (cfg.get("smtp_from") or os.getenv("SMTP_FROM", "") or smtp_user).strip()

    if not (smtp_user and smtp_pass and smtp_to):
        print("Missing SMTP settings in config_notify.yaml (and env fallbacks).")
        sys.exit(1)

    recips = [x.strip() for x in smtp_to.split(",") if x.strip()]

    # Load picklist, choose target week, slice top-k
    df = load_picklist(args.picklist)
    target_week = choose_week(df, args.week)

    slice_df = df[df["week_start"] == target_week].copy()
    if slice_df.empty:
        print(f"No rows for week_start == {target_week} in {args.picklist}")
        sys.exit(1)

    # keep ordering in file (already sorted); just take head(topk)
    top = slice_df.head(args.topk)[["symbol"]].values.tolist()

    # optional names
    name_map = load_name_map()

    # Compose
    subject = args.subject or f"Weekly Picks — {target_week:%Y-%m-%d} (Top-{args.topk})"
    body = build_email_text(target_week, top, name_map, args.topk)

    # Send
    try:
        send_email(smtp_user, smtp_pass, recips, smtp_from, subject, body)
        print("Email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

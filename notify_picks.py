#!/usr/bin/env python3
"""
Send weekly picks email from a picklist CSV.

Robust date parsing for 'week_start' so CSV quirks (BOMs, spaces, quoted
strings, etc.) don't break the run.

Usage (env SMTP_* already set by workflow):
  python notify_picks.py \
    --picklist backtests/picklist_highrsi_trend.csv \
    --topk 6 \
    [--week 2025-08-25]
"""

from __future__ import annotations
import argparse
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ---------- helpers

def read_yaml_safe(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def robust_parse_week_start(raw: pd.Series) -> pd.Series:
    """
    Turn whatever is in 'week_start' into datelike values.
    Strategy:
      1) pandas to_datetime (errors='coerce')
      2) if all NaT, extract YYYY-MM-DD substring then to_datetime again
    Returns a tz-naive date (datetime64[ns]) series.
    """
    s = raw.copy()
    # strip whitespace, tolerate BOM
    s = s.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    dt = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
    if dt.notna().any():
        return dt.tz_convert(None).tz_localize(None)

    # 2) try extracting canonical date pattern
    extracted = s.str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    dt2 = pd.to_datetime(extracted, errors="coerce", utc=True)
    return dt2.tz_convert(None).tz_localize(None)

def find_week_column(df: pd.DataFrame) -> str:
    for cand in ("week_start", "week-start", "week", "start_week"):
        if cand in df.columns:
            return cand
    raise ValueError(f"Could not find a 'week_start' column. Columns: {list(df.columns)}")

def pick_target_week(df: pd.DataFrame, week: Optional[str]) -> pd.Timestamp:
    """
    Choose which week to send:
      - if --week provided: use its Monday (parse safely)
      - else: use the max week present in the CSV
    """
    if week:
        w = pd.to_datetime(str(week), errors="coerce")
        if pd.isna(w):
            raise ValueError(f"Could not parse --week '{week}'")
        return w.normalize()
    # fall back to most recent week in CSV
    return df["week_start"].max()

def format_list(items: list[str]) -> str:
    return "\n".join(f"  • {sym}" for sym in items)

def build_email_html(week: str, symbols: list[str]) -> str:
    # simple, clean HTML (hide recipients by Bcc)
    items = "".join(f"<li>{sym}</li>" for sym in symbols)
    return f"""
    <html>
      <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; color:#111;">
        <h2 style="margin-bottom:0.2rem;">Weekly Picks</h2>
        <div style="margin:0 0 1rem; color:#666;">Week start: <b>{week}</b></div>
        <ol>
          {items}
        </ol>
        <p style="margin-top:1.2rem;">Good trading — very good!<br/>CEO of Kanute</p>
      </body>
    </html>
    """.strip()

def send_email(user: str, password: str, to_list: list[str], subject: str, html_body: str, from_addr: Optional[str] = None):
    if not from_addr:
        from_addr = user
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    # no visible "To" (we Bcc everyone)
    msg["To"] = from_addr

    msg.attach(MIMEText(html_body, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(user, password)
        server.sendmail(from_addr, to_list, msg.as_string())

# ---------- main

def load_picklist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Picklist not found: {path}")
    # handle UTF-8 with BOM and common CSV quirks
    df = pd.read_csv(path, encoding="utf-8-sig")
    wk_col = find_week_column(df)
    df["week_start"] = robust_parse_week_start(df[wk_col])
    if df["week_start"].notna().sum() == 0:
        raise ValueError("Could not parse any valid dates from week_start in picklist.")
    # keep only needed columns
    cols = [c for c in df.columns if c in ("week_start", "symbol", "rank", "score")]
    return df[cols].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", default=os.environ.get("WEEK", ""))  # optional override
    args = ap.parse_args()

    # load SMTP from config (optional), then ENV (takes precedence if set in workflow)
    cfg = read_yaml_safe(Path(args.config))
    smtp_user = os.environ.get("SMTP_USER") or cfg.get("smtp_user", "")
    smtp_pass = os.environ.get("SMTP_PASS") or cfg.get("smtp_pass", "")
    smtp_to   = os.environ.get("SMTP_TO")   or cfg.get("smtp_to", "")
    smtp_from = os.environ.get("SMTP_FROM") or cfg.get("smtp_from", smtp_user)

    if not (smtp_user and smtp_pass and smtp_to):
        raise SystemExit("Missing SMTP settings in config/ENV (need SMTP_USER, SMTP_PASS, SMTP_TO).")

    df = load_picklist(Path(args.picklist))
    # Pick the target week
    target = pick_target_week(df, args.week)
    this = df[df["week_start"] == target]
    if this.empty:
        # Help user diagnose
        recent_weeks = sorted(set(df["week_start"].dropna().dt.strftime("%Y-%m-%d")))[-5:]
        raise SystemExit(
            f"No rows for week_start = {target.date()} in {args.picklist}.\n"
            f"Recent available weeks: {recent_weeks}"
        )

    # rank by rank then score if they exist, else by symbol (stable)
    sort_keys = [c for c in ("rank", "score") if c in this.columns]
    if sort_keys:
        this = this.sort_values(sort_keys)
    else:
        this = this.sort_values("symbol")

    symbols = list(this["symbol"].astype(str).head(max(1, args.topk)))
    subject = f"Weekly Picks — {target.date()}"

    html = build_email_html(str(target.date()), symbols)

    # Send (Bcc all TO recipients)
    to_list = [addr.strip() for addr in smtp_to.replace(";", ",").split(",") if addr.strip()]
    send_email(smtp_user, smtp_pass, to_list, subject, html, smtp_from)

    print(f"Sent weekly picks for {target.date()}: {', '.join(symbols)}")

if __name__ == "__main__":
    main()

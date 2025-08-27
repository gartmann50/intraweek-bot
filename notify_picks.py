#!/usr/bin/env python3
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

# ----------------- utils

def read_yaml_safe(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def normalize_columns(cols) -> list[str]:
    out = []
    for c in cols:
        c = str(c).replace("\ufeff", "")  # strip BOM if it leaked into header
        c = c.strip().lower().replace(" ", "_").replace("-", "_")
        out.append(c)
    return out

def find_week_column(df: pd.DataFrame) -> str:
    # check normalized names
    for cand in ("week_start", "weekstart", "week", "start_week"):
        if cand in df.columns:
            return cand
    raise ValueError(f"Could not find a 'week_start' column. Columns: {list(df.columns)}")

def robust_parse_week_start(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    # first pass
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        return dt.tz_localize(None)
    # second pass: extract YYYY-MM-DD
    extracted = s.str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    dt2 = pd.to_datetime(extracted, errors="coerce", utc=True)
    return dt2.tz_localize(None)

def load_picklist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Picklist not found: {path}")

    # Auto-detect delimiter + handle BOM safely
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = normalize_columns(df.columns)

    wk = find_week_column(df)
    df["week_start"] = robust_parse_week_start(df[wk])

    if df["week_start"].notna().sum() == 0:
        # diagnostics to help us see what's in the file
        print("DEBUG: head(5) of raw file:")
        try:
            print(df.head(5).to_string(index=False))
        except Exception:
            pass
        print("DEBUG: sample of raw week_start column values:")
        try:
            print(df[wk].astype(str).head(10).to_list())
        except Exception:
            pass
        raise ValueError("Could not parse any valid dates from week_start in picklist.")
    # keep relevant cols if present
    keep = ["week_start"]
    for extra in ("symbol", "rank", "score"):
        if extra in df.columns:
            keep.append(extra)
    return df[keep].copy()

def pick_target_week(df: pd.DataFrame, week_override: Optional[str]) -> pd.Timestamp:
    if week_override:
        w = pd.to_datetime(str(week_override), errors="coerce")
        if pd.isna(w):
            raise ValueError(f"Could not parse --week '{week_override}'")
        return w.normalize()
    return df["week_start"].max()

def build_email_html(week: str, symbols: list[str]) -> str:
    items = "".join(f"<li>{sym}</li>" for sym in symbols)
    return f"""
    <html>
      <body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;color:#111;">
        <h2 style="margin:0 0 .2rem 0;">Weekly Picks</h2>
        <div style="margin:0 0 1rem;color:#666;">Week start: <b>{week}</b></div>
        <ol>{items}</ol>
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
    msg["To"] = from_addr  # we Bcc recipients

    msg.attach(MIMEText(html_body, "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(user, password)
        server.sendmail(from_addr, to_list, msg.as_string())

# ----------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", default=os.environ.get("WEEK", ""))
    args = ap.parse_args()

    # SMTP settings from env (workflow) first, then config file as fallback
    cfg = read_yaml_safe(Path(args.config))
    smtp_user = os.environ.get("SMTP_USER") or cfg.get("smtp_user", "")
    smtp_pass = os.environ.get("SMTP_PASS") or cfg.get("smtp_pass", "")
    smtp_to   = os.environ.get("SMTP_TO")   or cfg.get("smtp_to", "")
    smtp_from = os.environ.get("SMTP_FROM") or cfg.get("smtp_from", smtp_user)

    if not (smtp_user and smtp_pass and smtp_to):
        raise SystemExit("Missing SMTP settings (need SMTP_USER, SMTP_PASS, SMTP_TO).")

    df = load_picklist(Path(args.picklist))
    target = pick_target_week(df, args.week)
    this = df[df["week_start"] == target]

    if this.empty:
        recent = sorted(set(df["week_start"].dropna().dt.strftime("%Y-%m-%d")))[-5:]
        raise SystemExit(
            f"No rows for week_start = {target.date()} in {args.picklist}.\n"
            f"Recent weeks in file: {recent}"
        )

    if "rank" in this.columns or "score" in this.columns:
        keys = [k for k in ("rank", "score") if k in this.columns]
        this = this.sort_values(keys)
    else:
        if "symbol" in this.columns:
            this = this.sort_values("symbol")

    symbols = list(this["symbol"].astype(str).head(max(1, args.topk)))
    html = build_email_html(str(target.date()), symbols)

    to_list = [x.strip() for x in smtp_to.replace(";", ",").split(",") if x.strip()]
    subject = f"Weekly Picks — {target.date()}"
    send_email(smtp_user, smtp_pass, to_list, subject, html, smtp_from)
    print(f"Sent weekly picks for {target.date()}: {', '.join(symbols)}")

if __name__ == "__main__":
    main()

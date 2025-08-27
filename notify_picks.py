#!/usr/bin/env python3
# notify_picks.py
# Send weekly picks email from a picklist CSV, robust date parsing.

import os
import sys
import argparse
import smtplib
from pathlib import Path
from email.mime.text import MIMEText

import pandas as pd
import yaml

# ---------- helpers ----------

def load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def find_week_col(df: pd.DataFrame) -> str:
    # normalize headers
    df.columns = df.columns.str.strip().str.lower()
    candidates = ["week_start", "weekstart", "week", "weekdate", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No week_start-like column found. Columns: {list(df.columns)}")

def parse_week_col(df: pd.DataFrame, col: str) -> pd.Series:
    # be VERY permissive: coerce, handle tz, strip dates
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    # if nothing parsed, try again without utc (handles naive strings better)
    if s.isna().all():
        s = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce")
    # drop tz and keep date only
    s = s.dt.tz_localize(None)
    return s

def pick_target_week(weeks: pd.Series) -> pd.Timestamp:
    """
    Choose the target week:
      - If env UNTIL is set (YYYY-MM-DD), use that (<= max available if possible)
      - Else use max available date
    """
    weeks = weeks.dropna()
    if weeks.empty:
        raise ValueError("Could not parse any valid dates from week_start in picklist.")

    env_until = os.getenv("UNTIL", "").strip()
    if env_until:
        try:
            u = pd.to_datetime(env_until).normalize()
            # choose the latest week <= UNTIL if present, else fallback to latest
            candidates = weeks.dt.normalize()
            mask = candidates <= u
            if mask.any():
                return candidates[mask].max().normalize()
        except Exception:
            pass  # fall through to latest

    return weeks.dt.normalize().max()

def load_symbol_names() -> dict:
    """
    Optional friendly names. Looks for (in this priority):
      symbol_names.xlsx (sheet1 with 'symbol','name')
      symbol_names.csv  (columns 'symbol','name')
      backtests/symbol_names.csv
    Returns { 'AAPL': 'Apple Inc.', ... }
    """
    paths = [
        Path("symbol_names.xlsx"),
        Path("symbol_names.csv"),
        Path("backtests/symbol_names.csv"),
    ]
    for p in paths:
        if p.exists():
            try:
                if p.suffix.lower() == ".xlsx":
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
                df.columns = df.columns.str.strip().str.lower()
                if "symbol" in df.columns and "name" in df.columns:
                    return {str(r["symbol"]).strip().upper(): str(r["name"]).strip()
                            for _, r in df.iterrows()}
            except Exception:
                continue
    return {}

def build_email_body(week_date_str: str, symbols: list[str], names_map: dict, poems: list[str] | None) -> str:
    header = f"Weekly picks — week start {week_date_str}\n\n"
    lines = []
    for i, sym in enumerate(symbols, 1):
        nm = names_map.get(sym.upper(), "")
        if nm:
            lines.append(f"{i:>2}. {sym} — {nm}")
        else:
            lines.append(f"{i:>2}. {sym}")
    body = header + "\n".join(lines) + "\n"

    if poems:
        body += "\n" + ("—"*40) + "\n"
        body += "\n".join(poems) + "\n"

    body += "\nGood trading,\nVery good.\n\nCEO of Kanute"
    return body

def send_email(smtp_host: str, smtp_port: int, user: str, pwd: str,
               sender: str, to_list: list[str], subject: str, text: str):
    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    # hide recipients by BCCing only
    msg["To"] = "Undisclosed recipients"

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(sender, to_list, msg.as_string())

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Send weekly picks email from picklist.")
    ap.add_argument("--picklist", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--subject", default="Weekly picks")
    ap.add_argument("--week-start", default="auto", help='YYYY-MM-DD or "auto" (default)')
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    smtp_user = cfg.get("smtp", {}).get("user", "")
    smtp_pass = cfg.get("smtp", {}).get("pass", "")
    smtp_host = cfg.get("smtp", {}).get("host", "smtp.gmail.com")
    smtp_port = int(cfg.get("smtp", {}).get("port", 587))
    smtp_to   = cfg.get("smtp", {}).get("to", [])
    smtp_from = cfg.get("smtp", {}).get("from", smtp_user or "noreply@example.com")

    if not (smtp_user and smtp_pass and smtp_to):
        print("Missing SMTP settings in config_notify.yaml", file=sys.stderr)
        sys.exit(1)

    pl_path = Path(args.picklist)
    if not pl_path.exists():
        print(f"Picklist not found: {pl_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pl_path)
    # normalize & robustly parse
    col = find_week_col(df)
    weeks = parse_week_col(df, col)
    df["__week__"] = weeks.dt.normalize()

    # decide which week to use
    if args.week_start.lower() == "auto":
        target_week = pick_target_week(weeks)
    else:
        target_week = pd.to_datetime(args.week_start).normalize()

    # filter for target week and sort by rank/score if present
    sub = df[df["__week__"] == target_week].copy()
    if sub.empty:
        print(f"No rows for week_start={target_week.date()} in {pl_path}", file=sys.stderr)
        sys.exit(1)

    sub.columns = sub.columns.str.strip().str.lower()
    if "rank" in sub.columns:
        sub = sub.sort_values("rank", ascending=True)
    elif "score" in sub.columns:
        sub = sub.sort_values("score", ascending=False)

    if "symbol" not in sub.columns:
        # try uppercase column name fallback
        for c in df.columns:
            if c.lower() == "symbol":
                sub["symbol"] = df[c]
                break
    if "symbol" not in sub.columns:
        raise ValueError("No 'symbol' column in picklist.")

    symbols = [str(s).strip().upper() for s in sub["symbol"].head(args.topk).tolist()]

    # optional friendly names
    names_map = load_symbol_names()

    # include poems if present in cfg
    poems = cfg.get("poems", None)
    week_str = target_week.date().isoformat()
    body = build_email_body(week_str, symbols, names_map, poems)
    subject = args.subject

    # log preview to Actions log
    print("---- Email preview ----")
    print(f"Subject: {subject}")
    print(body)
    print("-----------------------")

    # send
    if os.getenv("DRY_RUN", "0") == "1":
        print("DRY_RUN=1: skipping SMTP send.")
        return

    recipients = smtp_to if isinstance(smtp_to, list) else [smtp_to]
    send_email(smtp_host, smtp_port, smtp_user, smtp_pass, smtp_from, recipients, subject, body)
    print("Email sent.")

if __name__ == "__main__":
    main()

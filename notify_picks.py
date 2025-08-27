#!/usr/bin/env python3
import argparse, smtplib, ssl, sys
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
import pandas as pd
import yaml

def load_cfg(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def pick_week(df: pd.DataFrame, week_str: str | None):
    # Expect columns: week_start, symbol and optionally rank/score
    if "week_start" not in df.columns or "symbol" not in df.columns:
        raise SystemExit("picklist must have columns: week_start, symbol (rank optional)")
    df["week_start"] = pd.to_datetime(df["week_start"])
    if week_str:
        wk = pd.Timestamp(week_str)
    else:
        wk = df["week_start"].max()
    week_df = df[df["week_start"] == wk].copy()
    if "rank" in week_df.columns:
        week_df = week_df.sort_values(["rank", "symbol"])
    return wk.date(), week_df

def format_body(week_date, rows, topk: int):
    rows = rows.head(topk).reset_index(drop=True)
    lines = [f"Weekly picks — week start {week_date}"]
    lines.append("")
    for i, r in rows.iterrows():
        rank = r["rank"] if "rank" in r else (i + 1)
        sym = r["symbol"]
        lines.append(f"{rank:>2}. {sym}")
    lines.append("")
    lines.append("Good trading,\nVery good.\n\nCEO of Kanute")
    return "\n".join(lines)

def send_mail(cfg, subject: str, body: str):
    smtp_host = cfg.get("smtp_host", "smtp.office365.com")
    smtp_port = int(cfg.get("smtp_port", 587))
    user = cfg["smtp_user"]
    pwd  = cfg["smtp_pass"]
    to_list = [e.strip() for e in str(cfg.get("smtp_to","")).split(",") if e.strip()]
    from_addr = cfg.get("smtp_from", user)

    # Hide recipients: put them in BCC, send To: undisclosed
    msg = MIMEText(body, _subtype="plain", _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = "Undisclosed-Recipients:;"
    msg["Date"] = formatdate(localtime=True)

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        s.starttls(context=context)
        s.login(user, pwd)
        s.sendmail(from_addr, to_list, msg.as_string())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", required=True, help="CSV with week_start,symbol,(rank)")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", default=None, help="YYYY-MM-DD (default: latest in CSV)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    if not {"smtp_user","smtp_pass","smtp_to"} <= set(cfg.keys()):
        raise SystemExit("Missing SMTP settings in config_notify.yaml")

    df = pd.read_csv(args.picklist)
    week_date, week_df = pick_week(df, args.week)
    if week_df.empty:
        raise SystemExit(f"No rows for week_start {week_date} in {args.picklist}")

    subject = "Weekly picks"
    body = format_body(week_date, week_df, args.topk)

    send_mail(cfg, subject, body)
    print("Email sent.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

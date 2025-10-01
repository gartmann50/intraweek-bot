#!/usr/bin/env python3
"""
Send a single email that contains:
  - your usual Top-K picks preview
  - optional 70-day highs digest if present

It reuses config_notify.yaml already built in the workflow.
"""

import os, sys, smtplib, ssl, argparse, yaml
from email.mime.text import MIMEText

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def read_if_exists(p):
    return open(p, "r", encoding="utf-8").read() if os.path.exists(p) else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--week", required=True)
    ap.add_argument("--topk_path", required=True)      # backtests/top6_preview.txt (already produced)
    ap.add_argument("--hi70_digest", default="backtests/hi70_digest.txt")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    smtp_user = cfg.get("smtp_user") or cfg.get("smtp",{}).get("user")
    smtp_pass = cfg.get("smtp_pass") or cfg.get("smtp",{}).get("pass")
    smtp_from = cfg.get("smtp_from") or cfg.get("from")
    smtp_to   = cfg.get("smtp_to") or cfg.get("to")
    host = cfg.get("host","smtp.gmail.com"); port = int(cfg.get("port",587)); use_tls = bool(cfg.get("use_tls", True))

    if isinstance(smtp_to, str):
        rcpts = [e.strip() for e in smtp_to.replace(";",",").split(",") if e.strip()]
    else:
        rcpts = smtp_to or []

    topk = read_if_exists(args.topk_path)
    hi70 = read_if_exists(args.hi70_digest)

    body = []
    body.append(f"Weekly Picks — week {args.week}\n")
    body.append(topk.strip() or "(no Top-K preview)")
    if hi70:
        body.append("\n\n— — —\n70-Day Highs (≥$1B mc)\n")
        body.append(hi70.strip())
    text = "\n".join(body) + "\n"

    msg = MIMEText(text, _subtype="plain", _charset="utf-8")
    msg["Subject"] = f"IW Bot — Picks + 70D Highs (week {args.week})"
    msg["From"] = smtp_from
    msg["To"] = ", ".join(rcpts)

    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as s:
        if use_tls: s.starttls(context=ctx)
        if smtp_user and smtp_pass:
            s.login(smtp_user, smtp_pass)
        s.sendmail(smtp_from, rcpts, msg.as_string())

    print(f"Email sent to {len(rcpts)} recipient(s).")

if __name__ == "__main__":
    main()

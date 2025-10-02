#!/usr/bin/env python3
"""
Send a single weekly email that includes:
- Weekly picks text (if found)
- 70D highs digest (if backtests/hi70_digest.txt exists)

SMTP settings are read from environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_FROM, SMTP_PASS, SMTP_TO
Optional:
  SUBJECT        (overrides default)
  FRIDAY         (YYYY-MM-DD) for the subject; if missing, auto last NY Friday
"""

from __future__ import annotations
import os, sys, glob
from datetime import datetime, timedelta, date
from email.message import EmailMessage
import smtplib, ssl

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore


# ---------- helpers ----------
NY = ZoneInfo("America/New_York")

def last_friday_ny() -> date:
    now_ny = datetime.now(NY).date()
    while now_ny.weekday() != 4:  # 4 = Friday
        now_ny -= timedelta(days=1)
    return now_ny

def need_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        sys.exit(f"FATAL: {name} is empty")
    return v

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def build_body() -> str:
    lines: list[str] = []

    # ---- Weekly picks ----
    lines.append("=== Weekly Picks ===")
    picks_file = None
    for pat in (
        "backtests/top6_preview.txt",
        "backtests/weekly-picks-*.txt",
        "backtests/weekly_picks_*.txt",
        "backtests/weekly_picks.txt",
    ):
        m = glob.glob(pat)
        if m:
            picks_file = sorted(m)[-1]
            break
    if picks_file and os.path.isfile(picks_file):
        lines.append(read_text(picks_file) or "(picks file is empty)")
    else:
        lines.append("(no weekly picks file found)")

    lines.append("")  # blank line

    # ---- 70D highs ----
    lines.append("=== 70D Highs (Top by market cap) ===")
    hi70_digest = "backtests/hi70_digest.txt"
    if os.path.isfile(hi70_digest):
        lines.append(read_text(hi70_digest) or "(hi70 digest empty)")
    else:
        lines.append("(no hi70 digest found)")

    return ("\n".join(lines)).rstrip() + "\n"

def main() -> None:
    # SMTP
    host = need_env("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    sender = need_env("SMTP_FROM")
    to = need_env("SMTP_TO")
    password = os.getenv("SMTP_PASS", "")

    # Subject
    friday = os.getenv("FRIDAY", "")
    if friday:
        subj_friday = friday
    else:
        subj_friday = last_friday_ny().isoformat()
    subject = os.getenv("SUBJECT", f"IW Bot — picks + 70D highs for week {subj_friday}")

    # Body
    body = build_body()

    # Compose message using EmailMessage (avoids header/body pitfalls)
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    # Send
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        try:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        except smtplib.SMTPException:
            pass

    if password:
        server.login(sender, password)

    server.send_message(msg)
    server.quit()
    print(f"Sent email to {to}; body length={len(body)} bytes.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

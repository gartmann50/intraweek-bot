#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send the weekly summary email (weekly picks + 70D highs).

It builds the body from:
  - Weekly picks (first match in this order):
      backtests/top6_preview.txt
      backtests/weekly-picks.txt
      backtests/weekly-picks-*.txt
      backtests/weekly_picks_*.txt
      **/weekly-picks-*.txt
      **/weekly_picks_*.txt
      **/top6_preview.txt

  - 70D highs digest (first match):
      backtests/hi70_digest.txt
      hi70_digest.txt
      **/hi70_digest.txt

Config sources (first existing):
  1) config_notify.yaml / config_notify.yml
  2) Environment variables:
       SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO,
       SMTP_TLS (true/false, default true), SUBJECT_PREFIX

Important for Gmail:
  - The *envelope sender* must be the authenticated account (SMTP_USER).
  - The header "From" defaults to SMTP_USER too. If you need a different
    visible From, add it as a verified “Send mail as” alias in Gmail;
    only then change SMTP_FROM to that address.

This script also writes the plain-text body to ./email_body.txt (for artifacts).
"""

from __future__ import annotations

import glob
import os
import ssl
import sys
import smtplib
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr

# Optional YAML reading
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -------------------------
# Utilities
# -------------------------

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def find_picks_file() -> Optional[str]:
    exacts = [
        "backtests/top6_preview.txt",
        "backtests/weekly-picks.txt",
    ]
    patterns = [
        "backtests/weekly-picks-*.txt",
        "backtests/weekly_picks_*.txt",
        "**/weekly-picks-*.txt",
        "**/weekly_picks_*.txt",
        "**/top6_preview.txt",
    ]
    candidates: List[str] = exacts[:]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat, recursive=True)))
    return find_first(candidates)


def find_hi70_digest() -> Optional[str]:
    exacts = [
        "backtests/hi70_digest.txt",
        "hi70_digest.txt",
    ]
    patterns = ["**/hi70_digest.txt"]
    candidates: List[str] = exacts[:]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat, recursive=True)))
    return find_first(candidates)


def parse_bool(s: str) -> bool:
    if s is None:
        return False
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def split_recipients(s: str) -> List[str]:
    if not s:
        return []
    out: List[str] = []
    for part in s.replace(";", ",").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def last_friday_utc_date() -> str:
    """Return last Friday (UTC date) like YYYY-MM-DD, used in subject."""
    d = datetime.now(timezone.utc).date()
    while d.weekday() != 4:  # Friday=4
        d -= timedelta(days=1)
    return d.isoformat()


# -------------------------
# Config
# -------------------------

def load_config() -> Dict[str, str]:
    cfg: Dict[str, str] = {}

    yaml_paths = ["config_notify.yaml", "config_notify.yml"]
    ypath = next((p for p in yaml_paths if os.path.isfile(p)), None)
    if ypath and yaml is not None:
        try:
            with open(ypath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        cfg[str(k).strip()] = "true" if (v is True) else "false" if (v is False) else str(v)
        except Exception:
            pass  # fall back to env below

    env_map = {
        "SMTP_HOST": "smtp_host",
        "SMTP_PORT": "smtp_port",
        "SMTP_USER": "smtp_user",
        "SMTP_PASS": "smtp_pass",
        "SMTP_FROM": "smtp_from",
        "SMTP_TO":   "smtp_to",
        "SMTP_TLS":  "smtp_tls",
        "SUBJECT_PREFIX": "subject_prefix",
    }
    for env_key, cfg_key in env_map.items():
        if not cfg.get(cfg_key):
            val = os.getenv(env_key, "")
            if val:
                cfg[cfg_key] = val

    cfg.setdefault("smtp_tls", "true")
    cfg.setdefault("subject_prefix", "IW Bot —")
    return cfg


# -------------------------
# Build email body
# -------------------------

def build_body() -> str:
    lines: List[str] = []

    lines.append("=== Weekly Picks ===")
    picks = find_picks_file()
    if picks:
        txt = read_text(picks).strip() or "(picks file is empty)"
        lines.append(txt)
        print(f"[email] Using picks: {picks}")
    else:
        lines.append("(no weekly picks file found)")
        print("[email] No weekly picks file found")

    lines.append("")  # blank

    lines.append("=== 70D Highs (Top by market cap) ===")
    digest = find_hi70_digest()
    if digest:
        txt = read_text(digest).strip() or "(hi70 digest empty)"
        lines.append(txt)
        print(f"[email] Using hi70 digest: {digest}")
    else:
        lines.append("(no hi70 digest found)")
        print("[email] No hi70 digest found")

    body = ("\n".join(lines)).rstrip() + "\n"

    try:
        with open("email_body.txt", "w", encoding="utf-8") as f:
            f.write(body)
        print("[email] Wrote email_body.txt")
    except Exception as e:
        print(f"[email] WARN: failed to write email_body.txt: {e}")

    return body


# -------------------------
# Send
# -------------------------

def _addr_only(s: str) -> str:
    """Return just the email address portion."""
    name, addr = parseaddr(s or "")
    return (addr or "").strip()

def send_email(cfg: Dict[str, str], subject: str, body: str) -> None:
    host = cfg.get("smtp_host", "").strip()
    port_s = cfg.get("smtp_port", "").strip() or "587"
    user = (cfg.get("smtp_user", "") or "").strip()
    pwd  = (cfg.get("smtp_pass", "") or "").strip()
    from_hdr = (cfg.get("smtp_from", "") or user).strip()
    to_csv = (cfg.get("smtp_to", "") or "").strip()
    use_tls = parse_bool(cfg.get("smtp_tls", "true"))

    if not host:
        sys.exit("FATAL: SMTP host is empty")
    try:
        port = int(port_s)
    except Exception:
        port = 587

    # Clean addresses
    login_addr = _addr_only(user)
    header_from_addr = _addr_only(from_hdr) or login_addr
    to_addrs_clean = [_addr_only(x) for x in split_recipients(to_csv) if _addr_only(x)]

    if not login_addr:
        sys.exit("FATAL: SMTP_USER (login) is empty or invalid")
    if not to_addrs_clean:
        sys.exit("FATAL: SMTP_TO / smtp_to is empty")

    # Build message (From header can be the same as login or a verified alias)
    msg = EmailMessage()
    msg["From"] = formataddr(("", header_from_addr))  # or set a display name instead of ""
    msg["To"] = ", ".join(to_addrs_clean)
    msg["Subject"] = subject
    msg.set_content(body, subtype="plain", charset="utf-8")

    # Envelope sender MUST be the authenticated account for Gmail
    envelope_sender = login_addr

    if port == 465:
        import ssl
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as s:
            s.login(login_addr, pwd)
            s.sendmail(envelope_sender, to_addrs_clean, msg.as_string())
    else:
        import ssl
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            if use_tls:
                s.starttls(context=ssl.create_default_context())
                s.ehlo()
            s.login(login_addr, pwd)
            s.sendmail(envelope_sender, to_addrs_clean, msg.as_string())

    print(f"[email] Sent email to {to_addrs_clean}; body length={len(body.encode('utf-8'))} bytes.")

# -------------------------
# Main
# -------------------------

def main() -> None:
    cfg = load_config()
    body = build_body()

    # Subject like: "IW Bot — Weekly picks + 70D highs (week 2025-09-26)"
    week_anchor = os.getenv("FRIDAY", "") or last_friday_utc_date()
    prefix = cfg.get("subject_prefix", "IW Bot —").strip()
    if prefix:
        subject = f"{prefix} Weekly picks + 70D highs (week {week_anchor})"
    else:
        subject = f"Weekly picks + 70D highs (week {week_anchor})"

    send_email(cfg, subject, body)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

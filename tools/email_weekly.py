#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send the weekly summary email (weekly picks + 70D highs).

Inputs it tries to find (recursively):
  - Weekly picks text (first match):
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

Email configuration is read from (first existing):
  - config_notify.yaml
  - config_notify.yml
or from environment variables as fallback:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO (comma-separated),
  SMTP_TLS (true/false, default true), SUBJECT_PREFIX (optional)

It also writes the built body to ./email_body.txt so the workflow can upload it.
"""

from __future__ import annotations

import os
import sys
import glob
import smtplib
import ssl
from typing import Dict, List, Optional

# Optional YAML reading
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -------------------------
# Small utility helpers
# -------------------------

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def find_first(paths: List[str]) -> Optional[str]:
    """Return the first existing file path from the list."""
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def find_picks_file() -> Optional[str]:
    # Try exact names first
    exacts = [
        "backtests/top6_preview.txt",
        "backtests/weekly-picks.txt",
    ]

    # Then allow wildcard patterns, including recursive fallbacks
    patterns = [
        "backtests/weekly-picks-*.txt",
        "backtests/weekly_picks_*.txt",
        "**/weekly-picks-*.txt",
        "**/weekly_picks_*.txt",
        "**/top6_preview.txt",
    ]

    candidates: List[str] = exacts[:]
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        candidates.extend(matches)

    return find_first(candidates)


def find_hi70_digest() -> Optional[str]:
    exacts = [
        "backtests/hi70_digest.txt",
        "hi70_digest.txt",
    ]
    patterns = [
        "**/hi70_digest.txt",
    ]

    candidates: List[str] = exacts[:]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat, recursive=True)))

    return find_first(candidates)


# -------------------------
# Email config
# -------------------------

def load_config() -> Dict[str, str]:
    """Load config from YAML file if present, else from environment variables."""
    cfg: Dict[str, str] = {}

    yaml_paths = ["config_notify.yaml", "config_notify.yml"]
    yaml_path = next((p for p in yaml_paths if os.path.isfile(p)), None)

    if yaml_path and yaml is not None:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    data = {}
                for k, v in data.items():
                    if isinstance(v, bool):
                        cfg[k] = "true" if v else "false"
                    else:
                        cfg[k] = str(v)
        except Exception:
            # Fall back to env below
            pass

    # Env fallbacks (do not override existing YAML keys)
    env_map = {
        "SMTP_HOST": "smtp_host",
        "SMTP_PORT": "smtp_port",
        "SMTP_USER": "smtp_user",
        "SMTP_PASS": "smtp_pass",
        "SMTP_FROM": "smtp_from",
        "SMTP_TO": "smtp_to",
        "SMTP_TLS": "smtp_tls",
        "SUBJECT_PREFIX": "subject_prefix",
    }
    for env_key, cfg_key in env_map.items():
        if cfg_key not in cfg or not cfg[cfg_key]:
            val = os.getenv(env_key, "")
            if val:
                cfg[cfg_key] = val

    # Defaults
    cfg.setdefault("smtp_tls", "true")
    cfg.setdefault("subject_prefix", "[Weekly]")

    return cfg


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


# -------------------------
# Build email body
# -------------------------

def build_body() -> str:
    lines: List[str] = []

    # Weekly picks section
    lines.append("=== Weekly Picks ===")
    picks = find_picks_file()
    if picks:
        txt = read_text(picks).strip()
        if not txt:
            txt = "(picks file is empty)"
        lines.append(txt)
        print(f"[email] Using picks file: {picks}")
    else:
        lines.append("(no weekly picks file found)")
        print("[email] No weekly picks file found")

    lines.append("")  # blank line

    # 70D highs section
    lines.append("=== 70D Highs (Top by market cap) ===")
    digest = find_hi70_digest()
    if digest:
        txt = read_text(digest).strip()
        if not txt:
            txt = "(hi70 digest empty)"
        lines.append(txt)
        print(f"[email] Using hi70 digest: {digest}")
    else:
        lines.append("(no hi70 digest found)")
        print("[email] No hi70 digest found")

    body = ("\n".join(lines)).rstrip() + "\n"
    # Write for artifact/debugging
    try:
        with open("email_body.txt", "w", encoding="utf-8") as f:
            f.write(body)
        print("[email] Wrote email_body.txt")
    except Exception as e:
        print(f"[email] WARN: failed to write email_body.txt: {e}")

    return body


# -------------------------
# Send email
# -------------------------

def send_email(cfg: Dict[str, str], subject: str, body: str) -> None:
    host = cfg.get("smtp_host", "").strip()
    port_s = cfg.get("smtp_port", "").strip() or "587"
    user = cfg.get("smtp_user", "").strip()
    pwd  = cfg.get("smtp_pass", "").strip()
    from_addr = cfg.get("smtp_from", "").strip()
    to_csv = cfg.get("smtp_to", "").strip()
    use_tls = parse_bool(cfg.get("smtp_tls", "true"))

    if not host:
        sys.exit("FATAL: SMTP host is empty")
    try:
        port = int(port_s)
    except Exception:
        port = 587

    to_addrs = split_recipients(to_csv)
    if not to_addrs:
        sys.exit("FATAL: SMTP_TO / smtp_to is empty")

    # Simple plain-text message
    msg = (
        f"From: {from_addr}\r\n"
        f"To: {', '.join(to_addrs)}\r\n"
        f"Subject: {subject}\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        f"{body}"
    ).encode("utf-8")

    # Connect and send
    if port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as s:
            if user:
                s.login(user, pwd)
            s.sendmail(from_addr, to_addrs, msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            if use_tls:
                context = ssl.create_default_context()
                s.starttls(context=context)
                s.ehlo()
            if user:
                s.login(user, pwd)
            s.sendmail(from_addr, to_addrs, msg)

    print(f"[email] Sent email to {to_addrs}; body length={len(body.encode('utf-8'))} bytes.")


# -------------------------
# Main
# -------------------------

def main() -> None:
    cfg = load_config()
    body = build_body()

    prefix = cfg.get("subject_prefix", "[Weekly]").strip()
    if prefix:
        subject = f"{prefix} Weekly picks + 70D highs"
    else:
        subject = "Weekly picks + 70D highs"

    send_email(cfg, subject, body)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

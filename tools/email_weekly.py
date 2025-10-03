#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send the weekly summary email (weekly picks + 70D highs).

Inputs it looks for:
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

Configuration:
  1) If present, it reads config_notify.yaml / .yml
  2) Then environment variables OVERRIDE the YAML:
       SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
       SMTP_FROM, SMTP_TO, SMTP_TLS, SUBJECT_PREFIX
  Defaults:
       SMTP_TLS=true, SUBJECT_PREFIX="[Weekly]"

Behavior:
  - If SMTP_USER is empty, it falls back to SMTP_FROM (sane for Gmail).
  - Writes the built body to ./email_body.txt for debugging / artifacts.
"""

from __future__ import annotations

import os
import sys
import glob
import ssl
import smtplib
from typing import Dict, List, Optional
from email.utils import parseaddr, formataddr

# Optional YAML (script works without it)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -------------------------
# Small file helpers
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

    candidates = exacts[:]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat, recursive=True)))
    return find_first(candidates)


def find_hi70_digest() -> Optional[str]:
    exacts = [
        "backtests/hi70_digest.txt",
        "hi70_digest.txt",
    ]
    patterns = [
        "**/hi70_digest.txt",
    ]

    candidates = exacts[:]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat, recursive=True)))
    return find_first(candidates)


# -------------------------
# Config
# -------------------------

def _load_yaml_config() -> Dict[str, str]:
    """Load config from YAML if present; return {} if not."""
    for p in ("config_notify.yaml", "config_notify.yml"):
        if os.path.isfile(p) and yaml is not None:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    # Normalize to str values
                    out: Dict[str, str] = {}
                    for k, v in data.items():
                        if isinstance(v, bool):
                            out[k] = "true" if v else "false"
                        elif v is None:
                            out[k] = ""
                        else:
                            out[k] = str(v)
                    return out
            except Exception:
                pass
    return {}


def _env_map() -> Dict[str, str]:
    return {
        "SMTP_HOST": "smtp_host",
        "SMTP_PORT": "smtp_port",
        "SMTP_USER": "smtp_user",
        "SMTP_PASS": "smtp_pass",
        "SMTP_FROM": "smtp_from",
        "SMTP_TO":   "smtp_to",
        "SMTP_TLS":  "smtp_tls",
        "SUBJECT_PREFIX": "subject_prefix",
    }


def load_config() -> Dict[str, str]:
    """
    Merge YAML + ENV with ENV taking priority.
    Adds sensible defaults and SMTP_USER fallback to SMTP_FROM.
    """
    cfg = _load_yaml_config()

    # ENV overrides YAML (CI-friendly)
    for env_key, cfg_key in _env_map().items():
        val = os.getenv(env_key, "")
        if val is not None and len(val.strip()) > 0:
            cfg[cfg_key] = val.strip()

    # Defaults
    cfg.setdefault("smtp_tls", "true")
    cfg.setdefault("subject_prefix", "[Weekly]")

    # Final fallback: if login missing, use FROM
    if not cfg.get("smtp_user", "").strip():
        if cfg.get("smtp_from", "").strip():
            cfg["smtp_user"] = cfg["smtp_from"].strip()

    return cfg


def parse_bool(s: str) -> bool:
    return str(s or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def split_recipients(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").replace(";", ",").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


# -------------------------
# Build email body
# -------------------------

def build_body() -> str:
    lines: List[str] = []

    # Weekly picks
    lines.append("=== Weekly Picks ===")
    picks = find_picks_file()
    if picks:
        txt = (read_text(picks) or "").strip() or "(picks file is empty)"
        lines.append(txt)
        print(f"[email] Using picks file: {picks}")
    else:
        lines.append("(no weekly picks file found)")
        print("[email] No weekly picks file found")

    lines.append("")  # blank line

    # 70D highs
    lines.append("=== 70D Highs (Top by market cap) ===")
    digest = find_hi70_digest()
    if digest:
        txt = (read_text(digest) or "").strip() or "(hi70 digest empty)"
        lines.append(txt)
        print(f"[email] Using hi70 digest: {digest}")
    else:
        lines.append("(no hi70 digest found)")
        print("[email] No hi70 digest found")

    # ---- Options snapshot (picks) ----
    lines.append("")
    lines.append("=== Options snapshot (±10% OTM, ~45d) — Picks ===")
    snap = None
    for pat in ("backtests/options_snapshot.txt", "**/options_snapshot.txt"):
        ms = sorted(glob.glob(pat, recursive=True))
        if ms:
            snap = ms[0]; break
    if snap:
        txt = read_text(snap).strip()
        lines.append(txt if txt else "(options snapshot empty)")
    else:
        lines.append("(no options snapshot)")

    # ---- Options snapshot (70D breakouts) ----
    lines.append("")
    lines.append("=== Options snapshot (±10% OTM, ~45d) — 70D breakouts ===")
    snap70 = None
    for pat in ("backtests/options_snapshot_hi70.txt", "**/options_snapshot_hi70.txt"):
        ms = sorted(glob.glob(pat, recursive=True))
        if ms:
            snap70 = ms[0]; break
    if snap70:
        txt = read_text(snap70).strip()
        lines.append(txt if txt else "(options snapshot empty)")
    else:
        lines.append("(no options snapshot for 70D)")
  
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

def _addr_only(s: str) -> str:
    """Return just the email address part."""
    name, addr = parseaddr(s or "")
    return addr or (s or "")


def _nice_sender(s: str) -> str:
    """Return a clean sender header (preserve name if present)."""
    name, addr = parseaddr(s or "")
    if not addr:
        return s or ""
    return formataddr((name, addr))


def send_email(cfg: Dict[str, str], subject: str, body: str) -> None:
    host = cfg.get("smtp_host", "").strip()
    port_s = cfg.get("smtp_port", "").strip() or "587"
    from_header = cfg.get("smtp_from", "").strip()
    to_csv = cfg.get("smtp_to", "").strip()

    # login/pass with fallbacks
    user = (cfg.get("smtp_user", "").strip() or from_header)
    pwd  = cfg.get("smtp_pass", "").strip()
    use_tls = parse_bool(cfg.get("smtp_tls", "true"))

    if not host:
        sys.exit("FATAL: SMTP_HOST is empty")
    try:
        port = int(port_s)
    except Exception:
        port = 587

    # Resolve addresses
    from_addr = _addr_only(from_header)
    if not from_addr:
        sys.exit("FATAL: SMTP_FROM is empty or invalid")

    to_addrs = split_recipients(to_csv)
    if not to_addrs:
        sys.exit("FATAL: SMTP_TO is empty")

    if not user:
        sys.exit("FATAL: SMTP_USER (login) is empty (after fallback to FROM)")

    # Compose minimal plain text email
    msg = (
        f"From: {_nice_sender(from_header)}\r\n"
        f"To: {', '.join(to_addrs)}\r\n"
        f"Subject: {subject}\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        f"{body}"
    ).encode("utf-8")

    # Brief diagnostics (no secrets)
    print(f"[email] SMTP host={host} port={port} tls={use_tls} "
          f"user={'present' if user else 'missing'} "
          f"from={from_addr} to={len(to_addrs)} recipient(s)")

    try:
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
    except smtplib.SMTPAuthenticationError as e:
        print("[email] SMTPAuthenticationError:", e)
        print(
            "[email] HINT (Gmail): use an App Password with 2-Step Verification: "
            "https://myaccount.google.com/apppasswords"
        )
        sys.exit(1)
    except Exception as e:
        print("[email] ERROR sending mail:", e)
        sys.exit(1)

    print(f"[email] Sent email to {to_addrs}; body bytes={len(body.encode('utf-8'))}.")


# -------------------------
# Main
# -------------------------

def main() -> None:
    cfg = load_config()
    body = build_body()

    prefix = (cfg.get("subject_prefix", "[Weekly]") or "").strip()
    subject = f"{prefix} Weekly picks + 70D highs" if prefix else "Weekly picks + 70D highs"

    send_email(cfg, subject, body)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

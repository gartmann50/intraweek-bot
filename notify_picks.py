#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Optional

import pandas as pd
import yaml
import smtplib
from email.message import EmailMessage


# ----------------- config -----------------
def load_cfg(path: Path) -> dict:
    """Read YAML config and fall back to env vars (useful on GitHub)."""
    cfg = {}
    if path and path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("smtp", {})
    s = cfg["smtp"]

    s.setdefault("user", os.getenv("SMTP_USER", ""))
    s.setdefault("pass", os.getenv("SMTP_PASS", ""))
    s.setdefault("to",   os.getenv("SMTP_TO", ""))
    s.setdefault("from", os.getenv("SMTP_FROM", s.get("user", "")))
    s.setdefault("host", os.getenv("SMTP_HOST", "smtp.gmail.com"))
    s.setdefault("port", int(os.getenv("SMTP_PORT", "587")))

    # normalize recipients to list
    if isinstance(s["to"], str):
        s["to"] = [e.strip() for e in s["to"].split(",") if e.strip()]
    return cfg


# ----------------- picklist helpers -----------------
def load_picklist(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"ERROR: picklist not found: {path}", file=sys.stderr)
        return None

    df = pd.read_csv(path)
    if df.empty:
        print("DEBUG: picklist CSV is empty (headers only).")
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}
    ws_col = cols.get("week_start") or cols.get("weekstart") or cols.get("week")
    if not ws_col:
        print("WARN: no 'week_start' column present; columns:", list(df.columns))
        return pd.DataFrame()

    df["week_start"] = pd.to_datetime(df[ws_col], errors="coerce", utc=True).dt.date
    df = df[pd.notna(df["week_start"])]

    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("symbols")
    if sym_col and sym_col != "symbol":
        df = df.rename(columns={sym_col: "symbol"})
    if "symbol" not in df.columns:
        print("WARN: no 'symbol' column present; columns:", list(df.columns))
        return pd.DataFrame()

    return df


def choose_target_week(df: pd.DataFrame) -> Optional[date]:
    if df is None or df.empty:
        return None
    today = datetime.now(timezone.utc).astimezone().date()
    candidates = df.loc[df["week_start"] <= today, "week_start"]
    if candidates.empty:
        return None
    return candidates.max()


def load_symbol_names() -> dict[str, str]:
    # optional helper file produced by your workflow
    for p in (Path("symbol_names.csv"), Path("notifications/symbol_names.csv")):
        if p.exists():
            try:
                d = pd.read_csv(p)
                if {"symbol", "name"}.issubset(d.columns):
                    return dict(zip(d["symbol"], d["name"]))
            except Exception:
                pass
    return {}


# ----------------- email helpers -----------------
def format_lines(symbols: list[str], name_map: dict[str, str]) -> str:
    parts = []
    for i, sym in enumerate(symbols, 1):
        full = name_map.get(sym, "")
        parts.append(f"{i:>2}. {sym} — {full}" if full else f"{i:>2}. {sym}")
    return "\n".join(parts)


def send_email(cfg: dict, subject: str, body: str) -> None:
    s = cfg["smtp"]
    if not s.get("user") or not s.get("pass") or not s.get("to"):
        raise SystemExit("Missing SMTP settings in config_notify.yaml (or env).")

    msg = EmailMessage()
    from_addr = s.get("from") or s["user"]
    msg["From"] = from_addr
    msg["To"] = from_addr          # deliver to yourself…
    msg["Subject"] = subject
    msg.set_content(body)

    # …and BCC the actual recipients by sending to combined list (no Bcc header)
    recipients = [from_addr] + list(s["to"])

    with smtplib.SMTP(s.get("host", "smtp.gmail.com"), s.get("port", 587)) as server:
        server.starttls()
        server.login(s["user"], s["pass"])
        server.sendmail(from_addr, recipients, msg.as_string())


# ----------------- main -----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--topk", default="6")
    args = ap.parse_args()

    topk = int(args.topk)
    cfg = load_cfg(Path(args.config))
    df = load_picklist(Path(args.picklist))

    if df is None:
        raise SystemExit(1)

    week = choose_target_week(df)
    name_map = load_symbol_names()

    if week is None:
        subject = "Weekly picks — No picks this week"
        body = (
            "Hi,\n\nThere are no valid rows in the picklist for the current week.\n"
            "This usually means the picklist file is empty or week_start values are all in the future.\n\n"
            "Good trading,\nCEO of Kanute"
        )
        send_email(cfg, subject, body)
        print("INFO: sent 'No picks' email.")
        return

    dfw = df[df["week_start"] == week]
    if "rank" in dfw.columns:
        dfw = dfw.sort_values("rank")
    symbols = [str(s) for s in dfw["symbol"].tolist()][:topk]

    subject = f"Weekly picks — {week.isoformat()}"
    if symbols:
        lines = format_lines(symbols, name_map)
        body = (
            f"Hi,\n\nWeekly picks for week starting {week}:\n\n{lines}\n\n"
            f"Good trading,\nCEO of Kanute"
        )
    else:
        body = (
            f"Hi,\n\nNo symbols selected for week starting {week}.\n\n"
            f"Good trading,\nCEO of Kanute"
        )

    send_email(cfg, subject, body)
    print("INFO: email sent with subject:", subject)


if __name__ == "__main__":
    main()

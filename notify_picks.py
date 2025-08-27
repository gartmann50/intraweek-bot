#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
notify_picks.py
---------------
Reads a weekly picklist CSV, chooses the right week (first future week >= today,
else latest past week), selects Top-K symbols, and emails the list (recipients hidden via BCC).

Usage:
  python notify_picks.py --config config_notify.yaml \
      --picklist backtests/picklist_highrsi_trend.csv --topk 6 [--week YYYY-MM-DD] [--dry]

Config file (YAML) keys (or environment fallbacks):
  smtp_user   (env: SMTP_USER)   - required
  smtp_pass   (env: SMTP_PASS)   - required (App Password for Gmail)
  smtp_to     (env: SMTP_TO)     - required (list or comma/semicolon-separated)
  smtp_from   (env: SMTP_FROM)   - optional (defaults to smtp_user)
  smtp_host   (env: SMTP_HOST)   - default: smtp.gmail.com
  smtp_port   (env: SMTP_PORT)   - default: 587

The email subject is: "Weekly picks — week of YYYY-MM-DD"
Body includes a ranked list and a CSV line.

Signed:
  "Good trading, very good. — CEO of Kanute"
"""

from __future__ import annotations
import argparse
import os
import smtplib
import sys
from datetime import date, datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


# ---------- Utilities: dates & logging ----------

def today_local_date() -> date:
    """Return today's date in local time (UTC aware but converted to local)."""
    return datetime.now(timezone.utc).astimezone().date()


def normalize_week_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have a plain-date column 'week_start_date' to filter on safely.
    Works whether original is string/datetime/date.
    """
    out = df.copy()
    if "week_start" not in out.columns:
        raise ValueError("Picklist is missing required column 'week_start'.")
    out["week_start_date"] = pd.to_datetime(out["week_start"], errors="coerce").dt.date
    return out


def choose_target_week(df: pd.DataFrame, override: Optional[str] = None) -> Optional[date]:
    """
    Pick the week we should email:
      - if override provided, use that (must exist in data)
      - else first future (>= today), else latest past (<= today)
    """
    ws = df["week_start_date"].dropna()
    if ws.empty:
        return None

    if override:
        try:
            want = pd.to_datetime(override, errors="raise").date()
        except Exception:
            raise ValueError(f"--week '{override}' is not a valid date (YYYY-MM-DD)")
        return want if want in set(ws) else None

    today = today_local_date()
    future = ws[ws >= today]
    if not future.empty:
        return min(future)

    past = ws[ws <= today]
    return max(past) if not past.empty else None


def debug_print(label: str, value) -> None:
    print(f"DEBUG: {label}: {value}", flush=True)


# ---------- Optional company-name mapping ----------

def load_symbol_name_map() -> Dict[str, str]:
    """
    Try to load a symbol->company name mapping if a local file exists.
    Supported guesses: symbol_names.csv / .xlsx with columns like ['symbol','name'].
    If nothing is found, return {} (we’ll just display symbols).
    """
    candidates = [
        Path("symbol_names.csv"),
        Path("symbol_names.xlsx"),
        Path("universe/symbol_names.csv"),
        Path("universe/symbol_names.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix.lower() == ".csv":
                    df = pd.read_csv(p)
                else:
                    df = pd.read_excel(p)
                cols = {c.lower(): c for c in df.columns}
                if "symbol" in cols and ("name" in cols or "company" in cols):
                    sym_col = cols["symbol"]
                    name_col = cols.get("name", cols.get("company"))
                    mp = {str(r[sym_col]).strip().upper(): str(r[name_col]).strip()
                          for _, r in df.iterrows()
                          if pd.notna(r.get(sym_col)) and pd.notna(r.get(name_col))}
                    debug_print("loaded symbol_names", f"{len(mp)} entries from {p}")
                    return mp
            except Exception as e:
                debug_print("symbol name map load failed", f"{p}: {e}")
    return {}


# ---------- SMTP config ----------

def _env_list(key: str) -> Optional[List[str]]:
    val = os.getenv(key)
    if not val:
        return None
    parts = [x.strip() for x in val.replace(";", ",").split(",")]
    return [x for x in parts if x]


def load_smtp_config(path: Path) -> dict:
    cfg = {}
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # fallbacks from environment
    cfg.setdefault("smtp_user", os.getenv("SMTP_USER"))
    cfg.setdefault("smtp_pass", os.getenv("SMTP_PASS"))
    cfg.setdefault("smtp_from", os.getenv("SMTP_FROM"))
    cfg.setdefault("smtp_host", os.getenv("SMTP_HOST", "smtp.gmail.com"))
    cfg.setdefault("smtp_port", int(os.getenv("SMTP_PORT", "587")))

    # 'smtp_to' can be list in YAML or comma/semicolon list in env
    if "smtp_to" not in cfg or not cfg["smtp_to"]:
        env_to = _env_list("SMTP_TO")
        if env_to:
            cfg["smtp_to"] = env_to

    # normalize types
    if isinstance(cfg.get("smtp_to"), str):
        cfg["smtp_to"] = [x.strip() for x in cfg["smtp_to"].replace(";", ",").split(",") if x.strip()]

    # validation
    missing = [k for k in ("smtp_user", "smtp_pass", "smtp_to") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing SMTP settings ({', '.join(missing)}) in {path} (and env fallbacks).")

    if not cfg.get("smtp_from"):
        cfg["smtp_from"] = cfg["smtp_user"]

    debug_print("smtp_to", cfg["smtp_to"])
    debug_print("smtp_host", cfg.get("smtp_host"))
    debug_print("smtp_from", cfg.get("smtp_from"))
    return cfg


# ---------- Email compose/send ----------

def build_email_bodies(week: date, symbols: List[str], name_map: Dict[str, str]) -> tuple[str, str]:
    """Return (plain_text, html) bodies."""
    lines = []
    for i, s in enumerate(symbols, 1):
        nm = name_map.get(s.upper(), "")
        right = f"{s}" if not nm else f"{s} — {nm}"
        lines.append(f"{i:>2}. {right}")

    csv_line = ",".join(symbols)
    header = f"Weekly picks — week of {week.isoformat()}"
    footer = "Good trading, very good.\n\n— CEO of Kanute"

    text = (
        f"{header}\n\n"
        + "\n".join(lines) + ("\n" if lines else "")
        + f"\nCSV: {csv_line}\n\n"
        + footer + "\n"
    )

    # simple HTML (safe)
    html_items = "".join(
        f"<li><strong>{s}</strong>{' — ' + name_map.get(s.upper(), '') if name_map.get(s.upper()) else ''}</li>"
        for s in symbols
    )
    html = f"""\
<html><body>
  <h2 style="margin:0 0 10px 0;">{header}</h2>
  <ol style="padding-left:18px;margin-top:6px;">
    {html_items}
  </ol>
  <p><b>CSV:</b> {csv_line}</p>
  <p style="margin-top:18px;">Good trading, very good.<br>— CEO of Kanute</p>
</body></html>
"""
    return text, html


def send_email(cfg: dict, subject: str, text: str, html: str, dry: bool = False) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["smtp_from"]
    msg["To"] = cfg["smtp_from"]          # so recipients are hidden
    if cfg.get("smtp_to"):
        msg["Bcc"] = ", ".join(cfg["smtp_to"])

    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    if dry:
        print("\n===== DRY RUN (email preview) =====")
        print(msg)
        print("===== END DRY =====")
        return

    with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"]), timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.login(cfg["smtp_user"], cfg["smtp_pass"])
        s.send_message(msg)
    print(f"INFO: Email sent to {len(cfg.get('smtp_to', []))} recipient(s) (bcc).", flush=True)


# ---------- Main flow ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Email Top-K weekly picks (recipients hidden via BCC).")
    ap.add_argument("--config", required=True, help="Path to config_notify.yaml")
    ap.add_argument("--picklist", required=True, help="CSV with weekly picks (week_start, symbol, rank/score).")
    ap.add_argument("--topk", type=int, default=6, help="How many symbols to include (default 6).")
    ap.add_argument("--week", default=None, help="Override target week (YYYY-MM-DD).")
    ap.add_argument("--dry", action="store_true", help="Preview email instead of sending.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_smtp_config(cfg_path)

    picklist_path = Path(args.picklist)
    if not picklist_path.exists():
        raise FileNotFoundError(f"Picklist not found: {picklist_path}")

    raw = pd.read_csv(picklist_path)
    debug_print("picklist path", str(picklist_path))
    debug_print("picklist head(5)", "\n" + str(raw.head(5)))

    df = normalize_week_column(raw)
    # Choose week
    target_week = choose_target_week(df, override=args.week)
    debug_print("chosen week", target_week)

    if target_week is None:
        uniq = sorted(set(df["week_start_date"].dropna()))
        raise ValueError(
            "Could not parse any valid dates from week_start in picklist.\n"
            f"Available weeks: {uniq[-10:]}"
        )

    wk = df[df["week_start_date"] == target_week].copy()
    debug_print(f"rows for {target_week}", len(wk))

    if wk.empty:
        uniq = sorted(set(df["week_start_date"].dropna()))
        raise ValueError(
            f"No rows for target_week={target_week} in picklist.\n"
            f"Available weeks: {uniq[-10:]}"
        )

    # Identify symbol column
    sym_col = "symbol" if "symbol" in wk.columns else ("ticker" if "ticker" in wk.columns else None)
    if not sym_col:
        raise ValueError("Picklist must contain 'symbol' (or 'ticker') column.")
    # Sort
    if "rank" in wk.columns:
        wk = wk.sort_values("rank", ascending=True)
    elif "score" in wk.columns:
        wk = wk.sort_values("score", ascending=False)

    topk = max(1, int(args.topk))
    symbols = [str(s).strip().upper() for s in wk[sym_col].head(topk).tolist()]
    debug_print("selected symbols", symbols)

    if not symbols:
        raise ValueError("After filtering & sorting, no symbols to send.")

    # Optional names
    name_map = load_symbol_name_map()

    # Build and send
    subject = f"Weekly picks — week of {target_week.isoformat()}"
    text, html = build_email_bodies(target_week, symbols, name_map)
    send_email(cfg, subject, text, html, dry=args.dry)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

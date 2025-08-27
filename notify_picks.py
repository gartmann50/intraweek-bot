#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml
import smtplib
from email.message import EmailMessage

def load_cfg(path: Path) -> dict:
    if path and path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}
    # env fallbacks (useful on GitHub)
    cfg.setdefault("smtp", {})
    s = cfg["smtp"]
    s.setdefault("user", os.getenv("SMTP_USER", ""))
    s.setdefault("pass", os.getenv("SMTP_PASS", ""))
    s.setdefault("to",   os.getenv("SMTP_TO", ""))
    s.setdefault("from", os.getenv("SMTP_FROM", s.get("user","")))
    s.setdefault("host", os.getenv("SMTP_HOST", "smtp.gmail.com"))
    s.setdefault("port", int(os.getenv("SMTP_PORT", "587")))
    # normalize list of recipients
    if isinstance(s["to"], str):
        s["to"] = [e.strip() for e in s["to"].split(",") if e.strip()]
    return cfg

def load_picklist(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"ERROR: picklist not found: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if df.empty:
        print("DEBUG: picklist CSV has headers but no rows.")
        return pd.DataFrame()  # empty on purpose
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    ws_col = cols.get("week_start") or cols.get("weekstart") or cols.get("week")
    if not ws_col:
        print("WARN: no 'week_start' column in picklist; columns:", list(df.columns))
        return pd.DataFrame()
    df["week_start"] = pd.to_datetime(df[ws_col], errors="coerce", utc=True).dt.date
    # keep only rows with valid dates
    df = df[pd.notna(df["week_start"])]
    # standardize symbol col
    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("symbols")
    if sym_col and sym_col != "symbol":
        df = df.rename(columns={sym_col: "symbol"})
    if "symbol" not in df.columns:
        print("WARN: no 'symbol' column in picklist; columns:", list(df.columns))
        return pd.DataFrame()
    return df

def choose_target_week(df: pd.DataFrame) -> datetime.date | None:
    if df is None or df.empty:
        return None
    today_oslo = datetime.now(timezone.utc).astimezone().date()
    # target the most recent week_start <= today (Oslo date)
    candidates = df.loc[df["week_start"] <= today_oslo, "week_start"]
    if candidates.empty:
        return None
    return candidates.max()

def load_symbol_names() -> dict:
    # optional file created by your workflow step
    for p in [Path("symbol_names.csv"), Path("notifications/symbol_names.csv")]:
        if p.exists():
            try:
                d = pd.read_csv(p)
                if "symbol" in d.columns and "name" in d.columns:
                    return dict(zip(d["symbol"], d["name"]))
            except Exception:
                pass
    return {}

def format_lines(symbols:list[str], name_map:dict) -> str:
    lines = []
    for i, sym in enumerate(symbols, 1):
        full = name_map.get(sym, "")
        if full:
            lines.append(f"{i:>2}. {sym} — {full}")
        else:
            lines.append(f"{i:>2}. {sym}")
    return "\n".join(lines)

def send_email(cfg:dict, subject:str, body:str):
    s = cfg["smtp"]
    required = [s.get("user"), s.get("pass"), s.get("to")]
    if not all(required) or not s["to"]:
        raise SystemExit("Missing SMTP settings in config_notify.yaml (or env).")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = s.get("from") or s["user"]
    # hide recipients: deliver to yourself, BCC the list
    msg["To"] = s.get("from") or s["user"]
    msg.set_content(body)
    # connect + send
    with smtplib.SMTP(s.get("host","smtp.gmail.com"), s.get("port",587)) as server:
        server.starttls()
        server.login(s["user"], s["pass"])
        server.send_message(msg, to_addrs=[s.get("from") or s["user"]], bcc_addrs=s["to"])

def main():
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
        # either empty file or no week_start <= today
        subject = "Weekly picks — No picks this week"
        body = (
            "Hi,\n\n"
            "There are no valid rows in the picklist for the current week.\n"
            "This usually means the picklist file is empty or the week_start values are all in the future.\n\n"
            "Cheers,\nCEO of Kanute"
        )
        send_email(cfg, subject, body)
        print("INFO: sent 'No picks' email.")
        return

    # pick that week and top-K
    dfw = df[df["week_start"] == week]
    if "rank" in dfw.columns:
        dfw = dfw.sort_values("rank")
    symbols = [s for s in dfw["symbol"].astype(str).tolist()][:topk]

    subject = f"Weekly picks — {week.isoformat()}"
    if symbols:
        lines = format_lines(symbols, name_map)
        body = (
            f"Hi,\n\n"
            f"Weekly picks for week starting {week}:\n\n{lines}\n\n"
            f"Good trading,\nCEO of Kanute"
        )
    else:
        body = (
            f"Hi,\n\n"
            f"No symbols selected for week starting {week}.\n\n"
            f"Good trading,\nCEO of Kanute"
        )
    send_email(cfg, subject, body)
    print("INFO: email sent with subject:", subject)

if __name__ == "__main__":
    main()

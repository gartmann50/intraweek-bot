# notify_picks.py
import argparse, csv, ssl, smtplib
from email.message import EmailMessage
from pathlib import Path
import pandas as pd
import yaml

def load_picklist(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'week_start' not in df.columns or 'symbol' not in df.columns:
        raise SystemExit(f"Picklist must have columns week_start,symbol. Got: {list(df.columns)}")
    df['week_start'] = pd.to_datetime(df['week_start']).dt.date
    return df

def load_names() -> dict:
    """
    Tries a few locations for a simple mapping: symbol,name
    If not found, returns {} and we’ll print symbols only.
    """
    candidates = [
        Path("universe/symbol_names.csv"),
        Path("symbol_names.csv"),
    ]
    for p in candidates:
        if p.exists():
            m = {}
            with p.open(newline='', encoding='utf-8') as f:
                r = csv.DictReader(f)
                sym_col = 'symbol' if 'symbol' in r.fieldnames else r.fieldnames[0]
                name_col = 'name' if 'name' in r.fieldnames else r.fieldnames[1]
                for row in r:
                    s = (row[sym_col] or '').strip().upper()
                    n = (row[name_col] or '').strip()
                    if s:
                        m[s] = n
            return m
    return {}

def format_email(week: str, symbols: list[str], names: dict, subject_prefix: str):
    lines = [f"{subject_prefix} — {week}", "", "Top-6:"]
    for i, sym in enumerate(symbols, 1):
        nm = names.get(sym, "")
        name_part = f" — {nm}" if nm else ""
        lines.append(f"{i}. {sym}{name_part}")
    lines += [
        "",
        f"CSV: {','.join(symbols)}",
        "",
        "Good trading — very good.",
        "",
        "— CEO of Kanute",
    ]
    body = "\n".join(lines)
    subject = f"{subject_prefix} — {week}"
    return subject, body

def send_email(cfg: dict, subject: str, body: str):
    smtp = cfg["smtp"]
    user = smtp["user"]
    pwd  = smtp["pass"]
    host = smtp.get("host", "smtp.gmail.com")
    port = int(smtp.get("port", 587))
    to   = smtp["to"]  # list or string
    from_addr = smtp.get("from", user)

    if isinstance(to, str):
        to_list = [t.strip() for t in to.split(",") if t.strip()]
    else:
        to_list = list(to)

    msg = EmailMessage()
    msg["From"] = from_addr
    # Hide recipients: deliver to yourself, Bcc actual list
    msg["To"] = user
    msg["Bcc"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, pwd)
        s.send_message(msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", default="backtests/picklist_highrsi_trend.csv")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", default="latest", help="YYYY-MM-DD or 'latest'")
    ap.add_argument("--subject-prefix", default="Weekly Picks")
    ap.add_argument("--dry-run", action="store_true", help="Print email instead of sending")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    df = load_picklist(args.picklist)
    if args.week.lower() == "latest" or not args.week:
        week_date = df["week_start"].max()
    else:
        week_date = pd.to_datetime(args.week).date()

    sel = df[df["week_start"] == week_date].copy()
    # Keep rank if present, otherwise sort by symbol for stability
    if "rank" in sel.columns:
        sel = sel.sort_values(["rank", "symbol"])
    else:
        sel = sel.sort_values("symbol")

    symbols = sel["symbol"].head(args.topk).tolist()
    if not symbols:
        raise SystemExit(f"No rows for week_start={week_date} in {args.picklist}")

    names = load_names()
    subject, body = format_email(week_date.isoformat(), symbols, names, args.subject_prefix)

    if args.dry_run:
        print(subject)
        print("-" * len(subject))
        print(body)
        return

    send_email(cfg, subject, body)
    print("Email sent.")

if __name__ == "__main__":
    main()

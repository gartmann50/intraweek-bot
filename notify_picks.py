import argparse, csv, ssl, smtplib
from email.message import EmailMessage
from pathlib import Path
import pandas as pd
import yaml

def load_picklist(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Picklist not found: {path}")

    # Read as-is, then normalize headers
    df = pd.read_csv(p)
    # strip whitespace and lower-case all column names
    df.rename(columns=lambda c: c.strip(), inplace=True)
    df.rename(columns=lambda c: c.lower(), inplace=True)

    if "week_start" not in df.columns or "symbol" not in df.columns:
        raise SystemExit(f"Picklist must have columns: week_start,symbol. Got: {list(df.columns)}")

    # Force-parse week_start regardless of dtype, then convert to date
    dt = pd.to_datetime(df["week_start"], errors="coerce", utc=False)
    if dt.isna().all():
        raise SystemExit("Could not parse any valid dates from week_start in picklist.")
    df["week_start"] = dt.dt.date
    return df

def load_names() -> dict:
    for p in (Path("universe/symbol_names.csv"), Path("symbol_names.csv")):
        if p.exists():
            out = {}
            with p.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                # try common headers, fall back to first two columns
                sym_col = "symbol" if "symbol" in r.fieldnames else r.fieldnames[0]
                name_col = "name"   if "name"   in r.fieldnames else r.fieldnames[1]
                for row in r:
                    s = (row.get(sym_col, "") or "").strip().upper()
                    n = (row.get(name_col, "") or "").strip()
                    if s:
                        out[s] = n
            return out
    return {}

def pick_week(df: pd.DataFrame, requested: str | None):
    if requested and requested.lower() != "latest":
        wd = pd.to_datetime(requested, errors="coerce")
        wd = wd.date() if pd.notna(wd) else None
    else:
        wd = None

    if wd is None:
        return df["week_start"].max()

    if (df["week_start"] == wd).any():
        return wd
    latest = df["week_start"].max()
    print(f"[WARN] Requested week {requested} not found; using latest {latest}.")
    return latest

def sorted_symbols_for_week(df: pd.DataFrame, week_date, topk: int) -> list[str]:
    sel = df[df["week_start"] == week_date].copy()
    if sel.empty:
        return []
    if "rank" in sel.columns:
        sel = sel.sort_values(["rank", "symbol"])
    elif "score" in sel.columns:
        sel = sel.sort_values(["score", "symbol"], ascending=[False, True])
    else:
        sel = sel.sort_values("symbol")
    return sel["symbol"].head(topk).tolist()

def format_email(week: str, symbols: list[str], names: dict, subject_prefix: str):
    lines = [f"{subject_prefix} — {week}", "", "Top-6:"]
    for i, sym in enumerate(symbols, 1):
        nm = names.get(sym, "")
        lines.append(f"{i}. {sym}" + (f" — {nm}" if nm else ""))
    lines += [
        "",
        f"CSV: {','.join(symbols)}",
        "",
        "Good trading — very good.",
        "",
        "— CEO of Kanute",
    ]
    return f"{subject_prefix} — {week}", "\n".join(lines)

def send_email(cfg: dict, subject: str, body: str):
    smtp = cfg["smtp"]
    user = smtp["user"]
    pwd  = smtp["pass"]
    host = smtp.get("host", "smtp.gmail.com")
    port = int(smtp.get("port", 587))
    to   = smtp["to"]
    from_addr = smtp.get("from", user)

    to_list = [t.strip() for t in (to if isinstance(to, list) else to.split(",")) if t.strip()]

    msg = EmailMessage()
    msg["From"] = from_addr
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df  = load_picklist(args.picklist)
    week_date = pick_week(df, args.week)
    symbols   = sorted_symbols_for_week(df, week_date, args.topk)

    if not symbols:
        avail = sorted({d.isoformat() for d in df["week_start"].dropna().unique()})
        raise SystemExit(
            f"No rows for week_start={week_date} in {args.picklist}\n"
            f"Available weeks: {', '.join(avail[-8:]) or '(none)'}"
        )

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

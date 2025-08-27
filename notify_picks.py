#!/usr/bin/env python3
import argparse, os, sys, smtplib, ssl, yaml
import pandas as pd
from email.message import EmailMessage

# ----------------------------- utils ---------------------------------
def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_company_names():
    """
    Optional: use symbol_names.xlsx or symbol_names.csv if present
    to enrich symbols with full company names.
    """
    for cand in ("symbol_names.xlsx", "symbol_names.csv"):
        if os.path.exists(cand):
            try:
                if cand.endswith(".xlsx"):
                    df = pd.read_excel(cand)
                else:
                    df = pd.read_csv(cand)
                df.columns = [c.strip().lower() for c in df.columns]
                c_sym = next((c for c in ("symbol","ticker") if c in df.columns), None)
                c_name = next((c for c in ("name","company","company_name") if c in df.columns), None)
                if c_sym and c_name:
                    return {str(r[c_sym]).upper(): str(r[c_name]) for _, r in df.iterrows()}
            except Exception:
                pass
    return {}

def read_picklist(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Picklist not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Picklist is empty.")
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError("Picklist must contain a 'symbol' column.")
    return df

def choose_week(df: pd.DataFrame, until: str | None) -> tuple[pd.DataFrame, str]:
    """
    Return (subset_for_week, week_label_str).
    Tries to parse dates; falls back to string comparisons if needed.
    """
    # find week column
    week_cands = ["week_start", "week_start_date", "weekstart", "week"]
    wk = next((c for c in week_cands if c in df.columns), None)
    if not wk:
        raise ValueError(f"No week column found (tried {week_cands}). Columns: {list(df.columns)}")

    s = pd.to_datetime(df[wk], errors="coerce")
    if s.notna().any():
        df["_wk"] = s
        try:
            until_dt = pd.to_datetime(until) if until else df["_wk"].max()
        except Exception:
            until_dt = df["_wk"].max()
        latest = df.loc[df["_wk"] <= until_dt, "_wk"].max()
        sub = df[df["_wk"] == latest].copy() if pd.notna(latest) else pd.DataFrame()
        label = latest.strftime("%Y-%m-%d") if pd.notna(latest) else ""
    else:
        # fallback: string compare on YYYY-MM-DD prefix
        ss = df[wk].astype(str).str.slice(0, 10)
        df["_wkstr"] = ss
        # pick the max label <= until (if provided), else overall max
        if until and len(until) >= 10:
            ss2 = ss[ss <= until[:10]]
            latest_s = ss2.max() if not ss2.empty else ss.max()
        else:
            latest_s = ss.max()
        sub = df[df["_wkstr"] == latest_s].copy()
        label = latest_s

    if sub.empty:
        raise ValueError("Could not parse any valid dates from week_start in picklist.")
    return sub, label

def select_top(sub: pd.DataFrame, topk: int) -> pd.DataFrame:
    if "rank" in sub.columns:
        sub = sub.sort_values("rank", ascending=True)
    elif "score" in sub.columns:
        sub = sub.sort_values("score", ascending=False)
    return sub.head(topk)[["symbol"] + [c for c in ("rank","score") if c in sub.columns]]

def format_lines(rows: pd.DataFrame, name_map: dict[str,str]) -> list[str]:
    lines = []
    for i, r in rows.reset_index(drop=True).iterrows():
        sym = str(r["symbol"]).upper()
        name = name_map.get(sym, "")
        suffix = f" — {name}" if name else ""
        rank = f" (rank {int(r['rank'])})" if "rank" in r and pd.notna(r["rank"]) else ""
        score = f" (score {r['score']:.2f})" if "score" in r and pd.notna(r["score"]) else ""
        lines.append(f"{i+1}. {sym}{suffix}{rank}{score}")
    return lines

def build_message(subject: str, body: str, sender: str, tos: list[str]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    # To: show only sender to hide recipients; put real recipients in Bcc
    msg["To"] = sender
    msg["Bcc"] = ", ".join(tos)
    msg.set_content(body)
    return msg

def send_smtp_gmail(user: str, password: str, msg: EmailMessage):
    host, port = "smtp.gmail.com", 587
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.ehlo()
        s.starttls(context=context)
        s.ehlo()
        s.login(user, password)
        s.send_message(msg)

# ----------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_notify.yaml")
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--week", default="")  # optional manual override YYYY-MM-DD
    args = ap.parse_args()

    # 1) SMTP config
    # 1) SMTP config
cfg = {}
try:
    cfg = load_yaml(args.config) if os.path.exists(args.config) else {}
except Exception:
    cfg = {}

# fallbacks from environment so Actions can still work even if the file is empty
user = (cfg.get("smtp_user") or os.getenv("SMTP_USER", "")).strip()
pwd  = (cfg.get("smtp_pass") or os.getenv("SMTP_PASS", "")).strip()
tos  = (cfg.get("smtp_to")   or os.getenv("SMTP_TO",   "")).strip()
sender = (cfg.get("smtp_from") or os.getenv("SMTP_FROM", "") or user).strip()

if not (user and pwd and tos):
    print("Missing SMTP settings in config_notify.yaml (and env fallbacks).")
    sys.exit(1)

recipients = [t.strip() for t in tos.split(",") if t.strip()]


    # 2) Picklist load + choose week
    df = read_picklist(args.picklist)
    # Prefer explicit --week, else env UNTIL (workflow), else latest in file
    until_hint = args.week or os.environ.get("UNTIL", "")
    sub, label = choose_week(df, until_hint)

    # 3) Select top-k
    top = select_top(sub, max(1, int(args.topk)))

    # 4) Company names (optional)
    name_map = load_company_names()
    lines = format_lines(top, name_map)

    # 5) Build email
    subject = f"Weekly Picks — Week of {label}"
    if lines:
        picks_block = "\n".join(lines)
    else:
        picks_block = "(no symbols found)"

    body = (
        f"Weekly Picks for the week of {label}\n\n"
        f"Top {len(lines)}:\n{picks_block}\n\n"
        f"Good trading — very good.\n"
        f"CEO, Kanute\n"
    )

    msg = build_message(subject, body, sender, recipients)

    # 6) Send (Gmail TLS)
    send_smtp_gmail(user, pwd, msg)
    print(f"Emailed weekly picks for {label} to {len(recipients)} recipient(s).")

if __name__ == "__main__":
    main()

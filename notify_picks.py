#!/usr/bin/env python3
import argparse, os, ssl, smtplib, sys
from email.message import EmailMessage
from datetime import datetime
import pandas as pd, yaml

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_smtp(cfg):
    s = cfg.get("smtp", {}) or {}
    host = s.get("host") or cfg.get("smtp_host")
    port = s.get("port") or cfg.get("smtp_port") or 587
    user = s.get("user") or cfg.get("smtp_user")
    pw   = s.get("password") or cfg.get("smtp_password")
    tls  = s.get("starttls", True) if "smtp" in cfg else cfg.get("smtp_starttls", True)
    from_addr = s.get("from") or cfg.get("smtp_from") or user
    # recipients: nested list or flat single
    recips = cfg.get("recipients")
    if not recips:
        to = cfg.get("smtp_to")
        recips = [to] if to else []
    return host, int(port), user, pw, bool(tls), from_addr, list(recips)

def build_body(picklist_path, topk):
    df = pd.read_csv(picklist_path)
    if "week_start" not in df or "symbol" not in df:
        raise ValueError("picklist missing columns (week_start, symbol)")
    wk = df["week_start"].max()
    sub = df[df["week_start"] == wk].copy()
    if "rank" in sub:
        sub = sub.sort_values("rank")
    elif "score" in sub:
        sub = sub.sort_values("score", ascending=False)
    syms = sub["symbol"].head(topk).tolist()
    lines = [f"Weekly picks — week start {wk}", ""]
    for i, s in enumerate(syms, 1):
        lines.append(f"{i:>2}. {s}")
    lines += ["", "good trading — very good.", "— CEO of Kanute"]
    return wk, syms, "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--picklist", required=True)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    host, port, user, pw, use_tls, from_addr, recips = get_smtp(cfg)
    if not (host and user and pw and recips):
        print("Missing SMTP settings in config_notify.yaml", file=sys.stderr)
        sys.exit(1)

    wk, syms, body = build_body(args.picklist, args.topk)
    msg = EmailMessage()
    msg["Subject"] = "Weekly picks"
    msg["From"] = from_addr or user
    msg["To"] = user  # send to yourself
    # hide recipients as Bcc
    for r in recips:
        if r and r != user:
            msg["Bcc"] = (msg.get("Bcc") + "," if msg.get("Bcc") else "") + r
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as s:
        if use_tls:
            s.starttls(context=ssl.create_default_context())
        s.login(user, pw)
        s.send_message(msg)

    print("Sent weekly picks for", wk, "->", syms)

if __name__ == "__main__":
    main()

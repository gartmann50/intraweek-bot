import os, smtplib
from email.mime.text import MIMEText

def _need(name):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"Missing required env {name}. "
            "Set it in the workflow step env: section (mapped from repo secrets)."
        )
    return v

def send_email(cfg, subject, body):
    # Read envs (with sensible defaults)
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = _need("SMTP_USER")     # required
    pwd  = _need("SMTP_PASS")     # required
    from_addr = os.getenv("SMTP_FROM", user)
    to_raw = _need("SMTP_TO")     # required
    to_addrs = [x.strip() for x in to_raw.split(",") if x.strip()]

    # Gmail safety: ensure From matches authenticated user unless alias configured
    if from_addr.strip().lower() != user.strip().lower():
        from_addr = user

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    print(f"[email] host={host} port={port} from={from_addr} to={to_addrs}")

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30) as s:
            s.login(user, pwd)
            s.sendmail(from_addr, to_addrs, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(user, pwd)
            s.sendmail(from_addr, to_addrs, msg.as_string())

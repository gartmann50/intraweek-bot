import os, smtplib
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid

def _need(name):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env {name}.")
    return v

def send_email(cfg, subject, body):
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = _need("SMTP_USER")
    pwd  = _need("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", user)
    to_raw = _need("SMTP_TO")
    to_addrs = [x.strip() for x in to_raw.split(",") if x.strip()]

    # Force From==auth user unless alias configured in Gmail
    if from_addr.strip().lower() != user.strip().lower():
        from_addr = user

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=user.split("@",1)[1])
    msg["Reply-To"] = from_addr
    msg["Auto-Submitted"] = "auto-generated"

    print(f"[email] host={host} port={port} from={from_addr}")
    print(f"[email] TO: {to_addrs}")
    print(f"[email] SUBJECT: {subject}")

    debug = os.getenv("SMTP_DEBUG","0") == "1"

    def send_with(s):
        if debug:
            s.set_debuglevel(1)  # prints SMTP dialogue (no secrets)
        s.ehlo() if port != 465 else None
        if port != 465:
            s.starttls(); s.ehlo()
        s.login(user, pwd)
        res = s.sendmail(from_addr, to_addrs, msg.as_string())
        print(f"[email] SMTP sendmail() returned: {res!r}")  # {} means all accepted

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30) as s:
            send_with(s)
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            send_with(s)

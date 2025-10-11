#!/usr/bin/env python3
import os, smtplib, pathlib as p
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# SMTP config (Gmail works with App Password)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)

# Recipients: put EVERYONE in SMTP_TO (comma-separated). They will be BCC'ed.
# Visible "To:" will be "Undisclosed recipients:;" so no one sees others.
SMTP_TO_BCC = os.getenv("SMTP_TO", "")  # comma-separated
VISIBLE_TO  = os.getenv("VISIBLE_TO", "Undisclosed recipients:;")

SUBJECT   = os.getenv("EMAIL_SUBJECT", "IW Bot — Weekly Picks")
ASSET_DIR = p.Path("backtests/email_charts")
HTML_FILE = ASSET_DIR / "email.html"

def main():
    html = HTML_FILE.read_text(encoding="utf-8") if HTML_FILE.exists() else "<p>No content</p>"

    msg = MIMEMultipart("related")
    msg["Subject"] = SUBJECT
    msg["From"]    = SMTP_FROM
    msg["To"]      = VISIBLE_TO            # <- visible To only (no CC/BCC headers)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText("Your email client does not support HTML.", "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)

    # Attach inline PNGs as cid:<filename>
    if ASSET_DIR.exists():
        for png in sorted(ASSET_DIR.glob("*.png")):
            with open(png, "rb") as f:
                im = MIMEImage(f.read(), _subtype="png", name=png.name)
                im.add_header("Content-ID", f"<{png.name}>")
                im.add_header("Content-Disposition", "inline", filename=png.name)
                msg.attach(im)

    # Envelope recipients = BCC list (hidden)
    rcpts = [x.strip() for x in SMTP_TO_BCC.split(",") if x.strip()]
    if not rcpts:
        raise SystemExit("No recipients: set SMTP_TO to a comma-separated list.")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.ehlo(); s.starttls(); s.ehlo()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_FROM, rcpts, msg.as_string())

if __name__ == "__main__":
    main()

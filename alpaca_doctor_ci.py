import os, sys, json, yaml
from pathlib import Path
from alpaca_trade_api import REST

def load_cfg(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))

def _ok(x): return x if x else None

def main():
    cfg_path = "config_highrsi.yaml"
    if "--config" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config")+1]
    cfg = load_cfg(cfg_path)

    alp = cfg.get("alpaca", {}) or {}
    key = _ok(alp.get("api_key_id") or os.getenv("ALPACA_API_KEY_ID"))
    sec = _ok(alp.get("secret_key") or os.getenv("ALPACA_API_SECRET_KEY"))
    url = _ok(alp.get("base_url")   or os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets")
    if not (key and sec):
        raise SystemExit("ERROR: missing Alpaca credentials in secrets/config.")

    api = REST(key, sec, url)
    acct = api.get_account()
    print(json.dumps({
        "account_id": acct.id,
        "status": acct.status,
        "equity": acct.equity,
        "buying_power": acct.buying_power
    }, indent=2))

    u = os.getenv("SMTP_USER"); p = os.getenv("SMTP_PASS")
    print("SMTP:", "present" if (u and p) else "missing (ok for status)")

if __name__ == "__main__":
    main()

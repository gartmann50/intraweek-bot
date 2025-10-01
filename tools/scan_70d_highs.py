import time
import requests
from urllib.parse import urlparse, parse_qs

def get_all_symbols_by_cap(api_key: str, cap_min: float = 1_000_000_000, max_pages: int = 5):
    """
    Pull US common stocks from Polygon v3/reference/tickers, filter by market cap,
    and return tickers sorted locally by market cap (desc).
    """
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "locale": "us",
        "type": "CS",       # common stock
        "limit": 1000,
        "apiKey": api_key,
        # NOTE: do NOT pass sort/order here; Polygon returns 400 for unsupported sorts
    }

    tickers = []  # (ticker, market_cap)
    pages = 0
    cursor = None

    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor

        # simple backoff for occasional 429s
        for attempt in range(6):
            r = requests.get(url, params=q, timeout=30)
            if r.status_code != 429:
                break
            time.sleep(0.5 * (2 ** attempt))

        r.raise_for_status()
        data = r.json() or {}

        for item in data.get("results", []) or []:
            mc = item.get("market_cap") or 0
            if mc and mc >= cap_min:
                tickers.append((item.get("ticker"), float(mc)))

        # pagination: v3 returns next_url with ?cursor=...
        next_url = data.get("next_url")
        if not next_url:
            break

        # extract cursor token
        try:
            cursor = parse_qs(urlparse(next_url).query).get("cursor", [None])[0]
        except Exception:
            cursor = None

        pages += 1
        if pages >= max_pages or not cursor:
            break

    # local sort by market cap desc
    tickers.sort(key=lambda t: t[1], reverse=True)
    return [t for t, _ in tickers]

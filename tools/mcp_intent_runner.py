#!/usr/bin/env python3
"""
MCP Intent Runner

Reads an intent JSON file (from GitHub repository_dispatch),
validates it, optionally looks up prices from Polygon,
and sends orders to Alpaca.

This is written to be simple + copy-paste friendly.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Literal

import alpaca_trade_api as tradeapi
import polygon


# --------- Types ---------

IntentType = Literal["buy", "close", "close_all"]


@dataclass
class MCPIntent:
    intent: IntentType
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    notional: Optional[float] = None
    side: str = "BUY"
    time_in_force: str = "DAY"
    dry_run: bool = True
    comment: Optional[str] = None
    meta: dict | None = None


# --------- Logging helper ---------

def log(msg: str) -> None:
    print(msg, flush=True)


# --------- Env helpers ---------

def get_env(name: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# --------- Clients (Alpaca + Polygon) ---------

_alpaca_client = None
_polygon_client = None


def get_alpaca_client():
    """Create (or reuse) an Alpaca REST client."""
    global _alpaca_client
    if _alpaca_client is None:
        api_key = get_env("ALPACA_API_KEY")
        api_secret = get_env("ALPACA_API_SECRET")
        base_url = get_env("ALPACA_BASE_URL")
        log(f"Connecting to Alpaca at {base_url}")
        _alpaca_client = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
        )
    return _alpaca_client


def get_polygon_client():
    """Create (or reuse) a Polygon Stocks client."""
    global _polygon_client
    if _polygon_client is None:
        api_key = get_env("POLYGON_API_KEY")
        log("Connecting to Polygon Stocks API")
        _polygon_client = polygon.StocksClient(api_key)
    return _polygon_client


def get_polygon_last_price(symbol: str) -> Optional[float]:
    """
    Get current market price for a ticker from Polygon.
    If anything fails, returns None instead of crashing.
    """
    try:
        client = get_polygon_client()
        price = client.get_current_price(symbol)  # uses get_last_trade under the hood
        # polygon.StocksClient.get_current_price(symbol) returns a float in client v1.2.8
        log(f"Polygon price for {symbol}: {price}")
        return float(price)
    except Exception as e:
        log(f"WARNING: Could not fetch Polygon price for {symbol}: {e}")
        return None


# --------- Load + validate intent ---------

def load_intent_from_file(path: str) -> MCPIntent:
    log(f"Loading MCP intent from {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def get(key, default=None):
        return raw.get(key, default)

    intent = get("intent")
    if intent not in ("buy", "close", "close_all"):
        raise ValueError(f"Unsupported intent: {intent}")

    symbol = get("symbol")
    if intent in ("buy", "close") and not symbol:
        raise ValueError(f"Intent '{intent}' requires 'symbol'")

    quantity = get("quantity")
    notional = get("notional")

    if intent == "buy" and not (quantity or notional):
        raise ValueError("Buy intent requires 'quantity' or 'notional'")

    dry_run = bool(get("dry_run", True))

    return MCPIntent(
        intent=intent,
        symbol=symbol,
        quantity=quantity,
        notional=notional,
        side=get("side", "BUY").upper(),
        time_in_force=get("time_in_force", "DAY").upper(),
        dry_run=dry_run,
        comment=get("comment"),
        meta=get("meta", {}),
    )


# --------- Trading actions ---------

def ensure_live_allowed(intent: MCPIntent) -> None:
    """Block live trading unless TRADING_ENV=PROD."""
    trading_env = os.getenv("TRADING_ENV", "PAPER").upper()
    if not intent.dry_run and trading_env != "PROD":
        raise RuntimeError(
            "LIVE trading (dry_run=false) is blocked because TRADING_ENV != 'PROD'. "
            "Set TRADING_ENV=PROD in GitHub secrets ONLY when you are ready."
        )


def place_buy_order(intent: MCPIntent) -> None:
    assert intent.symbol
    ensure_live_allowed(intent)

    # Decide quantity: if only notional is given, try to use Polygon price.
    qty = intent.quantity
    if qty is None and intent.notional is not None:
        last_price = get_polygon_last_price(intent.symbol)
        if last_price is not None and last_price > 0:
            qty = round(intent.notional / last_price, 4)  # allow fractional shares
            log(f"Computed quantity from notional: {qty} shares at ~{last_price}")
        else:
            raise RuntimeError("Could not determine quantity from notional (Polygon price unavailable)")

    if qty is None or qty <= 0:
        raise ValueError(f"Invalid quantity: {qty}")

    # Alpaca expects lowercase time_in_force (day, gtc, ...).
    tif = intent.time_in_force.lower()

    action = "[DRY RUN]" if intent.dry_run else "[LIVE]"
    log(f"{action} BUY {intent.symbol} qty={qty}, TIF={tif}")

    # For dry_run, just log and return.
    if intent.dry_run:
        if intent.comment:
            log(f"Comment: {intent.comment}")
        return

    api = get_alpaca_client()
    order = api.submit_order(
        symbol=intent.symbol,
        qty=qty,
        side="buy",
        type="market",
        time_in_force=tif,
    )
    log(f"Alpaca order submitted: {order}")


def close_position_symbol(intent: MCPIntent) -> None:
    assert intent.symbol
    ensure_live_allowed(intent)
    action = "[DRY RUN]" if intent.dry_run else "[LIVE]"
    log(f"{action} CLOSE position for {intent.symbol}")

    if intent.dry_run:
        if intent.comment:
            log(f"Comment: {intent.comment}")
        return

    api = get_alpaca_client()
    result = api.close_position(intent.symbol)
    log(f"Alpaca close_position result: {result}")


def close_all_positions(intent: MCPIntent) -> None:
    ensure_live_allowed(intent)
    action = "[DRY RUN]" if intent.dry_run else "[LIVE]"
    log(f"{action} CLOSE ALL positions")

    if intent.dry_run:
        if intent.comment:
            log(f"Comment: {intent.comment}")
        return

    api = get_alpaca_client()
    result = api.close_all_positions()
    log(f"Alpaca close_all_positions result: {result}")


def execute_intent(intent: MCPIntent) -> None:
    log(f"Received intent: {intent}")
    if intent.intent == "buy":
        place_buy_order(intent)
    elif intent.intent == "close":
        close_position_symbol(intent)
    elif intent.intent == "close_all":
        close_all_positions(intent)
    else:
        raise ValueError(f"Unknown intent: {intent.intent}")


# --------- CLI ---------

def parse_args():
    parser = argparse.ArgumentParser(description="MCP Intent Runner")
    parser.add_argument(
        "--payload-file",
        required=True,
        help="Path to JSON file with MCP intent payload (from GitHub repository_dispatch)",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        intent = load_intent_from_file(args.payload_file)
        execute_intent(intent)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

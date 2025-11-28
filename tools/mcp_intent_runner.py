#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Literal

# ----- Data structures -----

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

# ----- Parsing / validation -----

def load_intent_from_file(path: str) -> MCPIntent:
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
        side=get("side", "BUY"),
        time_in_force=get("time_in_force", "DAY"),
        dry_run=dry_run,
        comment=get("comment"),
        meta=get("meta", {}),
    )

# ----- Trading hooks (replace with your integration) -----

def log(msg: str):
    # Simple logging for Actions output
    print(msg, flush=True)

def place_buy_order(intent: MCPIntent):
    """
    TODO: plug this into your actual trading code.
    For now, this just prints what it would do.
    """
    assert intent.symbol
    action = "[DRY RUN]" if intent.dry_run else "[LIVE]"
    sizing_str = (
        f"{intent.quantity} shares"
        if intent.quantity is not None
        else f"${intent.notional} notional"
    )
    log(f"{action} BUY intent for {intent.symbol}: {sizing_str}, TIF={intent.time_in_force}")
    if intent.comment:
        log(f"Comment: {intent.comment}")

    # Example: if you want to call an internal script instead:
    # cmd = [
    #     sys.executable,
    #     "tools/place_weekly_buys.py",
    #     "--symbol", intent.symbol,
    #     "--quantity", str(intent.quantity or 0),
    #     "--dry-run" if intent.dry_run else "--live",
    # ]
    # subprocess.run(cmd, check=True)

def close_position_symbol(intent: MCPIntent):
    assert intent.symbol
    action = "[DRY RUN]" if intent.dry_run else "[LIVE]"
    log(f"{action} CLOSE position for {intent.symbol}")
    if intent.comment:
        log(f"Comment: {intent.comment}")

    # TODO: integrate with your broker API / moc_flatten logic

def close_all_positions(intent: MCPIntent):
    action = "[DRY RUN]" if intent.dry_run else "[LIVE]"
    log(f"{action} CLOSE ALL positions requested")
    if intent.comment:
        log(f"Comment: {intent.comment}")

    # TODO: integrate with your broker / portfolio snapshot

def execute_intent(intent: MCPIntent):
    log(f"Received intent: {intent}")

    # Safety: disallow LIVE unless TRADING_ENV says it's okay
    trading_env = os.getenv("TRADING_ENV", "PAPER").upper()
    if intent.dry_run is False and trading_env != "PROD":
        log("Refusing LIVE execution because TRADING_ENV != 'PROD'")
        raise RuntimeError("LIVE trading blocked in non-PROD environment")

    if intent.intent == "buy":
        place_buy_order(intent)
    elif intent.intent == "close":
        close_position_symbol(intent)
    elif intent.intent == "close_all":
        close_all_positions(intent)
    else:
        raise ValueError(f"Unknown intent: {intent.intent}")

# ----- CLI -----

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
        # Make sure GitHub Action fails if something is wrong
        log(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

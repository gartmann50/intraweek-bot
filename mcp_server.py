#!/usr/bin/env python3
"""
Claude Trading MCP Server

- Connects to Alpaca (paper or live, depending on ALPACA_BASE_URL)
- Optional Polygon price data
- Sends analytics to your Vercel app via webhook
- Exposes MCP tools for trading & analytics
"""

import os
import json
import asyncio
import base64
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, Dict, List, Any, Union

import requests
import alpaca_trade_api as tradeapi
from polygon import RESTClient as PolygonClient

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent
from mcp.server.http import StreamableHTTPSessionManager
import uvicorn

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-trading-mcp")

# --------------------------------------------------------------------------- #
# Environment & config
# --------------------------------------------------------------------------- #

# Prefer ALPACA_* names, but fall back to APCA_* if needed
ALPACA_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID") or os.getenv(
    "APCA_API_KEY_ID"
)
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv(
    "ALPACA_API_SECRET_KEY"
) or os.getenv("APCA_API_SECRET_KEY")

# Default to PAPER if not explicitly set
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL") or os.getenv(
    "APCA_API_BASE_URL", "https://paper-api.alpaca.markets"
)

POLYGON_KEY = os.getenv("POLYGON_API_KEY")

# Web app / analytics
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://iw-positions-app.vercel.app")
ANALYTICS_WEBHOOK = os.getenv("ANALYTICS_ENDPOINT", f"{WEBAPP_URL}/api/analytics")

# Risk limits
MAX_POSITION_SIZE = int(os.getenv("MAX_POSITION_SIZE", "1000"))
MAX_POSITION_VALUE = float(os.getenv("MAX_POSITION_VALUE", "10000"))

ALLOWED_SYMBOLS_FILE = "data/universe_liquid.txt"

# Log config summary (never secrets)
logger.info("=== MCP Server Config ===")
logger.info(f"ALPACA_BASE_URL: {ALPACA_BASE_URL}")
logger.info(f"ALPACA key present: {bool(ALPACA_KEY)}")
logger.info(f"ALPACA secret present: {bool(ALPACA_SECRET)}")
logger.info(f"Polygon key present: {bool(POLYGON_KEY)}")
logger.info(f"Web app URL: {WEBAPP_URL}")
logger.info(f"Analytics webhook: {ANALYTICS_WEBHOOK}")
logger.info("=========================")

# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #

alpaca: tradeapi.REST
polygon: Optional[PolygonClient] = None

try:
    alpaca = tradeapi.REST(
        key_id=ALPACA_KEY,
        secret_key=ALPACA_SECRET,
        base_url=ALPACA_BASE_URL,
        api_version="v2",
    )
    acct = alpaca.get_account()
    logger.info(
        "Alpaca auth OK at startup: status=%s, equity=%s, buying_power=%s",
        acct.status,
        acct.equity,
        acct.buying_power,
    )
except Exception as e:
    logger.error("Alpaca auth FAILED at startup: %s", e)

if POLYGON_KEY:
    try:
        polygon = PolygonClient(POLYGON_KEY)
        logger.info("Polygon client initialized")
    except Exception as e:
        logger.error("Failed to initialize Polygon client: %s", e)
        polygon = None
else:
    logger.info("No POLYGON_API_KEY set; chart_price will be limited")

# --------------------------------------------------------------------------- #
# Universe / symbols
# --------------------------------------------------------------------------- #


def load_allowed_symbols() -> set:
    try:
        with open(ALLOWED_SYMBOLS_FILE, "r") as f:
            return {line.strip().upper() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(
            "Allowed symbols file %s not found; all symbols allowed", ALLOWED_SYMBOLS_FILE
        )
        return set()


ALLOWED_SYMBOLS = load_allowed_symbols()


def validate_symbol(symbol: str) -> bool:
    if not ALLOWED_SYMBOLS:
        return True
    return symbol.upper() in ALLOWED_SYMBOLS


# --------------------------------------------------------------------------- #
# Webhook helper
# --------------------------------------------------------------------------- #


def send_to_webapp(
    analytics_type: str,
    data: dict,
    chart_base64: Optional[str] = None,
) -> None:
    """Send analytics results to your Vercel app."""
    payload = {
        "type": analytics_type,
        "data": data,
        "chart_data": chart_base64,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        resp = requests.post(ANALYTICS_WEBHOOK, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("✓ Sent %s to web app", analytics_type)
        else:
            logger.warning(
                "Failed to send %s to web app: %s %s",
                analytics_type,
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logger.error("Error sending analytics to web app: %s", e)


# --------------------------------------------------------------------------- #
# MCP Server setup
# --------------------------------------------------------------------------- #

app = Server("claude-trading-mcp")


@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="get_quote",
            description="Get real-time quote from Alpaca",
            inputSchema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_account",
            description="Get account information",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_positions",
            description="Get current positions",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="place_order",
            description="Place market order with risk checks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "qty": {"type": "integer"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "time_in_force": {
                        "type": "string",
                        "enum": ["day", "gtc"],
                        "default": "day",
                    },
                },
                "required": ["symbol", "qty", "side"],
            },
        ),
        Tool(
            name="close_position",
            description="Close entire position in a given symbol.",
            inputSchema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        ),
        Tool(
            name="analyze_portfolio",
            description="Analyze portfolio and send charts/summary to the web app.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="chart_price",
            description="Create price chart with indicators and send it to the web app.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 30},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["symbol"],
            },
        ),
    ]


ContentType = Union[TextContent, ImageContent]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[ContentType]:
    try:
        if name == "get_quote":
            return await get_quote(arguments["symbol"])
        if name == "get_account":
            return await get_account()
        if name == "get_positions":
            return await get_positions()
        if name == "place_order":
            return await place_order(
                arguments["symbol"],
                arguments["qty"],
                arguments["side"],
                arguments.get("time_in_force", "day"),
            )
        if name == "close_position":
            return await close_position(arguments["symbol"])
        if name == "analyze_portfolio":
            return await analyze_portfolio()
        if name == "chart_price":
            return await chart_price(
                arguments["symbol"],
                arguments.get("days", 30),
                arguments.get("indicators", []),
            )

        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=f"Error running tool {name}: {e}")]


# --------------------------------------------------------------------------- #
# Trading tools
# --------------------------------------------------------------------------- #


async def get_quote(symbol: str) -> List[TextContent]:
    symbol = symbol.upper()
    try:
        snap = alpaca.get_snapshot(symbol)
        price = float(snap.latest_trade.p) if getattr(snap, "latest_trade", None) else None
        data = {"symbol": symbol, "price": price}
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting quote: {e}")]


async def get_account() -> List[TextContent]:
    try:
        acct = alpaca.get_account()
        data = {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": acct.status,
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting account: {e}")]


async def get_positions() -> List[TextContent]:
    try:
        positions = alpaca.list_positions()
        data = [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting positions: {e}")]


async def place_order(
    symbol: str, qty: int, side: str, time_in_force: str
) -> List[TextContent]:
    symbol = symbol.upper()

    if not validate_symbol(symbol):
        return [
            TextContent(
                type="text", text=f"ERROR: {symbol} not in allowed universe (risk check failed)"
            )
        ]

    if qty > MAX_POSITION_SIZE:
        return [
            TextContent(
                type="text",
                text=f"ERROR: Quantity {qty} exceeds max position size {MAX_POSITION_SIZE}",
            )
        ]

    try:
        snap = alpaca.get_snapshot(symbol)
        price = float(snap.latest_trade.p) if getattr(snap, "latest_trade", None) else None
        if not price:
            return [TextContent(type="text", text="ERROR: Could not get latest price")]

        notional = qty * price
        if side.lower() == "buy" and notional > MAX_POSITION_VALUE:
            return [
                TextContent(
                    type="text",
                    text=f"ERROR: Order value ${notional:,.2f} exceeds limit ${MAX_POSITION_VALUE:,.2f}",
                )
            ]

        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type="market",
            time_in_force=time_in_force.lower(),
        )

        result = {
            "status": "submitted",
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": int(order.qty),
            "side": order.side,
            "created_at": str(order.created_at),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"ERROR placing order: {e}")]


async def close_position(symbol: str) -> List[TextContent]:
    symbol = symbol.upper()
    try:
        alpaca.close_position(symbol)
        return [TextContent(type="text", text=f"Closed {symbol} position")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error closing position: {e}")]


# --------------------------------------------------------------------------- #
# Analytics tools
# --------------------------------------------------------------------------- #


async def analyze_portfolio() -> List[ContentType]:
    """Analyze portfolio - sends chart to your web app."""
    try:
        positions = alpaca.list_positions()
        if not positions:
            return [TextContent(type="text", text="No positions to analyze")]

        symbols = [p.symbol for p in positions]
        values = [float(p.market_value) for p in positions]
        pnl = [float(p.unrealized_pl) for p in positions]
        pnl_pct = [float(p.unrealized_plpc) * 100 for p in positions]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Position sizes
        ax1.barh(symbols, values)
        ax1.set_xlabel("Market Value ($)")
        ax1.set_title("Position Sizes")
        ax1.grid(axis="x", alpha=0.3)

        # P&L
        colors_pnl = ["#4caf50" if x > 0 else "#f44336" for x in pnl]
        ax2.barh(symbols, pnl, color=colors_pnl)
        ax2.set_xlabel("Unrealized P&L ($)")
        ax2.set_title("Profit/Loss")
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.grid(axis="x", alpha=0.3)

        # Allocation pie
        ax3.pie(values, labels=symbols, autopct="%1.1f%%", startangle=90)
        ax3.set_title("Portfolio Allocation")

        # Performance %
        colors_ret = ["#4caf50" if x > 0 else "#f44336" for x in pnl_pct]
        ax4.barh(symbols, pnl_pct, color=colors_ret)
        ax4.set_xlabel("Return (%)")
        ax4.set_title("Performance %")
        ax4.axvline(x=0, color="black", linewidth=0.5)
        ax4.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()

        total_value = sum(values)
        total_pnl = sum(pnl)
        invested = total_value - total_pnl
        total_pnl_pct = (total_pnl / invested * 100) if invested > 0 else 0.0

        data = {
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "position_count": len(positions),
            "winners": len([x for x in pnl if x > 0]),
            "losers": len([x for x in pnl if x < 0]),
        }

        send_to_webapp("portfolio_analysis", data, img_base64)

        summary = f"""Portfolio Analysis:

Total Value: ${total_value:,.2f}
Total P&L:   ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)
Positions:   {len(positions)}
Winners:     {data['winners']} | Losers: {data['losers']}

✓ Chart & analytics sent to your web app at {WEBAPP_URL}
"""

        return [
            TextContent(type="text", text=summary),
            ImageContent(type="image", data=img_base64, mimeType="image/png"),
        ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error analyzing portfolio: {e}")]


async def chart_price(
    symbol: str, days: int, indicators: List[str]
) -> List[ContentType]:
    """Create price chart - sends to your web app."""
    symbol = symbol.upper()

    if not polygon:
        return [
            TextContent(
                type="text",
                text="Polygon client not available (no POLYGON_API_KEY set on server).",
            )
        ]

    try:
        end = datetime.now()
        start = end - timedelta(days=days)

        aggs = polygon.get_aggs(
            symbol,
            1,
            "day",
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            limit=days,
        )

        rows = [
            {
                "date": datetime.fromtimestamp(bar.timestamp / 1000),
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in aggs
        ]
        if not rows:
            return [TextContent(type="text", text="No price data returned from Polygon.")]

        df = pd.DataFrame(rows)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        ax1.plot(df["date"], df["close"], linewidth=2, label="Close")

        if "sma20" in indicators:
            df["sma20"] = df["close"].rolling(20).mean()
            ax1.plot(df["date"], df["sma20"], "--", alpha=0.7, label="SMA 20")

        if "sma50" in indicators:
            df["sma50"] = df["close"].rolling(50).mean()
            ax1.plot(df["date"], df["sma50"], "--", alpha=0.7, label="SMA 50")

        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{symbol} - {days} Day Chart")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.bar(df["date"], df["volume"], alpha=0.5)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date")
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()

        current = df["close"].iloc[-1]
        change = current - df["close"].iloc[0]
        change_pct = (change / df["close"].iloc[0]) * 100

        data = {
            "symbol": symbol,
            "current_price": current,
            "change": change,
            "change_pct": change_pct,
            "days": days,
        }

        send_to_webapp("price_chart", data, img_base64)

        summary = f"""{symbol} Chart ({days} days):

Current: ${current:.2f}
Change:  ${change:+.2f} ({change_pct:+.2f}%)

✓ Chart sent to your web app at {WEBAPP_URL}
"""

        return [
            TextContent(type="text", text=summary),
            ImageContent(type="image", data=img_base64, mimeType="image/png"),
        ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating chart: {e}")]


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


async def main_stdio() -> None:
    """StdIO mode – used by Claude Desktop on your PC."""
    from mcp.server.stdio import stdio_server

    logger.info("Starting Claude Trading MCP Server (stdio)...")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def run_http_server() -> None:
    """HTTP mode – used by Render / Railway / Claude web & mobile."""
    port = int(os.getenv("PORT", "8000"))

    logger.info("Starting Claude Trading MCP Server (HTTP)...")
    logger.info("Listening on 0.0.0.0:%s", port)

    manager = StreamableHTTPSessionManager(app, stateless=False)

    uvicorn.run(
        manager.as_asgi_app(),
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        asyncio.run(run_http_server())
    else:
        asyncio.run(main_stdio())

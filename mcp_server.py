#!/usr/bin/env python3
"""
MCP Server for Claude Mobile Trading
Exposes Polygon data + Alpaca trading to Claude via MCP protocol
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging

# MCP SDK
from mcp.server import Server
from mcp.types import Tool, TextContent

# API clients
import alpaca_trade_api as tradeapi
from polygon import RESTClient as PolygonClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIs
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')
polygon = PolygonClient(POLYGON_KEY)

# Safety limits
MAX_POSITION_SIZE = 1000  # max shares per trade
MAX_POSITION_VALUE = 10000  # max $ per position
ALLOWED_SYMBOLS_FILE = "data/universe_liquid.txt"  # only trade liquid stocks

app = Server("claude-trading-mcp")

def load_allowed_symbols() -> set:
    """Load universe of tradeable symbols"""
    try:
        with open(ALLOWED_SYMBOLS_FILE, 'r') as f:
            return set(line.strip().upper() for line in f if line.strip())
    except FileNotFoundError:
        logger.warning(f"Universe file not found: {ALLOWED_SYMBOLS_FILE}")
        return set()

ALLOWED_SYMBOLS = load_allowed_symbols()

def validate_symbol(symbol: str) -> bool:
    """Check if symbol is in allowed universe"""
    return symbol.upper() in ALLOWED_SYMBOLS

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Define available tools for Claude"""
    return [
        Tool(
            name="get_quote",
            description="Get real-time quote for a symbol from Polygon",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_bars",
            description="Get historical price bars (OHLCV) from Polygon",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "timespan": {"type": "string", "enum": ["day", "hour", "minute"], "default": "day"},
                    "limit": {"type": "integer", "default": 100, "description": "Number of bars"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_account",
            description="Get Alpaca account information (buying power, equity, positions)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_positions",
            description="Get current open positions in Alpaca",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="place_order",
            description="Place a market order in Alpaca (BUY or SELL)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "qty": {"type": "integer", "description": "Number of shares"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "time_in_force": {"type": "string", "enum": ["day", "gtc"], "default": "day"}
                },
                "required": ["symbol", "qty", "side"]
            }
        ),
        Tool(
            name="close_position",
            description="Close an entire position (market sell all shares)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker to close"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_orders",
            description="Get recent orders (today's orders by default)",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["open", "closed", "all"], "default": "all"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool execution"""
    try:
        if name == "get_quote":
            return await get_quote(arguments["symbol"])
        elif name == "get_bars":
            return await get_bars(
                arguments["symbol"],
                arguments.get("timespan", "day"),
                arguments.get("limit", 100)
            )
        elif name == "get_account":
            return await get_account()
        elif name == "get_positions":
            return await get_positions()
        elif name == "place_order":
            return await place_order(
                arguments["symbol"],
                arguments["qty"],
                arguments["side"],
                arguments.get("time_in_force", "day")
            )
        elif name == "close_position":
            return await close_position(arguments["symbol"])
        elif name == "get_orders":
            return await get_orders(arguments.get("status", "all"))
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_quote(symbol: str) -> List[TextContent]:
    """Fetch real-time quote from Polygon"""
    symbol = symbol.upper()
    
    try:
        quote = polygon.get_last_quote(symbol)
        data = {
            "symbol": symbol,
            "bid": quote.bid_price,
            "ask": quote.ask_price,
            "last": (quote.bid_price + quote.ask_price) / 2,
            "timestamp": quote.sip_timestamp
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching quote: {e}")]

async def get_bars(symbol: str, timespan: str, limit: int) -> List[TextContent]:
    """Fetch historical bars from Polygon"""
    symbol = symbol.upper()
    
    try:
        end = datetime.now()
        start = end - timedelta(days=limit if timespan == "day" else 7)
        
        bars = polygon.get_aggs(
            symbol,
            1,
            timespan,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            limit=limit
        )
        
        data = [{
            "time": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        } for bar in bars]
        
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching bars: {e}")]

async def get_account() -> List[TextContent]:
    """Get Alpaca account info"""
    try:
        account = alpaca.get_account()
        data = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching account: {e}")]

async def get_positions() -> List[TextContent]:
    """Get current positions"""
    try:
        positions = alpaca.list_positions()
        data = [{
            "symbol": p.symbol,
            "qty": int(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc)
        } for p in positions]
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching positions: {e}")]

async def place_order(symbol: str, qty: int, side: str, time_in_force: str) -> List[TextContent]:
    """Place market order with safety checks"""
    symbol = symbol.upper()
    
    # Safety checks
    if not validate_symbol(symbol):
        return [TextContent(type="text", text=f"ERROR: {symbol} not in allowed universe")]
    
    if qty > MAX_POSITION_SIZE:
        return [TextContent(type="text", text=f"ERROR: Quantity {qty} exceeds max {MAX_POSITION_SIZE}")]
    
    try:
        # Get current price estimate
        quote = polygon.get_last_quote(symbol)
        est_price = (quote.bid_price + quote.ask_price) / 2
        est_value = qty * est_price
        
        if side.lower() == "buy" and est_value > MAX_POSITION_VALUE:
            return [TextContent(type="text", text=f"ERROR: Order value ${est_value:.2f} exceeds max ${MAX_POSITION_VALUE}")]
        
        # Place order
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type='market',
            time_in_force=time_in_force.lower()
        )
        
        result = {
            "status": "submitted",
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": int(order.qty),
            "side": order.side,
            "type": order.type,
            "time_in_force": order.time_in_force,
            "submitted_at": str(order.submitted_at)
        }
        
        logger.info(f"Order placed: {result}")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return [TextContent(type="text", text=f"ERROR placing order: {e}")]

async def close_position(symbol: str) -> List[TextContent]:
    """Close entire position"""
    symbol = symbol.upper()
    
    try:
        # Check if position exists
        try:
            position = alpaca.get_position(symbol)
        except Exception:
            return [TextContent(type="text", text=f"No open position for {symbol}")]
        
        # Close position
        alpaca.close_position(symbol)
        
        result = {
            "status": "closed",
            "symbol": symbol,
            "qty_closed": int(position.qty)
        }
        
        logger.info(f"Position closed: {result}")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"ERROR closing position: {e}")]

async def get_orders(status: str) -> List[TextContent]:
    """Get recent orders"""
    try:
        orders = alpaca.list_orders(status=status, limit=50)
        data = [{
            "id": o.id,
            "symbol": o.symbol,
            "qty": int(o.qty),
            "side": o.side,
            "type": o.type,
            "status": o.status,
            "filled_qty": int(o.filled_qty) if o.filled_qty else 0,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "submitted_at": str(o.submitted_at)
        } for o in orders]
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching orders: {e}")]

async def main():
    """Run MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Claude Trading MCP Server...")
    logger.info(f"Loaded {len(ALLOWED_SYMBOLS)} symbols from universe")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
MCP Server with Webhook Support
Sends analytics results to your Vercel app
"""

import os
import json
import asyncio
import base64
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
from io import BytesIO

# MCP SDK
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent

# API clients
import alpaca_trade_api as tradeapi
from polygon import RESTClient as PolygonClient

# Analytics
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Config
ALPACA_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

# YOUR VERCEL APP URL
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://iw-positions-app.vercel.app")
ANALYTICS_WEBHOOK = f"{WEBAPP_URL}/api/analytics"

alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')
polygon = PolygonClient(POLYGON_KEY)

MAX_POSITION_SIZE = 1000
MAX_POSITION_VALUE = 10000
ALLOWED_SYMBOLS_FILE = "data/universe_liquid.txt"

app = Server("claude-trading-mcp")

def load_allowed_symbols() -> set:
    try:
        with open(ALLOWED_SYMBOLS_FILE, 'r') as f:
            return set(line.strip().upper() for line in f if line.strip())
    except FileNotFoundError:
        return set()

ALLOWED_SYMBOLS = load_allowed_symbols()

def validate_symbol(symbol: str) -> bool:
    return symbol.upper() in ALLOWED_SYMBOLS if ALLOWED_SYMBOLS else True

def send_to_webapp(analytics_type: str, data: dict, chart_base64: Optional[str] = None):
    """Send analytics results to your Vercel app"""
    try:
        payload = {
            "type": analytics_type,
            "data": data,
            "chart_data": chart_base64,
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(ANALYTICS_WEBHOOK, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✓ Sent {analytics_type} to web app")
        else:
            logger.warning(f"Failed to send to web app: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Failed to send to web app: {e}")

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="get_quote",
            description="Get real-time quote from Alpaca",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_account",
            description="Get account information",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_positions",
            description="Get current positions",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="place_order",
            description="Place market order",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "qty": {"type": "integer"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "time_in_force": {"type": "string", "enum": ["day", "gtc"], "default": "day"}
                },
                "required": ["symbol", "qty", "side"]
            }
        ),
        Tool(
            name="close_position",
            description="Close entire position",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="analyze_portfolio",
            description="Analyze portfolio with charts (displayed in your app!)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="chart_price",
            description="Create price chart with indicators (displayed in your app!)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 30},
                    "indicators": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["symbol"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
    try:
        if name == "get_quote":
            return await get_quote(arguments["symbol"])
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
        elif name == "analyze_portfolio":
            return await analyze_portfolio()
        elif name == "chart_price":
            return await chart_price(
                arguments["symbol"],
                arguments.get("days", 30),
                arguments.get("indicators", [])
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# Trading functions (simplified)
async def get_quote(symbol: str) -> List[TextContent]:
    symbol = symbol.upper()
    try:
        snapshot = alpaca.get_snapshot(symbol)
        price = float(snapshot.latest_trade.p) if hasattr(snapshot, 'latest_trade') else None
        
        data = {"symbol": symbol, "price": price}
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def get_account() -> List[TextContent]:
    try:
        account = alpaca.get_account()
        data = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power)
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def get_positions() -> List[TextContent]:
    try:
        positions = alpaca.list_positions()
        data = [{
            "symbol": p.symbol,
            "qty": int(p.qty),
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl)
        } for p in positions]
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def place_order(symbol: str, qty: int, side: str, time_in_force: str) -> List[TextContent]:
    symbol = symbol.upper()
    
    if not validate_symbol(symbol):
        return [TextContent(type="text", text=f"ERROR: {symbol} not in allowed universe")]
    
    if qty > MAX_POSITION_SIZE:
        return [TextContent(type="text", text=f"ERROR: Quantity exceeds limit")]
    
    try:
        snapshot = alpaca.get_snapshot(symbol)
        price = float(snapshot.latest_trade.p) if hasattr(snapshot, 'latest_trade') else None
        
        if not price:
            return [TextContent(type="text", text="ERROR: Could not get price")]
        
        if side.lower() == "buy" and qty * price > MAX_POSITION_VALUE:
            return [TextContent(type="text", text="ERROR: Order value too high")]
        
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
            "side": order.side
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"ERROR: {e}")]

async def close_position(symbol: str) -> List[TextContent]:
    symbol = symbol.upper()
    try:
        alpaca.close_position(symbol)
        return [TextContent(type="text", text=f"Closed {symbol} position")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

# Analytics functions with web app integration
async def analyze_portfolio() -> List[TextContent | ImageContent]:
    """Analyze portfolio - sends chart to your web app!"""
    try:
        positions = alpaca.list_positions()
        
        if not positions:
            return [TextContent(type="text", text="No positions to analyze")]
        
        # Create chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        symbols = [p.symbol for p in positions]
        values = [float(p.market_value) for p in positions]
        pnl = [float(p.unrealized_pl) for p in positions]
        pnl_pct = [float(p.unrealized_plpc) * 100 for p in positions]
        
        # Position sizes
        ax1.barh(symbols, values, color='steelblue')
        ax1.set_xlabel('Market Value ($)')
        ax1.set_title('Position Sizes', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # P&L
        colors = ['#4caf50' if x > 0 else '#f44336' for x in pnl]
        ax2.barh(symbols, pnl, color=colors)
        ax2.set_xlabel('Unrealized P&L ($)')
        ax2.set_title('Profit/Loss', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Allocation pie
        ax3.pie(values, labels=symbols, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
        
        # Performance %
        colors = ['#4caf50' if x > 0 else '#f44336' for x in pnl_pct]
        ax4.barh(symbols, pnl_pct, color=colors)
        ax4.set_xlabel('Return (%)')
        ax4.set_title('Performance %', fontsize=14, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Calculate stats
        total_value = sum(values)
        total_pnl = sum(pnl)
        total_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0
        
        data = {
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "position_count": len(positions),
            "winners": len([p for p in pnl if p > 0]),
            "losers": len([p for p in pnl if p < 0])
        }
        
        # Send to your web app!
        send_to_webapp("portfolio_analysis", data, img_base64)
        
        summary = f"""Portfolio Analysis:

Total Value: ${total_value:,.2f}
Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)
Positions: {len(positions)}
Winners: {data['winners']} | Losers: {data['losers']}

✓ Chart sent to your web app at {WEBAPP_URL}
"""
        
        return [
            TextContent(type="text", text=summary),
            ImageContent(type="image", data=img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def chart_price(symbol: str, days: int, indicators: List[str]) -> List[TextContent | ImageContent]:
    """Create price chart - sends to your web app!"""
    symbol = symbol.upper()
    
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        bars = polygon.get_aggs(
            symbol,
            1,
            "day",
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            limit=days
        )
        
        df = pd.DataFrame([{
            'date': datetime.fromtimestamp(bar.timestamp / 1000),
            'close': bar.close,
            'volume': bar.volume
        } for bar in bars])
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
        
        ax1.plot(df['date'], df['close'], linewidth=2, label='Close', color='steelblue')
        
        if 'sma20' in indicators:
            df['sma20'] = df['close'].rolling(20).mean()
            ax1.plot(df['date'], df['sma20'], label='SMA 20', linestyle='--', alpha=0.7)
        
        if 'sma50' in indicators:
            df['sma50'] = df['close'].rolling(50).mean()
            ax1.plot(df['date'], df['sma50'], label='SMA 50', linestyle='--', alpha=0.7)
        
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{symbol} - {days} Day Chart', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.bar(df['date'], df['volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        current = df['close'].iloc[-1]
        change = current - df['close'].iloc[0]
        change_pct = (change / df['close'].iloc[0]) * 100
        
        data = {
            "symbol": symbol,
            "current_price": current,
            "change": change,
            "change_pct": change_pct,
            "days": days
        }
        
        # Send to your web app!
        send_to_webapp("price_chart", data, img_base64)
        
        summary = f"""{symbol} Chart ({days} days):

Current: ${current:.2f}
Change: ${change:+.2f} ({change_pct:+.2f}%)

✓ Chart sent to your web app at {WEBAPP_URL}
"""
        
        return [
            TextContent(type="text", text=summary),
            ImageContent(type="image", data=img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def main():
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Claude Trading MCP Server...")
    logger.info(f"Web app URL: {WEBAPP_URL}")
    logger.info(f"Analytics webhook: {ANALYTICS_WEBHOOK}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())

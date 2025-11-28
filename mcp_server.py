#!/usr/bin/env python3
"""
Enhanced MCP Server with Analytics & Graph Generation
Supports natural language trading + analytics on Claude mobile
"""

import os
import json
import asyncio
import base64
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

# Analytics & Visualization
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIs
ALPACA_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')
polygon = PolygonClient(POLYGON_KEY)

# Safety limits
MAX_POSITION_SIZE = 1000
MAX_POSITION_VALUE = 10000
ALLOWED_SYMBOLS_FILE = "data/universe_liquid.txt"

app = Server("claude-trading-analytics-mcp")

def load_allowed_symbols() -> set:
    try:
        with open(ALLOWED_SYMBOLS_FILE, 'r') as f:
            return set(line.strip().upper() for line in f if line.strip())
    except FileNotFoundError:
        logger.warning(f"Universe file not found: {ALLOWED_SYMBOLS_FILE}")
        return set()

ALLOWED_SYMBOLS = load_allowed_symbols()

def validate_symbol(symbol: str) -> bool:
    return symbol.upper() in ALLOWED_SYMBOLS if ALLOWED_SYMBOLS else True

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Define available tools including analytics"""
    return [
        # Trading tools
        Tool(
            name="get_quote",
            description="Get real-time quote from Alpaca (live trading data)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_account",
            description="Get Alpaca account information",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_positions",
            description="Get current open positions",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="place_order",
            description="Place a market order (BUY or SELL)",
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
        
        # Analytics tools
        Tool(
            name="analyze_portfolio",
            description="Analyze portfolio performance with metrics and chart",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="chart_price_history",
            description="Create price chart with technical indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "days": {"type": "integer", "default": 30, "description": "Number of days"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Technical indicators: sma20, sma50, volume"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="compare_symbols",
            description="Compare multiple symbols with normalized price chart",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tickers to compare"
                    },
                    "days": {"type": "integer", "default": 30}
                },
                "required": ["symbols"]
            }
        ),
        Tool(
            name="position_breakdown",
            description="Create pie chart of portfolio allocation",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="performance_report",
            description="Generate detailed performance report with metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "Lookback period"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
    """Handle tool execution"""
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
        elif name == "chart_price_history":
            return await chart_price_history(
                arguments["symbol"],
                arguments.get("days", 30),
                arguments.get("indicators", [])
            )
        elif name == "compare_symbols":
            return await compare_symbols(
                arguments["symbols"],
                arguments.get("days", 30)
            )
        elif name == "position_breakdown":
            return await position_breakdown()
        elif name == "performance_report":
            return await performance_report(arguments.get("days", 7))
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# Trading functions (simplified - same as before)
async def get_quote(symbol: str) -> List[TextContent]:
    """Get real-time quote from Alpaca"""
    symbol = symbol.upper()
    try:
        snapshot = alpaca.get_snapshot(symbol)
        
        latest_trade = None
        if hasattr(snapshot, 'latest_trade') and snapshot.latest_trade:
            latest_trade = {
                "price": float(snapshot.latest_trade.p),
                "size": int(snapshot.latest_trade.s)
            }
        
        latest_quote = None
        if hasattr(snapshot, 'latest_quote') and snapshot.latest_quote:
            latest_quote = {
                "bid": float(snapshot.latest_quote.bp),
                "ask": float(snapshot.latest_quote.ap)
            }
        
        data = {
            "symbol": symbol,
            "latest_trade": latest_trade,
            "latest_quote": latest_quote
        }
        
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def get_account() -> List[TextContent]:
    """Get account info"""
    try:
        account = alpaca.get_account()
        data = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value)
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def get_positions() -> List[TextContent]:
    """Get positions"""
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
        return [TextContent(type="text", text=f"Error: {e}")]

async def place_order(symbol: str, qty: int, side: str, time_in_force: str) -> List[TextContent]:
    """Place order"""
    symbol = symbol.upper()
    
    if not validate_symbol(symbol):
        return [TextContent(type="text", text=f"ERROR: {symbol} not in allowed universe")]
    
    if qty > MAX_POSITION_SIZE:
        return [TextContent(type="text", text=f"ERROR: Quantity exceeds limit")]
    
    try:
        snapshot = alpaca.get_snapshot(symbol)
        est_price = float(snapshot.latest_trade.p) if hasattr(snapshot, 'latest_trade') else None
        
        if not est_price:
            return [TextContent(type="text", text=f"ERROR: Could not get price")]
        
        if side.lower() == "buy" and qty * est_price > MAX_POSITION_VALUE:
            return [TextContent(type="text", text=f"ERROR: Order value too high")]
        
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
    """Close position"""
    symbol = symbol.upper()
    try:
        alpaca.close_position(symbol)
        return [TextContent(type="text", text=f"Closed {symbol} position")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

# Analytics functions with graphs
async def analyze_portfolio() -> List[TextContent | ImageContent]:
    """Analyze portfolio with chart"""
    try:
        positions = alpaca.list_positions()
        
        if not positions:
            return [TextContent(type="text", text="No positions to analyze")]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Position sizes
        symbols = [p.symbol for p in positions]
        values = [float(p.market_value) for p in positions]
        
        ax1.barh(symbols, values, color='steelblue')
        ax1.set_xlabel('Market Value ($)')
        ax1.set_title('Position Sizes')
        ax1.grid(axis='x', alpha=0.3)
        
        # P&L
        pnl = [float(p.unrealized_pl) for p in positions]
        colors = ['green' if x > 0 else 'red' for x in pnl]
        ax2.barh(symbols, pnl, color=colors)
        ax2.set_xlabel('Unrealized P&L ($)')
        ax2.set_title('Profit/Loss by Position')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Allocation pie
        ax3.pie(values, labels=symbols, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Portfolio Allocation')
        
        # Performance %
        pnl_pct = [float(p.unrealized_plpc) * 100 for p in positions]
        colors = ['green' if x > 0 else 'red' for x in pnl_pct]
        ax4.barh(symbols, pnl_pct, color=colors)
        ax4.set_xlabel('Return (%)')
        ax4.set_title('Performance %')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Summary stats
        total_value = sum(values)
        total_pnl = sum(pnl)
        total_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0
        
        summary = f"""Portfolio Analysis:
        
Total Value: ${total_value:,.2f}
Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)
Positions: {len(positions)}

Winners: {len([p for p in pnl if p > 0])}
Losers: {len([p for p in pnl if p < 0])}
"""
        
        return [
            TextContent(type="text", text=summary),
            ImageContent(type="image", data=img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def chart_price_history(symbol: str, days: int, indicators: List[str]) -> List[TextContent | ImageContent]:
    """Create price chart with indicators"""
    symbol = symbol.upper()
    
    try:
        # Get historical data from Polygon
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
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in bars])
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Price chart
        ax1.plot(df['date'], df['close'], linewidth=2, label='Close', color='steelblue')
        
        # Add indicators
        if 'sma20' in indicators:
            df['sma20'] = df['close'].rolling(20).mean()
            ax1.plot(df['date'], df['sma20'], label='SMA 20', linestyle='--', alpha=0.7)
        
        if 'sma50' in indicators:
            df['sma50'] = df['close'].rolling(50).mean()
            ax1.plot(df['date'], df['sma50'], label='SMA 50', linestyle='--', alpha=0.7)
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{symbol} - {days} Day Price Chart')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Volume
        if 'volume' in indicators or not indicators:
            ax2.bar(df['date'], df['volume'], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Stats
        current_price = df['close'].iloc[-1]
        change = current_price - df['close'].iloc[0]
        change_pct = (change / df['close'].iloc[0]) * 100
        
        stats = f"""{symbol} Chart ({days} days):

Current: ${current_price:.2f}
Change: ${change:+.2f} ({change_pct:+.2f}%)
High: ${df['high'].max():.2f}
Low: ${df['low'].min():.2f}
Avg Volume: {df['volume'].mean():,.0f}
"""
        
        return [
            TextContent(type="text", text=stats),
            ImageContent(type="image", data=img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def compare_symbols(symbols: List[str], days: int) -> List[TextContent | ImageContent]:
    """Compare multiple symbols"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        for symbol in symbols:
            symbol = symbol.upper()
            
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
                'close': bar.close
            } for bar in bars])
            
            # Normalize to 100
            df['normalized'] = (df['close'] / df['close'].iloc[0]) * 100
            
            ax.plot(df['date'], df['normalized'], label=symbol, linewidth=2)
        
        ax.set_ylabel('Normalized Price (Base 100)')
        ax.set_xlabel('Date')
        ax.set_title(f'Symbol Comparison ({days} days)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return [
            TextContent(type="text", text=f"Comparing {', '.join(symbols)} over {days} days"),
            ImageContent(type="image", data=img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def position_breakdown() -> List[TextContent | ImageContent]:
    """Portfolio allocation pie chart"""
    try:
        positions = alpaca.list_positions()
        
        if not positions:
            return [TextContent(type="text", text="No positions")]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        symbols = [p.symbol for p in positions]
        values = [float(p.market_value) for p in positions]
        
        colors = plt.cm.Set3(range(len(symbols)))
        ax.pie(values, labels=symbols, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Portfolio Allocation')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        total = sum(values)
        breakdown = "\n".join([f"{s}: ${v:,.2f} ({v/total*100:.1f}%)" for s, v in zip(symbols, values)])
        
        return [
            TextContent(type="text", text=f"Portfolio Breakdown:\n\n{breakdown}\n\nTotal: ${total:,.2f}"),
            ImageContent(type="image", data=img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def performance_report(days: int) -> List[TextContent]:
    """Detailed performance report"""
    try:
        # Get account history
        end = datetime.now()
        start = end - timedelta(days=days)
        
        portfolio_history = alpaca.get_portfolio_history(
            date_start=start.strftime('%Y-%m-%d'),
            timeframe='1D'
        )
        
        equity_data = portfolio_history.equity
        timestamps = portfolio_history.timestamp
        
        if not equity_data or len(equity_data) < 2:
            return [TextContent(type="text", text="Insufficient data for performance report")]
        
        # Calculate metrics
        start_equity = equity_data[0]
        end_equity = equity_data[-1]
        total_return = ((end_equity - start_equity) / start_equity) * 100
        
        # Daily returns
        returns = [(equity_data[i] - equity_data[i-1]) / equity_data[i-1] * 100 
                   for i in range(1, len(equity_data))]
        
        avg_daily_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe = (avg_daily_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        max_equity = max(equity_data)
        min_equity = min(equity_data)
        drawdown = ((max_equity - min_equity) / max_equity) * 100
        
        report = f"""Performance Report ({days} days):

Total Return: {total_return:+.2f}%
Starting Equity: ${start_equity:,.2f}
Ending Equity: ${end_equity:,.2f}

Risk Metrics:
- Avg Daily Return: {avg_daily_return:+.2f}%
- Volatility: {volatility:.2f}%
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {drawdown:.2f}%

High: ${max_equity:,.2f}
Low: ${min_equity:,.2f}
"""
        
        return [TextContent(type="text", text=report)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def main():
    """Run MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Claude Trading & Analytics MCP Server...")
    logger.info(f"Loaded {len(ALLOWED_SYMBOLS)} symbols from universe")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
MCP Intent Runner - GitHub Actions Compatible
Processes trading intents from Claude mobile or GitHub webhook
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import alpaca_trade_api as tradeapi
from polygon import RESTClient as PolygonClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize APIs - matching your GitHub secret names
ALPACA_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version='v2')
polygon = PolygonClient(POLYGON_KEY)

# Safety configuration
MAX_POSITION_SIZE = int(os.getenv("MAX_POSITION_SIZE", "1000"))
MAX_POSITION_VALUE = float(os.getenv("MAX_POSITION_VALUE", "10000"))
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "data/universe_liquid.txt")

def load_universe() -> set:
    """Load allowed trading universe"""
    try:
        with open(UNIVERSE_FILE, 'r') as f:
            return set(line.strip().upper() for line in f if line.strip())
    except FileNotFoundError:
        logger.warning(f"Universe file not found: {UNIVERSE_FILE}")
        return set()

ALLOWED_SYMBOLS = load_universe()
logger.info(f"Loaded {len(ALLOWED_SYMBOLS)} symbols from universe")

class IntentProcessor:
    """Process trading intents with safety checks"""
    
    def __init__(self):
        self.alpaca = alpaca
        self.polygon = polygon
        self.allowed_symbols = ALLOWED_SYMBOLS
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is tradeable"""
        symbol = symbol.upper()
        if not self.allowed_symbols:
            logger.warning("No universe loaded - allowing all symbols")
            return True
        return symbol in self.allowed_symbols
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Alpaca (real-time trading data)"""
        try:
            # Use Alpaca for live trading data
            snapshot = self.alpaca.get_snapshot(symbol)
            
            # Try latest trade first
            if hasattr(snapshot, 'latest_trade') and snapshot.latest_trade:
                price = float(snapshot.latest_trade.p)
                logger.info(f"Alpaca live price for {symbol}: ${price:.2f}")
                return price
            
            # Fallback to latest quote
            if hasattr(snapshot, 'latest_quote') and snapshot.latest_quote:
                bid = float(snapshot.latest_quote.bp)
                ask = float(snapshot.latest_quote.ap)
                price = (bid + ask) / 2
                logger.info(f"Alpaca quote price for {symbol}: ${price:.2f}")
                return price
            
            # Last resort: get latest bar
            bars = self.alpaca.get_bars(symbol, '1Min', limit=1).df
            if not bars.empty:
                price = float(bars['close'].iloc[-1])
                logger.info(f"Alpaca bar price for {symbol}: ${price:.2f}")
                return price
                
            logger.error(f"No price data available for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def process_buy_intent(self, intent: Dict) -> Dict:
        """
        Process BUY intent
        Expected format:
        {
            "action": "buy",
            "symbol": "AAPL",
            "quantity": 10,
            "time_in_force": "day"  # optional
        }
        """
        symbol = intent.get("symbol", "").upper()
        quantity = int(intent.get("quantity", 0))
        tif = intent.get("time_in_force", "day").lower()
        
        # Validation
        if not symbol:
            return {"status": "error", "message": "Missing symbol"}
        
        if quantity <= 0:
            return {"status": "error", "message": "Invalid quantity"}
        
        if not self.validate_symbol(symbol):
            return {"status": "error", "message": f"{symbol} not in allowed universe"}
        
        if quantity > MAX_POSITION_SIZE:
            return {"status": "error", "message": f"Quantity {quantity} exceeds max {MAX_POSITION_SIZE}"}
        
        # Check position value
        price = self.get_current_price(symbol)
        if not price:
            return {"status": "error", "message": f"Could not get price for {symbol}"}
        
        est_value = quantity * price
        if est_value > MAX_POSITION_VALUE:
            return {
                "status": "error",
                "message": f"Order value ${est_value:.2f} exceeds max ${MAX_POSITION_VALUE}"
            }
        
        # Place order
        try:
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force=tif
            )
            
            result = {
                "status": "success",
                "action": "buy",
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": int(order.qty),
                "estimated_price": price,
                "estimated_value": est_value,
                "time_in_force": order.time_in_force,
                "submitted_at": str(order.submitted_at)
            }
            
            logger.info(f"BUY order placed: {symbol} x {quantity} @ ~${price:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place BUY order: {e}")
            return {"status": "error", "message": str(e)}
    
    def process_sell_intent(self, intent: Dict) -> Dict:
        """
        Process SELL intent
        Expected format:
        {
            "action": "sell",
            "symbol": "AAPL",
            "quantity": 10  # optional - if omitted, closes entire position
        }
        """
        symbol = intent.get("symbol", "").upper()
        quantity = intent.get("quantity")
        
        if not symbol:
            return {"status": "error", "message": "Missing symbol"}
        
        try:
            # Check if position exists
            try:
                position = self.alpaca.get_position(symbol)
            except Exception:
                return {"status": "error", "message": f"No position for {symbol}"}
            
            position_qty = int(position.qty)
            
            # If no quantity specified, close entire position
            if quantity is None:
                self.alpaca.close_position(symbol)
                logger.info(f"CLOSED position: {symbol} ({position_qty} shares)")
                return {
                    "status": "success",
                    "action": "close",
                    "symbol": symbol,
                    "quantity": position_qty,
                    "avg_entry": float(position.avg_entry_price)
                }
            
            # Otherwise, sell specified quantity
            quantity = int(quantity)
            if quantity > position_qty:
                return {
                    "status": "error",
                    "message": f"Cannot sell {quantity} shares - only holding {position_qty}"
                }
            
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            result = {
                "status": "success",
                "action": "sell",
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": int(order.qty),
                "submitted_at": str(order.submitted_at)
            }
            
            logger.info(f"SELL order placed: {symbol} x {quantity}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process SELL: {e}")
            return {"status": "error", "message": str(e)}
    
    def process_close_intent(self, intent: Dict) -> Dict:
        """
        Process CLOSE intent (close entire position)
        Expected format:
        {
            "action": "close",
            "symbol": "AAPL"
        }
        """
        return self.process_sell_intent({"action": "sell", "symbol": intent.get("symbol")})
    
    def process_query_intent(self, intent: Dict) -> Dict:
        """
        Process QUERY intents (get account, positions, orders)
        Expected format:
        {
            "action": "query",
            "type": "account" | "positions" | "orders"
        }
        """
        query_type = intent.get("type", "").lower()
        
        try:
            if query_type == "account":
                account = self.alpaca.get_account()
                return {
                    "status": "success",
                    "action": "query",
                    "type": "account",
                    "data": {
                        "equity": float(account.equity),
                        "cash": float(account.cash),
                        "buying_power": float(account.buying_power),
                        "portfolio_value": float(account.portfolio_value)
                    }
                }
            
            elif query_type == "positions":
                positions = self.alpaca.list_positions()
                return {
                    "status": "success",
                    "action": "query",
                    "type": "positions",
                    "data": [{
                        "symbol": p.symbol,
                        "qty": int(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": float(p.unrealized_plpc)
                    } for p in positions]
                }
            
            elif query_type == "orders":
                orders = self.alpaca.list_orders(status='all', limit=20)
                return {
                    "status": "success",
                    "action": "query",
                    "type": "orders",
                    "data": [{
                        "id": o.id,
                        "symbol": o.symbol,
                        "qty": int(o.qty),
                        "side": o.side,
                        "status": o.status,
                        "submitted_at": str(o.submitted_at)
                    } for o in orders]
                }
            
            else:
                return {"status": "error", "message": f"Unknown query type: {query_type}"}
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def process_intent(self, intent: Dict) -> Dict:
        """Route intent to appropriate processor"""
        action = intent.get("action", "").lower()
        
        logger.info(f"Processing intent: {action}")
        
        if action == "buy":
            return self.process_buy_intent(intent)
        elif action in ["sell", "close"]:
            return self.process_sell_intent(intent)
        elif action == "query":
            return self.process_query_intent(intent)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

def main():
    """Main entry point"""
    
    # Read intent from stdin or file
    if len(sys.argv) > 1:
        # Intent provided as file path
        intent_file = sys.argv[1]
        logger.info(f"Reading intent from file: {intent_file}")
        with open(intent_file, 'r') as f:
            intent = json.load(f)
    else:
        # Intent from stdin (for GitHub Actions)
        logger.info("Reading intent from stdin")
        intent = json.load(sys.stdin)
    
    # Process intent
    processor = IntentProcessor()
    result = processor.process_intent(intent)
    
    # Output result
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    if result.get("status") == "success":
        logger.info("Intent processed successfully")
        sys.exit(0)
    else:
        logger.error(f"Intent failed: {result.get('message')}")
        sys.exit(1)

if __name__ == "__main__":
    main()

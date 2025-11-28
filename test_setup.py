#!/usr/bin/env python3
"""
Test script to validate MCP trading setup
Run this to verify all components are working before going live
"""

import os
import sys
from datetime import datetime

def color(text, code):
    """Add color to terminal output"""
    return f"\033[{code}m{text}\033[0m"

def success(msg):
    print(f"✓ {color(msg, '92')}")

def error(msg):
    print(f"✗ {color(msg, '91')}")

def warning(msg):
    print(f"⚠ {color(msg, '93')}")

def info(msg):
    print(f"ℹ {color(msg, '94')}")

def test_environment_variables():
    """Test that all required env vars are set"""
    print("\n" + "="*60)
    print("Testing Environment Variables")
    print("="*60)
    
    required_vars = [
        ("ALPACA_API_KEY", ["ALPACA_API_KEY", "ALPACA_API_KEY_ID"]),
        ("ALPACA_SECRET_KEY", ["ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY"]),
        ("POLYGON_API_KEY", ["POLYGON_API_KEY"])
    ]
    
    optional_vars = [
        "ALPACA_BASE_URL",
        "MAX_POSITION_SIZE",
        "MAX_POSITION_VALUE",
        "UNIVERSE_FILE"
    ]
    
    all_good = True
    
    for display_name, var_options in required_vars:
        found = False
        for var in var_options:
            if os.getenv(var):
                success(f"{display_name} is set (as {var})")
                found = True
                break
        if not found:
            error(f"{display_name} is NOT set (tried: {', '.join(var_options)})")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            success(f"{var} is set: {os.getenv(var)}")
        else:
            warning(f"{var} is not set (will use default)")
    
    return all_good

def test_alpaca_connection():
    """Test Alpaca API connection"""
    print("\n" + "="*60)
    print("Testing Alpaca Connection")
    print("="*60)
    
    try:
        import alpaca_trade_api as tradeapi
        
        # Try both possible variable names
        api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
        api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
        
        api = tradeapi.REST(
            api_key,
            api_secret,
            os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        )
        
        account = api.get_account()
        success("Connected to Alpaca API")
        info(f"Account Status: {account.status}")
        info(f"Buying Power: ${float(account.buying_power):,.2f}")
        info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        info(f"Cash: ${float(account.cash):,.2f}")
        
        if account.pattern_day_trader:
            warning("Account is flagged as Pattern Day Trader")
        
        # Check if paper or live
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if "paper" in base_url:
            info("Using PAPER trading account (safe for testing)")
        else:
            warning("Using LIVE trading account - be careful!")
        
        return True
        
    except Exception as e:
        error(f"Failed to connect to Alpaca: {e}")
        return False

def test_polygon_connection():
    """Test Polygon API connection"""
    print("\n" + "="*60)
    print("Testing Polygon Connection")
    print("="*60)
    
    try:
        from polygon import RESTClient
        
        client = RESTClient(os.getenv("POLYGON_API_KEY"))
        
        # Test with a simple quote fetch
        quote = client.get_last_quote("AAPL")
        success("Connected to Polygon API")
        info(f"Test quote (AAPL): ${(quote.bid_price + quote.ask_price) / 2:.2f}")
        
        return True
        
    except Exception as e:
        error(f"Failed to connect to Polygon: {e}")
        return False

def test_universe_file():
    """Test universe file"""
    print("\n" + "="*60)
    print("Testing Universe File")
    print("="*60)
    
    universe_file = os.getenv("UNIVERSE_FILE", "data/universe_liquid.txt")
    
    if not os.path.exists(universe_file):
        error(f"Universe file not found: {universe_file}")
        warning("Create this file with one symbol per line")
        return False
    
    try:
        with open(universe_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        
        success(f"Universe file loaded: {len(symbols)} symbols")
        
        if len(symbols) == 0:
            warning("Universe file is empty!")
        elif len(symbols) < 10:
            warning(f"Universe is small ({len(symbols)} symbols)")
            info(f"Symbols: {', '.join(symbols)}")
        else:
            info(f"First 10 symbols: {', '.join(symbols[:10])}")
        
        return True
        
    except Exception as e:
        error(f"Failed to read universe file: {e}")
        return False

def test_mcp_server_exists():
    """Check if MCP server file exists"""
    print("\n" + "="*60)
    print("Testing MCP Server Files")
    print("="*60)
    
    files = {
        "mcp_server.py": "MCP server script",
        "tools/mcp_intent_runner.py": "Intent runner script",
        ".github/workflows/mcp-intent.yml": "GitHub Actions workflow"
    }
    
    all_good = True
    
    for file, desc in files.items():
        if os.path.exists(file):
            success(f"{desc} found: {file}")
        else:
            warning(f"{desc} not found: {file}")
            all_good = False
    
    return all_good

def test_safety_limits():
    """Display configured safety limits"""
    print("\n" + "="*60)
    print("Safety Limits Configuration")
    print("="*60)
    
    max_size = int(os.getenv("MAX_POSITION_SIZE", "1000"))
    max_value = float(os.getenv("MAX_POSITION_VALUE", "10000"))
    
    info(f"Max Position Size: {max_size} shares")
    info(f"Max Position Value: ${max_value:,.2f}")
    
    if max_value > 50000:
        warning("Position value limit is high - consider lowering for safety")
    
    return True

def run_live_test():
    """Run a live test with a small position"""
    print("\n" + "="*60)
    print("Live Test (Optional)")
    print("="*60)
    
    response = input("\nDo you want to run a live test? This will check if you can actually place orders (y/N): ")
    
    if response.lower() != 'y':
        info("Skipping live test")
        return True
    
    try:
        import alpaca_trade_api as tradeapi
        from polygon import RESTClient
        
        api = tradeapi.REST(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
            os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        )
        
        polygon = RESTClient(os.getenv("POLYGON_API_KEY"))
        
        # Check market status
        clock = api.get_clock()
        if not clock.is_open:
            warning("Market is currently closed - order will be queued")
        
        # Get a quote
        quote = polygon.get_last_quote("SPY")
        price = (quote.bid_price + quote.ask_price) / 2
        
        info(f"Current SPY price: ${price:.2f}")
        
        confirm = input(f"\nPlace a test order for 1 share of SPY at ~${price:.2f}? (y/N): ")
        
        if confirm.lower() != 'y':
            info("Test order cancelled")
            return True
        
        # Place test order
        order = api.submit_order(
            symbol='SPY',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        success(f"Test order placed: {order.id}")
        info(f"Status: {order.status}")
        
        # Cancel the order immediately (if still pending)
        try:
            if order.status in ['pending_new', 'accepted', 'new']:
                api.cancel_order(order.id)
                success("Test order cancelled successfully")
                info("This was just a test - no actual trade executed")
        except:
            warning("Could not cancel order - it may have filled")
        
        return True
        
    except Exception as e:
        error(f"Live test failed: {e}")
        return False

def main():
    """Run all tests"""
    print(color("\n" + "="*60, "95"))
    print(color("MCP Trading Setup Validation", "95"))
    print(color("="*60, "95"))
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Alpaca Connection", test_alpaca_connection),
        ("Polygon Connection", test_polygon_connection),
        ("Universe File", test_universe_file),
        ("MCP Server Files", test_mcp_server_exists),
        ("Safety Limits", test_safety_limits)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Optional live test
    if all(result for _, result in results[:2]):  # Only if API connections work
        try:
            run_live_test()
        except Exception as e:
            warning(f"Live test failed: {e}")
    
    # Summary
    print("\n" + color("="*60, "95"))
    print(color("Summary", "95"))
    print(color("="*60, "95"))
    
    for name, result in results:
        if result:
            success(f"{name}: PASSED")
        else:
            error(f"{name}: FAILED")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        success("\n✓ All tests passed! System is ready.")
        info("\nNext steps:")
        info("1. Configure Claude mobile with your MCP server")
        info("2. Test a few queries via Claude mobile")
        info("3. Try a small paper trade")
        info("4. Monitor for a week before going live")
    else:
        error("\n✗ Some tests failed. Fix issues before proceeding.")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

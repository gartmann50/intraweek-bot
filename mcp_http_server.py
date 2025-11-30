#!/usr/bin/env python3
"""
HTTP MCP Server wrapper for your existing trading tools.

This reuses the `app = Server("claude-trading-mcp")` defined in mcp_server.py,
but exposes it over HTTP so it can be called remotely (e.g. from Claude).

Trading logic, limits, analytics, etc. all stay in mcp_server.py.
"""

import os
import logging

from mcp.server.http import http_server
from mcp_server import app  # <-- this imports your existing MCP app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Host 0.0.0.0 and a PORT env var are standard for hosting platforms
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8080"))

    logger.info("Starting HTTP MCP server...")
    logger.info(f"Listening on {host}:{port}")
    logger.info("Using trading app from mcp_server.py (claude-trading-mcp)")

    # This starts the official MCP HTTP server using your app
    http_server(app, host=host, port=port)

def start_server():
    import uvicorn
    uvicorn.run(
        "mcp_http_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )



if __name__ == "__main__":
    main()

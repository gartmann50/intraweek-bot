#!/usr/bin/env python3
"""
HTTP wrapper for your existing MCP server.

This lets Claude Mobile talk to your trading tools over HTTPS
(on Railway / Render / etc.) while your original mcp_server.py
still works for Claude Desktop via stdio.
"""

import os
import logging

from mcp.server.http import http_server
from mcp_server import app  # <-- this imports the 'app = Server(...)' you already have

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    """
    Start the HTTP MCP server.

    Most platforms (Railway, Render, etc.) set a PORT environment
    variable automatically, so we respect it. Fallback to 8000 locally.
    """
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    logger.info(f"Starting HTTP MCP server on {host}:{port}")
    logger.info("Using trading app 'claude-trading-mcp' from mcp_server.py")

    # This call blocks and serves HTTP requests from Claude
    http_server(app, host=host, port=port)


if __name__ == "__main__":
    run()

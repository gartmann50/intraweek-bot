#!/usr/bin/env python3
"""
Small HTTP wrapper around your existing mcp_server.py

This does NOT replace mcp_server.py – it just imports it and exposes a few
simple HTTP endpoints that you can later host on Railway/Render/etc.
"""

import asyncio
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import your existing MCP server module
import mcp_server  # this is the file you showed me

from mcp.types import Tool, TextContent, ImageContent

app = FastAPI(title="IW Trading MCP HTTP")

# ---------- Pydantic models for HTTP ----------

class CallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image_base64: str | None = None
    mime_type: str | None = None


class CallResponse(BaseModel):
    contents: List[ContentItem]


def tool_to_dict(t: Tool) -> Dict[str, Any]:
    # Tool is a pydantic model, so we can model_dump it
    return t.model_dump()


def content_to_dict(c: TextContent | ImageContent) -> Dict[str, Any]:
    if isinstance(c, TextContent):
        return {"type": "text", "text": c.text}
    elif isinstance(c, ImageContent):
        return {
            "type": "image",
            "image_base64": c.data,
            "mime_type": c.mimeType,
        }
    else:
        return {"type": "unknown", "text": str(c)}


# ---------- Routes ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tools")
async def list_tools_http():
    """Return list of tools (name + description + schema)."""
    tools = await mcp_server.list_tools()
    return {"tools": [tool_to_dict(t) for t in tools]}


@app.post("/call", response_model=CallResponse)
async def call_tool_http(req: CallRequest):
    """Call one of the tools defined in mcp_server.py."""
    try:
        contents = await mcp_server.call_tool(req.name, req.arguments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = [ContentItem(**content_to_dict(c)) for c in contents]
    return CallResponse(contents=items)

"""MCP discovery and invocation used by the deterministic dispatcher demo."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass(frozen=True)
class MCPInfrastructureResult:
    discovered_tools: tuple[str, ...]
    bridge_status: dict[str, Any]


async def discover_infrastructure_tools() -> MCPInfrastructureResult:
    """Start the local stdio server, list its tools, and query bridge status."""
    server_script = Path(__file__).with_name("bridge_mcp_server.py").resolve()
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env={"SPREELAND_DATA_MODE": "training-replica"},
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            discovered = tuple(tool.name for tool in listed.tools)
            if "get_bridge_status" not in discovered:
                raise RuntimeError("MCP server did not advertise get_bridge_status")
            response = await session.call_tool("get_bridge_status", arguments={})
            if response.isError:
                raise RuntimeError(f"MCP bridge query failed: {response.content}")
            structured = response.structuredContent
            if not structured:
                text_items = [
                    item.text
                    for item in response.content
                    if getattr(item, "type", None) == "text"
                ]
                if not text_items:
                    raise RuntimeError("MCP bridge query returned no usable content")
                structured = json.loads(text_items[0])
            return MCPInfrastructureResult(
                discovered_tools=discovered,
                bridge_status=dict(structured),
            )

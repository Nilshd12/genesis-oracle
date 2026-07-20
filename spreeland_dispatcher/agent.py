"""Google ADK root agent for interactive use in ``adk web``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from .a2a_negotiation import negotiate_gherkin_order

_server_script = Path(__file__).with_name("bridge_mcp_server.py").resolve()


def negotiate_supplier_agents(
    quantity_kg: int = 2000, organic_required: bool = True
) -> dict:
    """Discover supplier Agent Cards and select the best eligible quote."""
    return negotiate_gherkin_order(
        quantity_kg=quantity_kg,
        organic_required=organic_required,
    )


infrastructure_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(_server_script)],
            env={"SPREELAND_DATA_MODE": "training-replica"},
        )
    )
)

root_agent = Agent(
    name="Spreeland_Dispatcher",
    model=os.getenv("SPREELAND_MODEL", "gemini-2.5-flash"),
    description="Coordinates resilient agricultural deliveries in Spreeland.",
    instruction="""You coordinate logistics between Burg/Spreewald and Cottbus.

Use MCP tools to inspect bridge status before proposing a route. Use the
negotiate_gherkin_order tool for the A2A-style supplier negotiation. Explain
decisions as concise operational summaries; never reveal hidden chain-of-thought.
Never claim that the local course-demo HMAC is a production AP2 signature.
Before any real purchase, require an owner-approved AP2 Checkout and Payment
Mandate from a trusted surface. Present final status as an A2UI delivery card
when the connected client supports A2UI.
""",
    tools=[infrastructure_tools, negotiate_supplier_agents],
)

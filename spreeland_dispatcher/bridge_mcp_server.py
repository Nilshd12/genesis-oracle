"""Local MCP server that simulates the City of Cottbus bridge database.

The server deliberately writes no application data to stdout because stdio is
reserved for MCP protocol frames. Replace the in-memory records with
parameterized PostgreSQL queries in a production deployment.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("spreeland-bridge-infrastructure")

BRIDGES = [
    {
        "bridge_id": "SPB-014",
        "name": "Nordumfahrung Luebbenau",
        "route": "Burg-Cottbus via Luebbenau",
        "status": "OPEN",
        "max_total_weight_t": 40.0,
        "estimated_transit_minutes": 52,
        "maintenance_window": None,
    },
    {
        "bridge_id": "SPB-021",
        "name": "Leipe River Bridge",
        "route": "Burg-Cottbus via Leipe",
        "status": "CLOSED",
        "max_total_weight_t": 0.0,
        "estimated_transit_minutes": None,
        "maintenance_window": "2026-07-20T06:00:00Z/2026-07-22T18:00:00Z",
    },
    {
        "bridge_id": "SPB-032",
        "name": "Vetschau Canal Crossing",
        "route": "Burg-Cottbus via Vetschau",
        "status": "RESTRICTED",
        "max_total_weight_t": 7.5,
        "estimated_transit_minutes": 61,
        "maintenance_window": None,
    },
]


@mcp.tool()
def get_bridge_status(route: str | None = None) -> dict:
    """Return current bridge records, optionally filtered by route text."""
    records = BRIDGES
    if route:
        needle = route.casefold()
        records = [
            bridge for bridge in BRIDGES if needle in bridge["route"].casefold()
        ]
    return {
        "source": "City PostgreSQL training replica",
        "observed_at": datetime.now(UTC).isoformat(),
        "bridges": records,
    }


@mcp.tool()
def get_route_capacity(route: str, vehicle_weight_t: float) -> dict:
    """Check whether a vehicle may use every matching bridge on a route."""
    matches = [
        bridge for bridge in BRIDGES if route.casefold() in bridge["route"].casefold()
    ]
    usable = bool(matches) and all(
        bridge["status"] != "CLOSED"
        and vehicle_weight_t <= bridge["max_total_weight_t"]
        for bridge in matches
    )
    return {
        "route": route,
        "vehicle_weight_t": vehicle_weight_t,
        "usable": usable,
        "matching_bridges": matches,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

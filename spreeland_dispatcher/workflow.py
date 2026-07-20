"""End-to-end deterministic workflow for the Spreeland dispatcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .a2a_negotiation import (
    consult_weather,
    load_agent_cards,
    negotiate_gherkin_order,
)
from .mcp_client import discover_infrastructure_tools
from .payments import create_demo_authorization, verify_demo_authorization

ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class DispatchResult:
    mcp_tools: tuple[str, ...]
    route: dict[str, Any]
    weather: dict[str, Any]
    negotiation: dict[str, Any]
    authorization: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_route(
    bridges: list[dict[str, Any]], weather: dict[str, Any], vehicle_weight_t: float
) -> dict[str, Any]:
    eligible = [
        bridge
        for bridge in bridges
        if bridge["status"] != "CLOSED"
        and bridge["max_total_weight_t"] >= vehicle_weight_t
        and weather["route_risk"].get(bridge["route"]) == "LOW"
    ]
    if not eligible:
        raise RuntimeError("No bridge is both capacity-safe and low weather risk")
    return min(eligible, key=lambda bridge: bridge["estimated_transit_minutes"])


async def run_dispatch(progress: ProgressCallback | None = None) -> DispatchResult:
    report = progress or (lambda _step, _summary: None)

    infrastructure = await discover_infrastructure_tools()
    report(
        "MCP_DISCOVERY",
        "Discovered "
        + ", ".join(infrastructure.discovered_tools)
        + " and queried the city bridge replica.",
    )

    cards = load_agent_cards()
    weather = consult_weather(cards)
    report(
        "A2A_WEATHER",
        f"Selected {weather['agent']} by Agent Card skill; {weather['summary']}",
    )

    route = _select_route(
        infrastructure.bridge_status["bridges"],
        weather,
        vehicle_weight_t=6.8,
    )
    report(
        "ROUTE_SELECTION",
        f"Selected {route['route']} via {route['name']} ({route['status']}).",
    )

    negotiation = negotiate_gherkin_order(quantity_kg=2000, cards=cards)
    report("A2A_NEGOTIATION", negotiation["rationale"])

    authorization = create_demo_authorization(negotiation["selected_quote"])
    if not authorization["authorized"] or not verify_demo_authorization(authorization):
        raise RuntimeError("Demo payment authorization failed verification")
    report(
        "AP2_GATE",
        "Budget and tamper-evidence checks passed in the course demo; "
        "a production checkout still requires owner signatures from an AP2 Trusted Surface.",
    )

    return DispatchResult(
        mcp_tools=infrastructure.discovered_tools,
        route=route,
        weather=weather,
        negotiation=negotiation,
        authorization=authorization,
    )

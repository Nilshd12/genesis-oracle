"""Agent-card discovery and deterministic A2A negotiation logic.

The local transport keeps the exercise reproducible without third-party
servers. Agent Cards follow the A2A 1.0 discovery shape. Production code would
replace ``local://`` handlers with HTTPS A2A SendMessage/SendStreamingMessage
calls while retaining the discovery and selection logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentInterface(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: str
    protocol_binding: str = Field(alias="protocolBinding")
    protocol_version: str = Field(alias="protocolVersion")


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = Field(default=False, alias="pushNotifications")


class AgentSkill(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str] = []
    input_modes: list[str] | None = Field(default=None, alias="inputModes")
    output_modes: list[str] | None = Field(default=None, alias="outputModes")


class AgentCard(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    supported_interfaces: list[AgentInterface] = Field(alias="supportedInterfaces")
    version: str
    capabilities: AgentCapabilities
    default_input_modes: list[str] = Field(alias="defaultInputModes")
    default_output_modes: list[str] = Field(alias="defaultOutputModes")
    skills: list[AgentSkill]


@dataclass(frozen=True)
class NegotiationResult:
    selected_agent: str
    selected_quote: dict[str, Any]
    all_quotes: tuple[dict[str, Any], ...]
    rationale: str


def load_agent_cards(card_directory: Path | None = None) -> list[AgentCard]:
    directory = card_directory or Path(__file__).with_name("agent_cards")
    cards = []
    for path in sorted(directory.glob("*.json")):
        cards.append(AgentCard.model_validate_json(path.read_text(encoding="utf-8")))
    if not cards:
        raise RuntimeError(f"No A2A Agent Cards found in {directory}")
    return cards


def discover_agents(cards: list[AgentCard], required_tag: str) -> list[AgentCard]:
    """Return agents advertising at least one skill with the required tag."""
    return [
        card
        for card in cards
        if any(required_tag in skill.tags for skill in card.skills)
    ]


def _local_response(card: AgentCard, skill_tag: str, payload: dict[str, Any]) -> dict:
    """Deterministic stand-in for the remote A2A SendMessage operation."""
    endpoint = card.supported_interfaces[0].url
    if endpoint == "local://weather-predictor" and skill_tag == "route-weather":
        return {
            "agent": card.name,
            "forecast_window": payload["delivery_window"],
            "river_trend": "slowly_rising",
            "route_risk": {
                "Burg-Cottbus via Luebbenau": "LOW",
                "Burg-Cottbus via Leipe": "HIGH",
                "Burg-Cottbus via Vetschau": "MEDIUM",
            },
            "summary": "Moderate rain; avoid Leipe low-water crossing.",
        }
    if endpoint == "local://bio-spree-supplier" and skill_tag == "gherkin-wholesale":
        quantity = int(payload["quantity_kg"])
        return {
            "agent": card.name,
            "quote_id": "BIO-SPREE-2026-0720",
            "quantity_kg": quantity,
            "available_kg": 2300,
            "organic_certified": True,
            "unit_price_eur": 1.85,
            "freight_eur": 90.0,
            "delivery_hours": 6,
        }
    if endpoint == "local://lausitz-supplier" and skill_tag == "gherkin-wholesale":
        quantity = int(payload["quantity_kg"])
        return {
            "agent": card.name,
            "quote_id": "LAUSITZ-2026-0720",
            "quantity_kg": quantity,
            "available_kg": 5000,
            "organic_certified": True,
            "unit_price_eur": 1.74,
            "freight_eur": 380.0,
            "delivery_hours": 9,
        }
    raise LookupError(f"No local A2A handler for {endpoint!r} and {skill_tag!r}")


def consult_weather(
    cards: list[AgentCard], delivery_window: str = "2026-07-20T16:00:00+02:00"
) -> dict[str, Any]:
    candidates = discover_agents(cards, "route-weather")
    if not candidates:
        raise RuntimeError("No weather agent advertises the route-weather skill")
    selected = candidates[0]
    return _local_response(
        selected,
        "route-weather",
        {"region": "Spreeland", "delivery_window": delivery_window},
    )


def negotiate_gherkin_order(
    quantity_kg: int = 2000,
    organic_required: bool = True,
    cards: list[AgentCard] | None = None,
) -> dict[str, Any]:
    """Query eligible supplier agents and choose the cheapest valid landed quote."""
    agent_cards = cards or load_agent_cards()
    suppliers = discover_agents(agent_cards, "gherkin-wholesale")
    if not suppliers:
        raise RuntimeError("No supplier advertises the gherkin-wholesale skill")

    quotes: list[dict[str, Any]] = []
    for supplier in suppliers:
        quote = _local_response(
            supplier,
            "gherkin-wholesale",
            {
                "quantity_kg": quantity_kg,
                "organic_required": organic_required,
                "destination": "Cottbus",
            },
        )
        quote["landed_total_eur"] = round(
            quantity_kg * quote["unit_price_eur"] + quote["freight_eur"], 2
        )
        quote["eligible"] = (
            quote["available_kg"] >= quantity_kg
            and (quote["organic_certified"] or not organic_required)
        )
        quotes.append(quote)

    eligible = [quote for quote in quotes if quote["eligible"]]
    if not eligible:
        raise RuntimeError("No supplier quote meets quantity and certification constraints")
    selected = min(
        eligible, key=lambda quote: (quote["landed_total_eur"], quote["delivery_hours"])
    )
    result = NegotiationResult(
        selected_agent=selected["agent"],
        selected_quote=selected,
        all_quotes=tuple(quotes),
        rationale=(
            f"{selected['agent']} satisfies stock and organic constraints and has "
            f"the lowest landed total ({selected['landed_total_eur']:.2f} EUR)."
        ),
    )
    return {
        "selected_agent": result.selected_agent,
        "selected_quote": result.selected_quote,
        "all_quotes": list(result.all_quotes),
        "rationale": result.rationale,
        "a2a_protocol_version": "1.0",
        "transport": "local deterministic adapter (replace with HTTPS in production)",
    }


def dump_discovery_summary(cards: list[AgentCard]) -> str:
    """Return a compact JSON summary useful to the ADK agent."""
    return json.dumps(
        [
            {
                "name": card.name,
                "skills": [skill.id for skill in card.skills],
                "streaming": card.capabilities.streaming,
            }
            for card in cards
        ],
        ensure_ascii=False,
    )

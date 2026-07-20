from __future__ import annotations

import asyncio
import copy
import json

from spreeland_dispatcher.a2a_negotiation import (
    load_agent_cards,
    negotiate_gherkin_order,
)
from spreeland_dispatcher.a2ui import load_a2ui_sample, validate_a2ui_sample
from spreeland_dispatcher.mcp_client import discover_infrastructure_tools
from spreeland_dispatcher.payments import (
    create_demo_authorization,
    verify_demo_authorization,
)
from spreeland_dispatcher.workflow import run_dispatch


def test_agent_cards_and_negotiation() -> None:
    cards = load_agent_cards()
    assert len(cards) == 3
    result = negotiate_gherkin_order(cards=cards)
    assert result["selected_agent"] == "Bio-Spree Cooperative"
    assert result["selected_quote"]["quantity_kg"] == 2000
    assert result["selected_quote"]["landed_total_eur"] == 3790.0


def test_mcp_tool_discovery_and_call() -> None:
    result = asyncio.run(discover_infrastructure_tools())
    assert "get_bridge_status" in result.discovered_tools
    assert len(result.bridge_status["bridges"]) == 3


def test_a2ui_delivery_card_structure() -> None:
    sample = load_a2ui_sample()
    validate_a2ui_sample(sample)
    surface = sample["messages"][0]["createSurface"]
    assert surface["catalogId"].endswith("/v1_0/catalogs/basic/catalog.json")


def test_demo_authorization_detects_tampering() -> None:
    quote = negotiate_gherkin_order()["selected_quote"]
    evidence = create_demo_authorization(quote)
    assert verify_demo_authorization(evidence)
    tampered = copy.deepcopy(evidence)
    tampered["cart"]["total_eur"] = 1.0
    assert not verify_demo_authorization(tampered)


def test_full_dispatch_flow() -> None:
    result = asyncio.run(run_dispatch()).to_dict()
    assert result["route"]["bridge_id"] == "SPB-014"
    assert result["authorization"]["authorized"] is True
    assert json.dumps(result)


def test_google_adk_agent_is_constructible() -> None:
    from spreeland_dispatcher.agent import root_agent

    assert root_agent.name == "Spreeland_Dispatcher"
    assert type(root_agent.tools[0]).__name__ == "McpToolset"
    assert root_agent.tools[1].__name__ == "negotiate_supplier_agents"

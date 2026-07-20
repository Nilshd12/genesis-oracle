"""Load, update, and structurally validate the submitted A2UI card."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def a2ui_path() -> Path:
    return Path(__file__).with_name("a2ui_delivery_card.json")


def load_a2ui_sample(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or a2ui_path()).read_text(encoding="utf-8"))


def validate_a2ui_sample(sample: dict[str, Any]) -> None:
    """Perform offline structural checks relevant to the submitted card."""
    if not isinstance(sample.get("messages"), list) or not sample["messages"]:
        raise ValueError("A2UI sample must contain a non-empty messages list")
    message = sample["messages"][0]
    if message.get("version") != "v1.0" or "createSurface" not in message:
        raise ValueError("First message must be an A2UI v1.0 createSurface")
    surface = message["createSurface"]
    components = surface.get("components", [])
    ids = [component.get("id") for component in components]
    if "root" not in ids or len(ids) != len(set(ids)):
        raise ValueError("A2UI component IDs must be unique and include root")
    known = set(ids)
    for component in components:
        references: list[str] = []
        if isinstance(component.get("child"), str):
            references.append(component["child"])
        if isinstance(component.get("children"), list):
            references.extend(component["children"])
        missing = [reference for reference in references if reference not in known]
        if missing:
            raise ValueError(
                f"Component {component['id']} references unknown IDs: {missing}"
            )


def render_card_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Bind a dispatch result into the sample's initial data model."""
    sample = copy.deepcopy(load_a2ui_sample())
    surface = sample["messages"][0]["createSurface"]
    surface["dataModel"] = {
        "title": "Spreeland Delivery Status",
        "status": "AUTHORIZED (course demo)",
        "route": result["route"]["route"],
        "bridge": result["route"]["name"],
        "eta": f"{result['route']['estimated_transit_minutes']} minutes",
        "weather": result["weather"]["summary"],
        "supplier": result["negotiation"]["selected_agent"],
        "cargo": "2,000 kg organic gherkins",
        "total": (
            f"{result['negotiation']['selected_quote']['landed_total_eur']:.2f} EUR"
        ),
        "authorization": (
            "Demo signature verified; production requires AP2 Trusted Surface"
        ),
    }
    validate_a2ui_sample(sample)
    return sample

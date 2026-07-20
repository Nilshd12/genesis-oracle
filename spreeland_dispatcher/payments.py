"""Clearly labelled, non-production AP2 evidence for the course demo."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

DEMO_KEY = b"SPREELAND_COURSE_DEMO_KEY_NOT_FOR_PRODUCTION"


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def create_demo_authorization(
    quote: dict[str, Any], owner_budget_eur: float = 4000.0
) -> dict[str, Any]:
    """Create verifiable local evidence; this is not a normative AP2 Mandate."""
    cart = {
        "quote_id": quote["quote_id"],
        "supplier": quote["agent"],
        "item": "organic gherkins",
        "quantity_kg": quote["quantity_kg"],
        "total_eur": quote["landed_total_eur"],
    }
    authorized = cart["total_eur"] <= owner_budget_eur
    evidence = {
        "profile": "COURSE_DEMO_ONLY_NOT_AP2_CONFORMANT",
        "authorized": authorized,
        "owner_budget_eur": owner_budget_eur,
        "checkout_sha256": hashlib.sha256(_canonical_bytes(cart)).hexdigest(),
        "cart": cart,
    }
    evidence["demo_hmac_sha256"] = hmac.new(
        DEMO_KEY, _canonical_bytes(evidence), hashlib.sha256
    ).hexdigest()
    return evidence


def verify_demo_authorization(evidence: dict[str, Any]) -> bool:
    signature = evidence.get("demo_hmac_sha256")
    unsigned = {key: value for key, value in evidence.items() if key != "demo_hmac_sha256"}
    expected = hmac.new(
        DEMO_KEY, _canonical_bytes(unsigned), hashlib.sha256
    ).hexdigest()
    return bool(signature) and hmac.compare_digest(signature, expected)

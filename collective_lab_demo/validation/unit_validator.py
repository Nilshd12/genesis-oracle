"""Deterministic thermal-conductivity unit conversion checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite


CANONICAL_CENTIMETRE_UNIT = "W/(cm*K)"
CANONICAL_METRE_UNIT = "W/(m*K)"
SUPPORTED_UNITS = frozenset(
    {CANONICAL_CENTIMETRE_UNIT, CANONICAL_METRE_UNIT}
)

_ALIASES = {
    "W/(cm*K)": CANONICAL_CENTIMETRE_UNIT,
    "W/(cm·K)": CANONICAL_CENTIMETRE_UNIT,
    "W/(cm K)": CANONICAL_CENTIMETRE_UNIT,
    "W/cm K": CANONICAL_CENTIMETRE_UNIT,
    "W/(m*K)": CANONICAL_METRE_UNIT,
    "W/(m·K)": CANONICAL_METRE_UNIT,
    "W/(m K)": CANONICAL_METRE_UNIT,
    "W/m K": CANONICAL_METRE_UNIT,
}

_TO_SI_FACTORS = {
    CANONICAL_CENTIMETRE_UNIT: 100.0,
    CANONICAL_METRE_UNIT: 1.0,
}


@dataclass(frozen=True, slots=True)
class UnitValidationResult:
    """Detailed outcome for an auditable conversion comparison."""

    valid: bool
    expected_value: float | None
    expected_unit: str | None
    message: str


def normalize_unit(unit: str) -> str | None:
    """Return a canonical supported unit, or ``None`` if unknown."""

    return _ALIASES.get(unit.strip()) if isinstance(unit, str) else None


def is_supported_unit(unit: str) -> bool:
    """Whether ``unit`` is a recognized conductivity unit."""

    return normalize_unit(unit) is not None


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert thermal conductivity between centimetre and metre units."""

    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if source is None:
        raise ValueError(f"Unsupported source unit: {from_unit!r}")
    if target is None:
        raise ValueError(f"Unsupported target unit: {to_unit!r}")
    if not isfinite(value):
        raise ValueError("The conductivity value must be finite.")

    value_in_si = value * _TO_SI_FACTORS[source]
    return value_in_si / _TO_SI_FACTORS[target]


def validate_conversion(
    source_value: float,
    source_unit: str,
    received_value: float,
    received_unit: str,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> UnitValidationResult:
    """Compare a payload against the deterministic source conversion."""

    try:
        canonical_target = normalize_unit(received_unit)
        expected = convert_value(source_value, source_unit, received_unit)
    except (TypeError, ValueError) as error:
        return UnitValidationResult(False, None, None, str(error))

    matches = isfinite(received_value) and isclose(
        expected,
        received_value,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )
    return UnitValidationResult(
        valid=matches,
        expected_value=expected,
        expected_unit=canonical_target,
        message="conversion verified" if matches else "unit conversion mismatch",
    )

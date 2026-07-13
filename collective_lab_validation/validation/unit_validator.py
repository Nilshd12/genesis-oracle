"""Element-wise deterministic conductivity conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Mapping, Sequence

from collective_lab_validation.models.parameter_payload import Matrix


SOURCE_UNIT = "W/(cm*K)"
TARGET_UNIT = "W/(m*K)"
_ALIASES = {
    "W/(cm*K)": SOURCE_UNIT,
    "W/(cm·K)": SOURCE_UNIT,
    "W/(cm K)": SOURCE_UNIT,
    "W/(m*K)": TARGET_UNIT,
    "W/(m·K)": TARGET_UNIT,
    "W/(m K)": TARGET_UNIT,
}
_TO_SI = {SOURCE_UNIT: 100.0, TARGET_UNIT: 1.0}


@dataclass(frozen=True, slots=True)
class MatrixConversionResult:
    valid: bool
    expected_matrix: Matrix | None
    expected_unit: str | None
    errors: tuple[str, ...]


def normalize_unit(unit: Any) -> str | None:
    return _ALIASES.get(unit.strip()) if isinstance(unit, str) else None


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if source is None:
        raise ValueError(f"unsupported source unit: {from_unit!r}")
    if target is None:
        raise ValueError(f"unsupported target unit: {to_unit!r}")
    if not isfinite(float(value)):
        raise ValueError("conductivity must be finite")
    return round(float(value) * _TO_SI[source] / _TO_SI[target], 12)


def source_matrix(source_values: Mapping[str, float]) -> Matrix:
    if "parallel" not in source_values or "perpendicular" not in source_values:
        raise ValueError("source_values require parallel and perpendicular")
    return [
        [float(source_values["parallel"]), 0.0],
        [0.0, float(source_values["perpendicular"])],
    ]


def convert_matrix(matrix: Sequence[Sequence[float]], from_unit: str, to_unit: str) -> Matrix:
    return [
        [convert_value(float(value), from_unit, to_unit) for value in row]
        for row in matrix
    ]


def validate_matrix_conversion(
    source_values: Mapping[str, float],
    source_unit: str,
    received_matrix: Sequence[Sequence[float]],
    received_unit: str,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> MatrixConversionResult:
    try:
        expected = convert_matrix(
            source_matrix(source_values),
            source_unit,
            TARGET_UNIT,
        )
    except (KeyError, TypeError, ValueError) as error:
        return MatrixConversionResult(False, None, TARGET_UNIT, (str(error),))

    errors: list[str] = []
    if normalize_unit(received_unit) != TARGET_UNIT:
        errors.append("target unit must be W/(m*K)")
    if len(received_matrix) != 2 or any(len(row) != 2 for row in received_matrix):
        errors.append("matrix must have shape 2x2")
        return MatrixConversionResult(False, expected, TARGET_UNIT, tuple(errors))
    for row_index in range(2):
        for column_index in range(2):
            received = float(received_matrix[row_index][column_index])
            wanted = float(expected[row_index][column_index])
            if not isfinite(received) or not isclose(
                received,
                wanted,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                errors.append(
                    f"matrix[{row_index}][{column_index}] expected {wanted:g}, "
                    f"received {received:g}"
                )
    return MatrixConversionResult(not errors, expected, TARGET_UNIT, tuple(errors))

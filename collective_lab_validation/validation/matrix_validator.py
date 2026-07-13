"""Physical and structural checks for a 2x2 conductivity tensor."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class MatrixValidationResult:
    valid: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]


def validate_conductivity_matrix(matrix: Any) -> MatrixValidationResult:
    checks = {
        "matrix_shape_2x2": False,
        "matrix_elements_numeric": False,
        "matrix_values_finite": False,
        "matrix_symmetric": False,
        "positive_diagonal": False,
        "positive_definite": False,
        "off_diagonal_supported_by_source": False,
    }
    errors: list[str] = []
    if (
        not isinstance(matrix, Sequence)
        or isinstance(matrix, (str, bytes))
        or len(matrix) != 2
        or any(
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 2
            for row in matrix
        )
    ):
        errors.append("matrix must have shape 2x2")
        return MatrixValidationResult(False, checks, tuple(errors))
    checks["matrix_shape_2x2"] = True

    values = [value for row in matrix for value in row]
    numeric = all(isinstance(value, Real) and not isinstance(value, bool) for value in values)
    checks["matrix_elements_numeric"] = numeric
    if not numeric:
        errors.append("all matrix elements must be numeric")
        return MatrixValidationResult(False, checks, tuple(errors))

    finite = all(isfinite(float(value)) for value in values)
    checks["matrix_values_finite"] = finite
    if not finite:
        errors.append("all matrix elements must be finite")
        return MatrixValidationResult(False, checks, tuple(errors))

    a, b = float(matrix[0][0]), float(matrix[0][1])
    c, d = float(matrix[1][0]), float(matrix[1][1])
    checks["matrix_symmetric"] = isclose(b, c, rel_tol=1e-9, abs_tol=1e-12)
    if not checks["matrix_symmetric"]:
        errors.append("conductivity matrix must be symmetric")
    checks["positive_diagonal"] = a > 0 and d > 0
    if not checks["positive_diagonal"]:
        errors.append("conductivity diagonal values must be positive")
    checks["positive_definite"] = a > 0 and (a * d - b * c) > 0
    if not checks["positive_definite"]:
        errors.append("conductivity matrix must be positive definite")
    checks["off_diagonal_supported_by_source"] = isclose(
        b, 0.0, abs_tol=1e-12
    ) and isclose(c, 0.0, abs_tol=1e-12)
    if not checks["off_diagonal_supported_by_source"]:
        errors.append("source provides principal values only; off-diagonal entries must be zero")
    return MatrixValidationResult(not errors, checks, tuple(errors))

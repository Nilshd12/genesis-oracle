from __future__ import annotations

from collective_lab_validation.validation.matrix_validator import (
    validate_conductivity_matrix,
)


def test_non_symmetric_matrix_is_rejected() -> None:
    result = validate_conductivity_matrix([[1.4, 0.2], [0.0, 1.5]])
    assert not result.valid
    assert not result.checks["matrix_symmetric"]


def test_negative_conductivity_is_rejected() -> None:
    result = validate_conductivity_matrix([[-1.4, 0.0], [0.0, 1.5]])
    assert not result.valid
    assert not result.checks["positive_diagonal"]


def test_wrong_matrix_dimension_and_nonfinite_values_are_rejected() -> None:
    assert not validate_conductivity_matrix([[1.4, 0.0, 0.0]]).valid
    result = validate_conductivity_matrix([[float("inf"), 0.0], [0.0, 1.5]])
    assert not result.valid
    assert not result.checks["matrix_values_finite"]

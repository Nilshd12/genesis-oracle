from __future__ import annotations

import pytest

from collective_lab_validation.validation.unit_validator import (
    convert_matrix,
    convert_value,
)


def test_parallel_value_converts_to_si() -> None:
    assert convert_value(0.014, "W/(cm*K)", "W/(m*K)") == pytest.approx(1.4)


def test_perpendicular_value_converts_to_si() -> None:
    assert convert_value(0.015, "W/(cm*K)", "W/(m*K)") == pytest.approx(1.5)


def test_complete_tensor_converts_elementwise() -> None:
    converted = convert_matrix(
        [[0.014, 0.0], [0.0, 0.015]],
        "W/(cm*K)",
        "W/(m*K)",
    )
    assert converted == [[1.4, 0.0], [0.0, 1.5]]

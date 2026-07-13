"""Load VERIFIED JSON plus gate decision and run the anisotropic JAX model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from collective_lab_validation.cli._io import write_json
from collective_lab_validation.models.gate_decision import GateDecision
from collective_lab_validation.models.parameter_payload import ParameterPayload
from collective_lab_validation.simulation.jax_heat_simulation import (
    run_heat_simulation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run approved JAX heat simulation.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--grid-size", type=int, default=51)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        payload = ParameterPayload.model_validate_json(
            Path(args.payload).read_text(encoding="utf-8")
        )
        gate = GateDecision.model_validate_json(
            Path(args.gate).read_text(encoding="utf-8")
        )
        result = run_heat_simulation(
            payload,
            gate,
            image_path=args.image,
            grid_size=args.grid_size,
            steps=args.steps,
        )
        write_json(args.output, result)
        print("JAX-Wärmesimulation erfolgreich beendet.")
        return 0
    except Exception as error:
        print(f"JAX-Wärmesimulation failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic gate CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from collective_lab_validation.cli._io import write_json
from collective_lab_validation.models.audit_result import AuditResult
from collective_lab_validation.models.gate_decision import GateStatus
from collective_lab_validation.models.parameter_payload import ParameterPayload
from collective_lab_validation.validation.deterministic_gate import evaluate_gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic execution gate.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        payload = ParameterPayload.model_validate_json(
            Path(args.payload).read_text(encoding="utf-8")
        )
        audit = AuditResult.model_validate_json(
            Path(args.audit).read_text(encoding="utf-8")
        )
        decision = evaluate_gate(payload, audit)
        write_json(args.output, decision)
        if decision.status is GateStatus.APPROVED:
            print("APPROVED: JAX execution allowed")
        else:
            print("BLOCKED: JAX execution denied")
            print("JAX wurde nicht gestartet.")
        return 0
    except Exception as error:
        print(f"deterministisches Gate failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

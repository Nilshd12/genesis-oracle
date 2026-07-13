"""Scholar-Prime CLI for initial extraction or audit-directed correction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from collective_lab_validation.agents.scholar_prime import ScholarPrime
from collective_lab_validation.cli._io import write_json
from collective_lab_validation.models.audit_result import AuditResult
from collective_lab_validation.models.parameter_payload import ParameterPayload
from collective_lab_validation.research.science_skill_adapter import research_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Scholar-Prime.")
    parser.add_argument("--mode", choices=("demo", "online"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-output")
    parser.add_argument("--inject-reproducible-error", action="store_true")
    parser.add_argument("--correct-payload")
    parser.add_argument("--audit-report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scholar = ScholarPrime()
        if args.correct_payload or args.audit_report:
            if not args.correct_payload or not args.audit_report:
                raise ValueError("Correction needs --correct-payload and --audit-report")
            payload = ParameterPayload.model_validate_json(
                Path(args.correct_payload).read_text(encoding="utf-8")
            )
            audit = AuditResult.model_validate_json(
                Path(args.audit_report).read_text(encoding="utf-8")
            )
            corrected = scholar.correct(payload, audit)
            print("Scholar-Prime korrigiert die Leitfähigkeitsmatrix.")
            print(write_json(args.output, corrected))
            return 0
        if not args.mode:
            raise ValueError("Initial extraction requires --mode")
        research = research_source(args.mode)
        if args.source_output:
            write_json(args.source_output, research.source)
        if args.mode == "demo" or research.actual_mode == "fixture-fallback":
            print("Lokale Quellen-Fixture – keine aktuelle Online-Recherche.")
        else:
            print(f"Recherchemodus: {research.actual_mode}")
        if args.inject_reproducible_error:
            print("Reproduzierbarer Einheitenfehler wird für die Demonstration injiziert.")
        payload = scholar.extract(
            research.source,
            inject_reproducible_error=args.inject_reproducible_error,
        )
        print(write_json(args.output, payload))
        return 0
    except Exception as error:
        print(f"Scholar-Prime failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Auditor-Agent CLI for untrusted JSON review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from collective_lab_validation.agents.auditor_agent import AuditorAgent
from collective_lab_validation.cli._io import write_json
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    WorkflowStatus,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Auditor-Agent review.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verified-payload-output")
    args = parser.parse_args(argv)
    try:
        untrusted = json.loads(Path(args.input).read_text(encoding="utf-8"))
        audit = AuditorAgent().audit(untrusted)
        write_json(args.output, audit)
        if (
            audit.status is WorkflowStatus.VERIFIED
            and args.verified_payload_output
        ):
            verified = ParameterPayload.model_validate(untrusted).model_copy(
                update={"validation_status": WorkflowStatus.VERIFIED}
            )
            write_json(args.verified_payload_output, verified)
        print(audit.audit_summary)
        return 0
    except Exception as error:
        print(f"Auditor-Agent failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

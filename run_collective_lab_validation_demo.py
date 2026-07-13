"""Local backup runner for the matrix-based Collective Lab validation demo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from collective_lab_validation.agents import AuditorAgent, ScholarPrime
from collective_lab_validation.cli._io import write_json
from collective_lab_validation.models.audit_result import AuditResult
from collective_lab_validation.models.gate_decision import GateDecision
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    WorkflowStatus,
)
from collective_lab_validation.research import ResearchResult, research_source
from collective_lab_validation.simulation import SimulationResult, run_heat_simulation
from collective_lab_validation.validation.deterministic_gate import evaluate_gate


PROJECT_ROOT = Path(__file__).resolve().parent
GENERATED_NAMES = (
    "source_record.json",
    "extracted_payload.json",
    "rejected_audit_report.json",
    "blocked_gate_decision.json",
    "corrected_payload.json",
    "verified_audit_report.json",
    "approved_gate_decision.json",
    "verified_conductivity_matrix.json",
    "simulation_result.json",
    "jax_heat_result.png",
)


@dataclass(frozen=True, slots=True)
class DemoSummary:
    research: ResearchResult
    rejected_audit: AuditResult
    blocked_gate: GateDecision
    verified_payload: ParameterPayload
    verified_audit: AuditResult
    approved_gate: GateDecision
    simulation: SimulationResult
    artifacts: tuple[Path, ...]


def _pause(enabled: bool, output: Callable[[str], None]) -> None:
    if not enabled:
        return
    if sys.stdin.isatty():
        input("[Präsentation] Enter für den nächsten Schritt ... ")
    else:
        output("[Präsentation] Pause übersprungen (nicht interaktiv).")


def _phase(number: int, text: str, output: Callable[[str], None]) -> None:
    output(f"\n[{number:02d}] {text}")


def run_demo(
    *,
    mode: str = "demo",
    presentation: bool = False,
    no_pause: bool = False,
    artifacts_dir: str | Path = PROJECT_ROOT / "collective_lab_artifacts",
    output: Callable[[str], None] = print,
) -> DemoSummary:
    """Reproduce the sequential peer-review loop without pretending UI agents."""

    artifact_dir = Path(artifacts_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_NAMES:
        candidate = artifact_dir / name
        if candidate.is_file():
            candidate.unlink()
    paths = {name: artifact_dir / name for name in GENERATED_NAMES}
    written: list[Path] = []
    scholar = ScholarPrime()
    auditor = AuditorAgent()

    output("=" * 72)
    output("DAS KOLLEKTIVE LABOR – Wissenschaftliche Parameter-Validierung")
    output("=" * 72)
    output("Lokaler Test-/Backup-Runner; native Antigravity-Agenten siehe Anleitung.")

    _phase(1, "Scholar-Prime recherchiert eine wissenschaftliche Quelle", output)
    research = research_source(mode)
    source = research.source
    if research.actual_mode == "science-skill":
        output("DeepMind Science Skill: literature-search-openalex")
        output("Online-Recherche über OpenAlex erfolgreich.")
    elif research.actual_mode == "openalex-api-fallback":
        output("Direkter OpenAlex-API-Fallback – nicht als Science Skill ausgegeben.")
    else:
        output("Lokale Quellen-Fixture – keine aktuelle Online-Recherche.")
    output(f"Quelle: {source.title} ({source.publication_year})")
    output(f"DOI: {source.doi}")
    output("Quellenwerte: parallel 0.014, senkrecht 0.015 W/(cm*K)")
    written.append(write_json(paths["source_record.json"], source))
    _pause(presentation and not no_pause, output)

    _phase(2, "Scholar-Prime erzeugt das strukturierte JSON-Payload", output)
    output("Reproduzierbarer Einheitenfehler wird für die Demonstration injiziert.")
    extracted = scholar.extract(source, inject_reproducible_error=True)
    output(json.dumps(extracted.matrix, indent=2))
    output(f"Einheit: {extracted.unit}")
    written.append(write_json(paths["extracted_payload.json"], extracted))

    _phase(3, "Auditor-Agent beginnt die unabhängige Prüfung", output)
    rejected_audit = auditor.audit(extracted)
    rejected_payload = extracted.model_copy(
        update={"validation_status": rejected_audit.status}
    )
    output(rejected_audit.audit_summary)
    output("REJECTED – Korrektur erforderlich")
    output("Erwartete Matrix:")
    output(json.dumps(rejected_audit.expected_matrix, indent=2))
    output("Erhaltene Matrix:")
    output(json.dumps(rejected_audit.received_matrix, indent=2))
    written.append(
        write_json(paths["rejected_audit_report.json"], rejected_audit)
    )

    _phase(4, "Deterministisches Gate prüft den ersten Datensatz", output)
    blocked_gate = evaluate_gate(rejected_payload, rejected_audit)
    output("BLOCKED: JAX execution denied")
    output("JAX wurde nicht gestartet.")
    written.append(
        write_json(paths["blocked_gate_decision.json"], blocked_gate)
    )
    _pause(presentation and not no_pause, output)

    _phase(5, "Audit-Feedback wird an Scholar-Prime zurückgegeben", output)
    output(rejected_audit.correction_instruction or "Kein Korrekturauftrag")

    _phase(6, "Scholar-Prime korrigiert die Leitfähigkeitsmatrix", output)
    corrected = scholar.correct(rejected_payload, rejected_audit)
    output("[[0.014, 0.0], [0.0, 0.015]] W/(cm*K)")
    output("wird zu")
    output("[[1.4, 0.0], [0.0, 1.5]] W/(m*K)")
    written.append(write_json(paths["corrected_payload.json"], corrected))

    _phase(7, "Auditor-Agent prüft den korrigierten Datensatz erneut", output)
    verified_audit = auditor.audit(corrected)
    if verified_audit.status is not WorkflowStatus.VERIFIED:
        raise RuntimeError(verified_audit.audit_summary)
    verified_payload = corrected.model_copy(
        update={"validation_status": WorkflowStatus.VERIFIED}
    )
    output(verified_audit.audit_summary)
    output("VERIFIED – Freigabe für JAX")
    written.append(
        write_json(paths["verified_audit_report.json"], verified_audit)
    )
    written.append(
        write_json(
            paths["verified_conductivity_matrix.json"],
            verified_payload,
        )
    )

    _phase(8, "Deterministisches Gate trifft die finale Entscheidung", output)
    approved_gate = evaluate_gate(verified_payload, verified_audit)
    if not approved_gate.execution_allowed:
        raise RuntimeError("Finales Gate hat die JAX-Ausführung blockiert.")
    output("APPROVED: JAX execution allowed")
    written.append(
        write_json(paths["approved_gate_decision.json"], approved_gate)
    )
    _pause(presentation and not no_pause, output)

    _phase(9, "JAX-Wärmesimulation startet", output)
    # The simulation consumes the persisted verified JSON artifact.
    persisted_payload = ParameterPayload.model_validate_json(
        paths["verified_conductivity_matrix.json"].read_text(encoding="utf-8")
    )
    simulation = run_heat_simulation(
        persisted_payload,
        approved_gate,
        image_path=paths["jax_heat_result.png"],
    )
    output(f"Leitfähigkeitsmatrix: {simulation.conductivity_matrix}")
    output(f"Einheit: {simulation.unit}")
    output(f"Rastergröße: {simulation.grid_shape}")
    output(f"Simulationsschritte: {simulation.steps}")
    output(
        "Temperatur [°C] – "
        f"min {simulation.minimum_temperature_celsius:.2f}, "
        f"max {simulation.maximum_temperature_celsius:.2f}, "
        f"mittel {simulation.mean_temperature_celsius:.2f}"
    )
    output("JAX-Wärmesimulation erfolgreich beendet.")
    written.append(write_json(paths["simulation_result.json"], simulation))
    written.append(paths["jax_heat_result.png"])

    _phase(10, "Artefakte und Ergebnis werden angezeigt", output)
    for path in written:
        output(f"- {path}")
    return DemoSummary(
        research=research,
        rejected_audit=rejected_audit,
        blocked_gate=blocked_gate,
        verified_payload=verified_payload,
        verified_audit=verified_audit,
        approved_gate=approved_gate,
        simulation=simulation,
        artifacts=tuple(written),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collective Lab validation demo")
    parser.add_argument("--mode", choices=("demo", "online"), default="demo")
    parser.add_argument("--presentation", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "collective_lab_artifacts"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_demo(
            mode=args.mode,
            presentation=args.presentation,
            no_pause=args.no_pause,
            artifacts_dir=args.artifacts_dir,
        )
    except Exception as error:
        print(f"Collective Lab validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

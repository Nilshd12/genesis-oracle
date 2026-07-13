"""Presentation-friendly entry point for the Collective Lab demo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from collective_lab_demo.agents import AuditorAgent, ScholarPrime
from collective_lab_demo.models.parameter_models import (
    GateDecision,
    ParameterRecord,
    SimulationResult,
    WorkflowStatus,
)
from collective_lab_demo.services import ArtifactService, LiteratureService
from collective_lab_demo.simulation import run_heat_simulation
from collective_lab_demo.validation.gate import evaluate_gate


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class DemoRunSummary:
    """Programmatic result used by the integration test and CLI."""

    success: bool
    final_parameter: ParameterRecord
    final_gate: GateDecision
    simulation: SimulationResult
    artifacts: tuple[Path, ...]
    actual_literature_mode: str


def _pause(enabled: bool, output: Callable[[str], None]) -> None:
    if not enabled:
        return
    if sys.stdin.isatty():
        input("    [Enter] für den nächsten Schritt ... ")
    else:
        output("    Pause übersprungen (keine interaktive Eingabe).")


def _phase(number: int, title: str, output: Callable[[str], None]) -> None:
    output(f"\n[{number:02d}] {title}")


def _parameter_json(record: ParameterRecord) -> str:
    return json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False)


def run_demo(
    *,
    mode: str = "demo",
    no_pause: bool = False,
    artifacts_dir: str | Path = PROJECT_ROOT / "demo_artifacts",
    output: Callable[[str], None] = print,
) -> DemoRunSummary:
    """Run extraction, rejection/correction when needed, gate, and JAX."""

    artifacts = ArtifactService(artifacts_dir)
    artifacts.prepare_run()
    written: list[Path] = []
    scholar = ScholarPrime()
    auditor = AuditorAgent()

    output("=" * 68)
    output("COLLECTIVE LAB – geprüfte Parameter vor JAX")
    output("=" * 68)
    output(f"Angeforderter Modus: {mode}")

    _phase(1, "Scholar-Prime lädt die wissenschaftliche Quelle", output)
    literature = LiteratureService().research(mode)
    if literature.actual_mode == "online-openalex":
        output("Quellenmodus: ONLINE – echte OpenAlex-Abfrage erfolgreich")
    elif literature.actual_mode == "fixture-fallback":
        output("Quellenmodus: FIXTURE-FALLBACK – keine aktuelle Online-Recherche")
        output(f"Fallback-Grund: {literature.fallback_reason}")
    else:
        output("Quellenmodus: REPRODUZIERBARE LOKALE FIXTURE")
    source = literature.source
    output(f"Quelle: {source.title}")
    output(f"Autor(en): {', '.join(source.authors)} ({source.publication_year})")
    output(f"URL/DOI: {source.url}")
    output(
        f"Quellenparameter: {source.original_value:g} {source.original_unit}"
    )
    written.append(artifacts.write_json("source_record.json", source))
    _pause(not no_pause, output)

    _phase(2, "Scholar-Prime erzeugt das strukturierte JSON", output)
    inject_demo_error = mode == "demo"
    parameter = scholar.extract(source, inject_demo_error=inject_demo_error)
    if inject_demo_error:
        output("Reproduzierbarer Demofehler wird injiziert.")
    output(_parameter_json(parameter))
    written.append(
        artifacts.write_json("extracted_parameters.json", parameter)
    )
    _pause(not no_pause, output)

    _phase(3, "Auditor-Agent prüft Quelle, Schema, Wert und Einheit", output)
    first_audit = auditor.audit(parameter)
    parameter = parameter.model_copy(
        update={"validation_status": first_audit.status}
    )
    output(first_audit.audit_summary)

    if first_audit.status is WorkflowStatus.REJECTED:
        written.append(
            artifacts.write_json("rejected_audit_report.json", first_audit)
        )
        output(
            f"Erwartet: {first_audit.expected_value:g} "
            f"{first_audit.expected_unit} | Erhalten: "
            f"{first_audit.received_value:g} {first_audit.received_unit}"
        )

        _phase(4, "Deterministisches Gate bewertet den ersten Datensatz", output)
        blocked = evaluate_gate(parameter, first_audit)
        output("BLOCKED: JAX execution denied")
        output("JAX wurde nicht gestartet.")
        written.append(
            artifacts.write_json("blocked_gate_decision.json", blocked)
        )
        # gate_decision.json always reflects the latest decision; the blocked
        # result remains separately traceable.
        artifacts.write_json("gate_decision.json", blocked)
        _pause(not no_pause, output)

        _phase(5, "Audit-Feedback geht zurück an Scholar-Prime", output)
        output(first_audit.correction_instruction or "Keine Korrektur verfügbar")
        corrected = scholar.correct(parameter, first_audit)
        output(
            f"Scholar-Prime korrigiert: {parameter.source_value:g} "
            f"{parameter.source_unit} -> {corrected.value:g} {corrected.unit}"
        )
        written.append(
            artifacts.write_json("corrected_parameters.json", corrected)
        )
        _pause(not no_pause, output)

        _phase(6, "Auditor-Agent prüft den korrigierten Datensatz erneut", output)
        final_audit = auditor.audit(corrected)
        final_parameter = corrected.model_copy(
            update={"validation_status": final_audit.status}
        )
        output(final_audit.audit_summary)
    else:
        # Online mode does not inject an error; a correct extraction needs no
        # artificial correction loop.
        final_audit = first_audit
        final_parameter = parameter

    if final_audit.status is not WorkflowStatus.VERIFIED:
        raise RuntimeError("Auditor verification failed; simulation remains blocked.")
    written.append(
        artifacts.write_json("verified_audit_report.json", final_audit)
    )

    _phase(7, "Deterministisches Gate trifft die finale Entscheidung", output)
    final_gate = evaluate_gate(final_parameter, final_audit)
    if not final_gate.execution_allowed:
        output("BLOCKED: JAX execution denied")
        artifacts.write_json("gate_decision.json", final_gate)
        raise RuntimeError("Final deterministic gate did not approve JAX.")
    output("APPROVED: JAX execution allowed")
    written.append(artifacts.write_json("gate_decision.json", final_gate))
    _pause(not no_pause, output)

    _phase(8, "JAX-Wärmesimulation startet nach Freigabe", output)
    simulation = run_heat_simulation(final_parameter, final_gate)
    output(
        f"Leitfähigkeit: {simulation.thermal_conductivity:g} {simulation.unit}"
    )
    output(f"Simulationsschritte: {simulation.steps}")
    output(
        "Temperatur [°C] – "
        f"min: {simulation.minimum_temperature_celsius:.2f}, "
        f"max: {simulation.maximum_temperature_celsius:.2f}, "
        f"mittel: {simulation.mean_temperature_celsius:.2f}"
    )
    output("Simulation erfolgreich abgeschlossen.")
    written.append(
        artifacts.write_json("simulation_result.json", simulation)
    )

    _phase(9, "Artefakte", output)
    for path in written:
        output(f"- {path}")
    output("\nCOLLECTIVE LAB erfolgreich beendet.")
    return DemoRunSummary(
        success=True,
        final_parameter=final_parameter,
        final_gate=final_gate,
        simulation=simulation,
        artifacts=tuple(written),
        actual_literature_mode=literature.actual_mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collective Lab: audit-gated JAX heat simulation"
    )
    parser.add_argument(
        "--mode",
        choices=("demo", "online"),
        default="demo",
        help="demo uses a fixture; online queries OpenAlex with fixture fallback",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="disable presentation pauses",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "demo_artifacts"),
        help="directory for human-readable JSON artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_demo(
            mode=args.mode,
            no_pause=args.no_pause,
            artifacts_dir=args.artifacts_dir,
        )
    except Exception as error:
        print(f"\nCOLLECTIVE LAB fehlgeschlagen: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

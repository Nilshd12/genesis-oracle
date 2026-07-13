# Antigravity-Orchestrator: Das kollektive Labor

Verwende das aktuell in Antigravity ausgewählte verfügbare Modell. Koordiniere eine sequenzielle wissenschaftliche Peer-Review-Schleife mit **genau zwei nativen Subagenten** als direkte Kinder: `Scholar-Prime` und `Auditor-Agent`.

Der lokale Runner `run_collective_lab_validation_demo.py` darf nicht als Ersatz für die native Agentenkommunikation verwendet werden. Die Fraktal-Explorer gehören nicht zu dieser Aufgabe.

## Verbindlicher Ablauf

1. Starte den nativen Subagenten `Scholar-Prime` mit `collective_lab_validation/antigravity/scholar_prime_prompt.md`.
2. Scholar-Prime muss den installierten DeepMind Science Skill `literature-search-openalex` über die bereitgestellte CLI verwenden und diese Dateien erzeugen:
   - `collective_lab_artifacts/source_record.json`
   - `collective_lab_artifacts/extracted_payload.json`
3. Warte auf die strukturierte Rückgabe von Scholar-Prime. Starte erst danach den nativen Subagenten `Auditor-Agent` mit `collective_lab_validation/antigravity/auditor_agent_prompt.md`.
4. Übergib `extracted_payload.json` an denselben Auditor-Agenten. Er muss `rejected_audit_report.json` erzeugen und `REJECTED: unit conversion mismatch` zurückgeben.
5. Führe als Orchestrator das erste deterministische Gate aus:

   ```powershell
   uv run python -m collective_lab_validation.cli.gate_check --payload collective_lab_artifacts/extracted_payload.json --audit collective_lab_artifacts/rejected_audit_report.json --output collective_lab_artifacts/blocked_gate_decision.json
   ```

   Verlange `BLOCKED: JAX execution denied`. Starte JAX nicht.
6. Übergib das Audit-Feedback an **denselben Scholar-Prime-Subagenten**. Er muss denselben Datensatz korrigieren und `corrected_payload.json` erzeugen.
7. Übergib das korrigierte Payload erneut an **denselben Auditor-Agenten**. Er muss `verified_audit_report.json` und `verified_conductivity_matrix.json` erzeugen und nur bei bestandenen deterministischen Checks `VERIFIED: parameters accepted` melden.
8. Führe danach das finale Gate aus:

   ```powershell
   uv run python -m collective_lab_validation.cli.gate_check --payload collective_lab_artifacts/verified_conductivity_matrix.json --audit collective_lab_artifacts/verified_audit_report.json --output collective_lab_artifacts/approved_gate_decision.json
   ```

9. Nur nach `APPROVED: JAX execution allowed` darfst du die Simulation starten:

   ```powershell
   uv run python -m collective_lab_validation.cli.run_jax_simulation --payload collective_lab_artifacts/verified_conductivity_matrix.json --gate collective_lab_artifacts/approved_gate_decision.json --output collective_lab_artifacts/simulation_result.json --image collective_lab_artifacts/jax_heat_result.png
   ```

10. Fasse Audit-Trail, Ablehnung, Korrekturmatrix, Verifikation, Gate und Simulation zusammen. Erfinde keine Resultate und gib keine Freigabe aufgrund frei formulierter Agententexte.

## Erwartetes visuelles Tracing

```text
Orchestrator
├── Scholar-Prime
└── Auditor-Agent
```

Die Agenten arbeiten bewusst sequenziell. Zeige im Trace die Rückgabe des Audit-Feedbacks an Scholar-Prime und die erneute Prüfung durch denselben Auditor-Agenten. Täusche keine Parallelität vor.

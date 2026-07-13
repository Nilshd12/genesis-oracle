# Das kollektive Labor – Wissenschaftliche Parameter-Extraktion und Validierung

Diese eigenständige Demo zeigt Adversarial Collaboration zwischen `Scholar-Prime` und `Auditor-Agent`. Sie ergänzt die vorhandene skalare Parameterdemo und die Fraktal-Demo, ohne beide zu verändern.

## Architektur

```text
DeepMind Science Skill literature-search-openalex
        ↓
OpenAlex / reale wissenschaftliche Quelle
        ↓
Scholar-Prime → strukturiertes 2×2-Payload
        ↓
Auditor-Agent + deterministische Python-Validatoren
        ↓
deterministisches Gate
        ↓
zweidimensionale anisotrope JAX-Wärmesimulation
```

Beim ersten Audit wird der kontrolliert injizierte Einheitenfehler abgelehnt. Das Gate blockiert JAX. Audit-Feedback geht zurück zu Scholar-Prime, derselbe Datensatz wird korrigiert und derselbe Auditor prüft erneut. JAX gehört nicht zur Korrekturschleife.

## Wissenschaftliche Quelle

- Titel: *Thermal conductivities of some novel nonlinear optical materials*
- Autor: J. Donald Beasley
- Jahr: 1994
- DOI: `10.1364/AO.33.001000`
- AgGaS2 parallel: `0.014 W/(cm*K)`
- AgGaS2 senkrecht: `0.015 W/(cm*K)`

Die Quelle beschreibt die beiden Richtungen parallel beziehungsweise senkrecht zur optischen Achse. In der Demo bilden sie die Diagonale der anisotropen Leitfähigkeitsmatrix.

## DeepMind Science Skill und Fallbacks

Der Online-Modus verwendet tatsächlich den vorhandenen Skill:

```text
science-skills/skills/literature_search_openalex/SKILL.md
science-skills/skills/literature_search_openalex/scripts/openalex_cli.py
```

Die Reihenfolge ist:

1. `literature-search-openalex` über sein vorgesehenes `uv run`-CLI,
2. klar gekennzeichneter direkter OpenAlex-API-Fallback,
3. klar gekennzeichnete lokale Fixture.

Nur Stufe 1 setzt `science_skill_used` auf `literature-search-openalex`. Der API-Fallback wird niemals als Science Skill dargestellt. OpenAlex funktioniert ohne Schlüssel im limitierten kostenlosen Pool. Ein optionaler `OPENALEX_API_KEY` darf ausschließlich lokal in einer nicht versionierten `.env` gespeichert werden; niemals in Code, Chat oder Git.

Nutzende müssen die OpenAlex-Bedingungen unter <https://developers.openalex.org/> und die Lizenz der abgerufenen Publikation prüfen. Die Benachrichtigung ist unter `.licenses/literature_search_openalex_LICENSE.txt` dokumentiert.

## Kontrollierter Fehler und Korrektur

Erstes Payload:

```json
{
  "matrix": [[0.014, 0.0], [0.0, 0.015]],
  "unit": "W/(m*K)",
  "source_unit": "W/(cm*K)"
}
```

Die Einheitenbezeichnung wurde geändert, die Zahlenwerte aber nicht. Der Auditor berechnet elementweise:

```text
1 W/(cm*K) = 100 W/(m*K)
0.014 W/(cm*K) = 1.4 W/(m*K)
0.015 W/(cm*K) = 1.5 W/(m*K)
```

Korrektes Payload:

```json
{
  "matrix": [[1.4, 0.0], [0.0, 1.5]],
  "unit": "W/(m*K)"
}
```

## Deterministische Prüfungen

- vollständiges Pydantic-Schema und strikte numerische Typen
- Titel, DOI/URL, Textausschnitt, beide Quellenwerte und Quelleneinheit
- exakt 2×2, numerisch, endlich, symmetrisch und positiv definit
- positive Diagonalwerte und quellenkonforme Null-Offdiagonale
- elementweise Einheitenumrechnung mit Toleranz
- Plausibilitätsbereich
- Korrekturhistorie nach `CORRECTED`
- keine unbelegten Matrixwerte

Das Gate wiederholt diese Prüfungen, verlangt einen `VERIFIED`-Auditorstatus und bindet die Freigabe per SHA-256-Fingerabdruck an das exakte Payload. Eine Freigabe kann nicht für ein nachträglich verändertes Payload verwendet werden.

## Lokaler Test-/Backup-Lauf

```powershell
uv sync --dev
uv run python run_collective_lab_validation_demo.py --mode demo --no-pause
```

Online:

```powershell
uv run python run_collective_lab_validation_demo.py --mode online --presentation --no-pause
```

Optionen: `--mode demo|online`, `--presentation`, `--no-pause`, `--artifacts-dir`.

Der lokale Runner reproduziert Codepfad und Audit-Trail. Er ersetzt ausdrücklich nicht die nativen Antigravity-Subagenten und gibt sich nicht als visuelles Tracing aus.

## Native Antigravity-Demo

Startpunkt ist [antigravity/orchestrator_prompt.md](antigravity/orchestrator_prompt.md). Die vollständigen manuellen Schritte stehen in [antigravity/live_demo_instructions.md](antigravity/live_demo_instructions.md).

Erwartetes Tracing:

```text
Orchestrator
├── Scholar-Prime
└── Auditor-Agent
```

Die Agenten arbeiten sequenziell: Scholar-Payload → Auditor-Ablehnung → Feedback an Scholar → korrigiertes Payload → erneuter Auditor → Gate → JAX. Diese UI-Interaktion muss in der installierten Antigravity-Anwendung manuell geprüft werden.

## Eigenständige CLIs

- `python -m collective_lab_validation.cli.scholar_extract`
- `python -m collective_lab_validation.cli.auditor_review`
- `python -m collective_lab_validation.cli.gate_check`
- `python -m collective_lab_validation.cli.run_jax_simulation`

Die Antigravity-Prompts enthalten vollständige PowerShell-Aufrufe für beide Audit-Durchläufe.

## Artefakte

Der Lauf erzeugt unter `collective_lab_artifacts`:

- `source_record.json`
- `extracted_payload.json`
- `rejected_audit_report.json`
- `blocked_gate_decision.json`
- `corrected_payload.json`
- `verified_audit_report.json`
- `approved_gate_decision.json`
- `verified_conductivity_matrix.json`
- `simulation_result.json`
- `jax_heat_result.png`

## Tests und Präsentation

```powershell
uv run pytest
```

Die Tests decken Umrechnung, Matrixfehler, Quelle/Schema, beide Auditstatus, beide Gateentscheidungen, JAX-Sperre, Simulation, PNG und vollständige Korrekturschleife ab. Der Folienabgleich steht in [presentation_alignment.md](presentation_alignment.md); das zeitlich strukturierte Skript in [presentation_script_5_minutes.md](presentation_script_5_minutes.md).

Die Demo verspricht keine absolute mathematische Unfehlbarkeit. Ihr Ziel ist robustere und nachvollziehbarere Validierung, sodass kein ungeprüftes Ergebnis die physikalische Simulation erreicht.

# Collective Lab

Collective Lab ist eine eigenständige Multi-Agenten-Demo innerhalb von `genesis-oracle`. Sie zeigt, wie ein wissenschaftlicher Parameter erst nach unabhängiger Prüfung und einer harten deterministischen Freigabe in eine JAX-Wärmesimulation gelangt.

## Architektur

```text
OpenAlex oder lokale Quellen-Fixture
  → Scholar-Prime
  → typisiertes ParameterRecord-JSON
  → Auditor-Agent + deterministische Validatoren
  → deterministisches Gate
  → JAX-Wärmesimulation

Bei Ablehnung:
Auditor-Feedback → Scholar-Prime-Korrektur → erneutes Audit → Gate → JAX
```

Die Simulation gehört nicht zur Korrekturschleife. Das JAX-Modul importiert JAX erst, nachdem ein `APPROVED`-Gateobjekt vorliegt. Die Freigabe enthält außerdem einen SHA-256-Fingerabdruck des verifizierten Parameters; eine Freigabe für einen anderen oder nachträglich veränderten Datensatz wird abgelehnt.

## Struktur

- `agents/`: Scholar-Prime und unabhängiger Auditor-Agent
- `models/`: strikte Pydantic-Datenmodelle
- `services/`: OpenAlex-/Fixture-Recherche und JSON-Artefakte
- `validation/`: Schema-, Einheiten- und Gate-Prüfung
- `simulation/`: gategeschützte eindimensionale JAX-Wärmeleitung
- `fixtures/`: klar markierte, reproduzierbare Quellen-Fixture
- `prompts/`: Rollen- und Schemaanweisungen für optionale LLM-Agenten
- `../run_collective_lab_demo.py`: zentrale CLI-Orchestrierung
- `../tests/`: automatisierte Unit- und Ablauftests

## Installation und Start

Die bestehende uv-Umgebung und die vorhandenen Laufzeitabhängigkeiten JAX und Pydantic werden verwendet.

```powershell
uv sync --dev
uv run python run_collective_lab_demo.py --mode demo --no-pause
```

Für eine Präsentation kann `--no-pause` entfallen. An einem interaktiven Terminal wartet die Demo dann zwischen Hauptschritten auf Enter. Ein alternatives Artefaktverzeichnis wird mit `--artifacts-dir <pfad>` gewählt.

## Demo-Modus und injizierter Fehler

`--mode demo` ist der Standard und lädt eine lokale Fixture, die ausdrücklich nicht als aktuelle Online-Recherche ausgegeben wird. Sie basiert auf J. Donald Beasleys Veröffentlichung „Thermal conductivities of some novel nonlinear optical materials“ (1994), DOI `10.1364/AO.33.001000`.

Die Quelle enthält `0.015 W/(cm*K)`. Für die reproduzierbare Vorführung übernimmt Scholar-Prime absichtlich zunächst `0.015 W/(m*K)`. Das Terminal kennzeichnet diese kontrollierte Injektion. Der Auditor berechnet deterministisch:

```text
1 W/(cm*K) = 100 W/(m*K)
0.015 W/(cm*K) = 1.5 W/(m*K)
```

Er lehnt den ersten Payload ab, das Gate blockiert JAX, und Scholar-Prime erstellt anhand des Auditberichts einen korrigierten Datensatz samt Historie. Erst nach erneutem Audit und Gate-Freigabe startet JAX.

## Online-Modus

```powershell
uv run python run_collective_lab_demo.py --mode online --no-pause
```

Der Online-Modus fragt OpenAlex tatsächlich ab, rekonstruiert den Abstract und akzeptiert den Parameter nur, wenn der Quellenwert im gelieferten Abstract gefunden wird. Er injiziert keinen Demofehler. Bei Netzwerk-, API- oder Evidenzfehlern wird transparent auf die lokale Fixture zurückgefallen; Terminal und `source_record.json` kennzeichnen den tatsächlichen Modus. Es werden keine API-Schlüssel benötigt.

## Rollen und Freigabe

Scholar-Prime dokumentiert Quelle, Originalwert und -einheit und kann nur anhand eines strukturierten `REJECTED`-Berichts korrigieren. Der Auditor kontrolliert Schema, Typen, Evidenz, bekannte Einheiten, Umrechnung, Positivität, Plausibilitätsbereich und gegebenenfalls die Korrekturhistorie. Das Gate wiederholt die objektiven Kernprüfungen unabhängig und akzeptiert ausschließlich `VERIFIED`.

## Artefakte

Jeder Lauf schreibt menschenlesbare JSON-Dateien nach `demo_artifacts` (oder in das mit `--artifacts-dir` angegebene Verzeichnis):

- `source_record.json`
- `extracted_parameters.json`
- `rejected_audit_report.json`
- `blocked_gate_decision.json`
- `corrected_parameters.json`
- `verified_audit_report.json`
- `gate_decision.json`
- `simulation_result.json`

Nur diese bekannten Dateien werden beim nächsten Lauf kontrolliert ersetzt. Der Online-Modus ohne Ablehnung erzeugt naturgemäß keinen Ablehnungs- oder Korrekturbericht.

## Tests

```powershell
uv run pytest
```

Die Tests decken Umrechnung, falsche und negative Werte, fehlende Quellen, ungültige Schemas, beide Gate-Ausgänge, die Runtime-Sperre der JAX-Simulation und den vollständigen Demoablauf ab.

## Präsentationsablauf

1. Quellenextraktion und erstes JSON zeigen.
2. Kontrollierten Einheitenfehler und `REJECTED` erläutern.
3. Sichtbare Gate-Blockierung hervorheben: JAX läuft nicht.
4. Audit-Feedback und dokumentierte Korrektur zeigen.
5. `VERIFIED` und datensatzgebundene Gate-Freigabe zeigen.
6. Anfangs-/Endprofil und Temperaturstatistik der Simulation zeigen.

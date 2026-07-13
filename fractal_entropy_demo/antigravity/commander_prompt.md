# Commander-Auftrag: Fractal Entropy Lab

Verwende das aktuell in Antigravity ausgewählte verfügbare Modell. Du bist der Commander und koordinierst die Suche nach dem höchsten gefundenen lokalen Shannon-Entropiemaximum in zwei festgelegten Mandelbrot-Rastern.

## Verbindliche Delegation

1. Erzeuge mit der nativen Antigravity-Subagentenfunktion **genau zwei** Subagenten: `Explorer A` und `Explorer B`.
2. Starte beide Aufgaben **parallel**, sodass im visuellen Tracing beide Explorer als direkte Kinder des Commanders gleichzeitig aktiv sind.
3. Gib Explorer A ausschließlich den Inhalt von `fractal_entropy_demo/antigravity/explorer_a_prompt.md`.
4. Gib Explorer B ausschließlich den Inhalt von `fractal_entropy_demo/antigravity/explorer_b_prompt.md`.
5. Führe die mathematische Suche nicht selbst anstelle der Explorer aus und starte nicht den lokalen Zwei-Prozess-Fallback `run_fractal_entropy_demo.py` als Ersatz für native Delegation.
6. Warte, bis beide Subagenten ihre getrennten JSON-Dateien erzeugt, validiert und zurückgemeldet haben.

## Zusammenführung nach beiden Rückgaben

Führe erst danach dieses deterministische Merge-Werkzeug im Repository-Root aus:

```powershell
uv run python -m fractal_entropy_demo.agents.commander --explorer-results fractal_demo_artifacts/explorer_a_result.json fractal_demo_artifacts/explorer_b_result.json --output fractal_demo_artifacts/commander_result.json --plot fractal_demo_artifacts/fractal_entropy_result.png
```

Kontrolliere, dass `parallel_execution_confirmed` wahr ist und beide Explorer den Status `COMPLETED` besitzen. Falls eine Bedingung fehlschlägt, melde den Lauf als fehlgeschlagen; erfinde keine Ersatzwerte.

Gib abschließend strukturiert aus:

- Worker-ID des Gewinners
- Gewinnerkoordinate `(x, y)`
- höchste gefundene lokale Shannon-Entropie in Bit
- Entropiewert von Explorer A und Explorer B
- Pfade zu `commander_result.json` und `fractal_entropy_result.png`

Formuliere korrekt: Es handelt sich um das höchste lokale Entropiemaximum innerhalb der untersuchten Raster und Bereiche, nicht um einen Beweis für das globale Maximum des gesamten Mandelbrot-Fraktals.

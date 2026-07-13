# Native Antigravity-Live-Demo

Im Repository wurde keine aktuelle, unterstützte Antigravity-Konfigurationsdatei gefunden. Deshalb werden keine proprietären Konfigurationsnamen erfunden. Die folgenden Schritte verwenden funktionale Bezeichnungen; die sichtbaren Menütexte können je nach installierter Antigravity-Version abweichen.

## Vorbereitung

1. Im PowerShell-Terminal im Repository-Root einmal `uv sync --dev` ausführen.
2. Optional den lokalen technischen Fallback prüfen:

   ```powershell
   uv run python run_fractal_entropy_demo.py --presentation --no-pause
   ```

3. Beachten: Dieser lokale Lauf prüft Mathematik, Prozesse und Artefakte, erzeugt aber **kein** natives Antigravity-Agententracing.

## Live-Ablauf

1. Das Repository `genesis-oracle` als Workspace in Antigravity öffnen.
2. Die Ansicht für das visuelle Agent-/Task-Tracing öffnen. Falls die Version keinen identischen Menünamen besitzt, die Ansicht wählen, die Eltern-Kind-Agenten und ihre Laufzeiten darstellt.
3. Einen Commander-Agenten im Workspace erstellen oder auswählen. Das aktuell in Antigravity ausgewählte verfügbare Modell beibehalten.
4. Den vollständigen Inhalt von `fractal_entropy_demo/antigravity/commander_prompt.md` als Auftrag einfügen und starten.
5. Im visuellen Tracing kontrollieren, dass exakt diese Struktur erscheint:

   ```text
   Commander
   ├── Explorer A
   └── Explorer B
   ```

6. Kontrollieren, dass beide Explorer direkte Kinder sind und zeitlich parallel aktiv werden. Wenn der Commander sie nacheinander startet, Lauf abbrechen und die Parallelitätsanweisung erneut ausdrücklich erteilen.
7. Im integrierten Terminal beobachten, dass beide Explorer ihre eigenen JAX-CLI-Befehle ausführen. Die Ausgaben `JAX-Berechnung gestartet` sollen von beiden Workern erscheinen.
8. Im Tracing die beiden strukturierten Rückgaben verfolgen. Beide JSON-Dateien müssen `COMPLETED` melden.
9. Zeigen, wie der Commander nach beiden Rückgaben das deterministische Merge-Werkzeug aufruft und Gewinner, Koordinate sowie Entropie ausgibt.
10. `fractal_demo_artifacts/commander_result.json` und `fractal_demo_artifacts/fractal_entropy_result.png` in Antigravity beziehungsweise der Workspace-Dateiansicht öffnen.

## Manuell zu bestätigende Punkte

- Die native Eltern-Kind-Struktur ist sichtbar.
- Es existieren genau zwei Explorer-Subagenten.
- Beide Subagenten sind zeitlich parallel aktiv.
- Jeder Subagent führt nur seinen eigenen Bereich aus.
- Der Commander wartet auf beide Rückgaben und rechnet nicht an ihrer Stelle.

Diese UI-/Tracing-Eigenschaften können durch die Python-Tests nicht verifiziert werden. Automatisiert geprüft werden der echte parallele Prozessstart, überlappende JSON-Zeitintervalle, JAX-Berechnung, Schema, Merge-Logik und Grafik.

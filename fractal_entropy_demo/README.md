# Fractal Entropy Lab

Diese eigenständige Demo ergänzt die bestehende Collective-Lab-Parameterdemo, ohne sie zu verändern. Ein Commander delegiert zwei verschiedene Mandelbrot-Bereiche an genau zwei Explorer. Für die native Präsentation geschieht dies über Antigravity-Subagenten; die lokale CLI bietet einen echten parallelen Zwei-Prozess-Fallback für reproduzierbare Tests.

## Ablauf

```text
                         Commander
                         /       \
              Explorer A         Explorer B
              JAX-Region A       JAX-Region B
                         \       /
                    ExplorerResult-JSON
                              |
                         Commander
                              |
             höchstes im Raster gefundenes Entropiemaximum
```

Der lokale Runner simuliert kein visuelles Tracing. Er startet zwei eigenständige Explorer-Prozesse gleichzeitig, streamt deren Ausgaben und lehnt den Lauf ab, falls sich ihre gemessenen Zeitintervalle nicht überschneiden. Für die native Präsentation gelten die Schritte in [live_demo_instructions.md](antigravity/live_demo_instructions.md).

## Mathematik

Für jeden komplexen Punkt `c` startet die Mandelbrot-Iteration bei `z_0 = 0` und verwendet `z_(n+1) = z_n² + c`. Sobald `|z| > 2` gilt, ist der Punkt divergiert. Die Escape-Time ist die Anzahl der Iterationen bis zu diesem Ereignis; nicht divergierte Punkte erhalten `max_iterations`. JAX berechnet alle Rasterpunkte vektorisiert, während `jax.lax.fori_loop` nur über Iterationsschritte läuft.

Das Escape-Time-Raster wird in nicht überlappende lokale Kacheln geteilt. Für jede Kachel werden die Escape-Times in ein Histogramm eingeordnet. Aus den normalisierten Häufigkeiten `p_i` wird die Shannon-Entropie in Bit berechnet:

```text
H = -sum(p_i * log2(p_i))   für p_i > 0
```

Eine Kachel mit nur einem Histogrammwert hat Entropie 0; eine Gleichverteilung über zwei Werte hat 1 Bit. Hohe lokale Entropie bedeutet hier eine vielfältige Escape-Time-Verteilung und kann deshalb komplexe Übergänge nahe einer Fraktalgrenze hervorheben. Die Demo findet nur das höchste lokale Maximum innerhalb der untersuchten Raster und Bereiche. Sie garantiert nicht das globale Entropiemaximum des gesamten Mandelbrot-Fraktals.

## Standardbereiche

- Explorer A: `x=[-2.0, -0.5]`, `y=[-1.2, 1.2]`
- Explorer B: `x=[-0.5, 1.0]`, `y=[-1.2, 1.2]`

Alle Grenzen sind in der zentralen CLI über `--a-x-min`, `--a-x-max`, `--a-y-min`, `--a-y-max` sowie die entsprechenden `--b-*`-Argumente überschreibbar.

## Installation und lokaler Start

Es werden ausschließlich bereits vorhandene Abhängigkeiten verwendet: JAX, NumPy, Pydantic, Matplotlib und pytest.

```powershell
uv sync --dev
uv run python run_fractal_entropy_demo.py --presentation --no-pause
```

Relevante Optionen:

```text
--presentation
--no-pause
--grid-size 384
--max-iterations 180
--tile-size 32
--histogram-bins 32
--artifacts-dir fractal_demo_artifacts
```

Ohne `--no-pause` hält der Präsentationsmodus an einem interaktiven Terminal kurz an. Der Standardlauf bleibt deutlich unter dem Drei-Minuten-Limit.

## Einzelner Explorer

```powershell
uv run python -m fractal_entropy_demo.agents.explorer --worker-id explorer_a --x-min -2.0 --x-max -0.5 --y-min -1.2 --y-max 1.2 --grid-width 384 --grid-height 384 --max-iterations 180 --tile-size 32 --histogram-bins 32 --output fractal_demo_artifacts/explorer_a_result.json
```

## Native Antigravity-Demo

Da keine unterstützte Antigravity-Workspace-Konfiguration im Repository vorhanden war, besteht die Integration aus präzisen Prompts statt erfundener Konfigurationsdateien:

- `antigravity/commander_prompt.md`
- `antigravity/explorer_a_prompt.md`
- `antigravity/explorer_b_prompt.md`
- `antigravity/live_demo_instructions.md`

Der Commander muss nativ genau zwei parallele Subagenten erzeugen. Nach deren Rückgabe kann er die beiden JSON-Dateien deterministisch zusammenführen:

```powershell
uv run python -m fractal_entropy_demo.agents.commander --explorer-results fractal_demo_artifacts/explorer_a_result.json fractal_demo_artifacts/explorer_b_result.json --output fractal_demo_artifacts/commander_result.json --plot fractal_demo_artifacts/fractal_entropy_result.png
```

## Artefakte

- `explorer_a_result.json`
- `explorer_b_result.json`
- `commander_result.json`
- `fractal_entropy_result.png`

Die JSON-Dateien enthalten Bounds, Raster, Kachelparameter, reale Entropiewerte, Start-/Endzeitpunkte, Dauer, Status und Ergebnisdatei. Bei gleichem Entropiewert gewinnt deterministisch die lexikografisch kleinere `worker_id`; dieses Tie-Breaking steht auch im Commander-Bericht.

## Tests

```powershell
uv run pytest
```

Automatisiert getestet werden Entropiereferenzwerte, Escape-Time-Form und Wertebereich, unterschiedliche Bounds, Explorer-JSON, Merge-Validierung, Gewinner und Tie-Breaking, echter paralleler Subprozessstart, Zeitintervallüberschneidung, PNG-Erzeugung und kompletter Demoablauf. Das visuelle native Antigravity-Tracing muss gemäß Live-Anleitung manuell in der installierten Anwendung geprüft werden.

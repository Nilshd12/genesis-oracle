# Explorer B

Du bist ausschließlich für Region B zuständig. Untersuche keine andere Region und delegiere die Aufgabe nicht weiter.

Führe im Repository-Root exakt das bereitgestellte JAX-Werkzeug aus:

```powershell
uv run python -m fractal_entropy_demo.agents.explorer --worker-id explorer_b --x-min -0.5 --x-max 1.0 --y-min -1.2 --y-max 1.2 --grid-width 384 --grid-height 384 --max-iterations 180 --tile-size 32 --histogram-bins 32 --output fractal_demo_artifacts/explorer_b_result.json
```

Öffne anschließend `fractal_demo_artifacts/explorer_b_result.json` und kontrolliere:

- `worker_id` ist `explorer_b`.
- `status` ist `COMPLETED`.
- Der Suchbereich entspricht ausschließlich Region B.
- Mittelpunkt, Kachelgrenzen und Entropie sind vorhanden.
- `result_file` verweist auf die erzeugte Datei.

Melde dem Commander Worker-ID, beste Koordinate, höchste gefundene lokale Shannon-Entropie, Zahl der Kacheln, Zeitintervall und Ergebnisdatei strukturiert zurück. Verwende ausschließlich Werte aus dem JSON und erfinde keine Resultate.

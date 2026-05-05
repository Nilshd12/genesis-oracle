# Genesis Oracle

## Projektüberblick

`genesis-oracle` ist ein Projekt zur Simulation physikalischer Systeme und zur Erkennung von Anomalien in Zeitreihendaten mittels Deep Learning. Das Projekt verfolgt das Ziel, dynamische Signalverläufe zu modellieren und unerwartete Abweichungen durch künstliche neuronale Netze automatisch zu identifizieren.

Im Fokus steht ein simuliertes physikalisches Signal, genauer gesagt der Output eines RC-Filters (Tiefpass), der mit künstlich eingefügten Störsignalen (Spikes) versehen wird. Zur Erkennung dieser Anomalien wird ein Autoencoder eingesetzt, der ausschließlich auf das Normalverhalten trainiert wird und dadurch signifikante Abweichungen erkennen kann.

## Technischer Hintergrund

**RC-Filter und Zeitreihen-Generierung**  
Das Projekt simuliert das Lade- und Entladeverhalten eines RC-Tiefpassfilters. Die erzeugten Zeitreihen spiegeln die natürliche Dynamik des Systems wider. Für Testzwecke wird eine künstliche Anomalie – ein plötzlicher, hochfrequenter Spike – in das Signal injiziert, um zu evaluieren, wie gut das Modell auf unerwartete Signaländerungen reagiert.

**Anomalieerkennung mit Autoencodern**  
Ein Autoencoder ist ein neuronales Netz, das darauf trainiert wird, seine Eingabedaten zu komprimieren (Encoder) und anschließend möglichst exakt wiederherzustellen (Decoder). Wird das Modell ausschließlich mit normalen, störungsfreien RC-Filter-Daten trainiert, lernt es die zugrundeliegende physikalische Systemdynamik.

**Reconstruction Loss (MAE) als Anomalieindikator**  
Tritt eine Anomalie auf, weicht das Eingangssignal so stark vom Normalzustand ab, dass der Autoencoder es nicht mehr korrekt rekonstruieren kann. Der Fehler zwischen dem Originalsignal und der Rekonstruktion (Reconstruction Loss) steigt drastisch an. In diesem Projekt wird der Mean Absolute Error (MAE) gemessen; überschreitet dieser einen definierten Schwellenwert, wird das Signal erfolgreich als Anomalie markiert.

**Warum Conv1D für Zeitreihen sinnvoll ist**  
Während klassische, vollständig verbundene Schichten (`Dense`) ein Zeitfenster als starren Vektor betrachten, schieben 1D-Faltungsschichten (`Conv1D`) Filter über die Zeitachse. Dadurch bleiben lokale zeitliche Muster, Trends und zyklische Eigenschaften erhalten. Diese Translationsinvarianz macht die `Conv1D`-Architektur deutlich robuster und leistungsfähiger bei der Verarbeitung von physikalischen Zeitreihen als klassische Dense-Netzwerke.

## Projektstruktur

Die Struktur des Projekts ist wie folgt organisiert:

- `src/data_generator.py`: Skript zur Erzeugung der RC-Filter-Simulationsdaten inklusive der künstlichen Anomalien.
- `src/oracle_setup.py`: Überprüft das Keras 3 Setup und stellt sicher, dass das JAX-Backend korrekt initialisiert ist.
- `src/architecture.py`: Die ursprüngliche Implementierung des Autoencoders mit vollständigen `Dense`-Schichten.
- `src/architecture_conv1d.py`: Die refaktorierte und leistungsstärkere Autoencoder-Architektur, basierend auf 1D-Faltungsschichten (`Conv1D` / `Conv1DTranspose`).
- `docs/index.md`: Die Dokumentation für die GitHub Pages, die einen Überblick über die Experimente und Ergebnisse gibt.
- `public_data/data_feed.png`: Visualisierung der generierten Trainingsdaten des RC-Filters. *(sofern generiert)*
- `public_data/anomaly_detection.png` & `docs/anomaly_detection.png`: Plot der erfolgreichen Anomalieerkennung durch den Reconstruction Loss.
- `pyproject.toml`: Konfigurationsdatei für das Projekt und die zugehörigen Metadaten.
- `uv.lock`: Lockfile zur Sicherstellung deterministischer und reproduzierbarer Umgebungen durch den Paketmanager `uv`.

## Installation und Setup

Voraussetzungen:
- **Python** (>=3.13)
- **Git**
- **uv** (Ein extrem schneller Python Paket- und Projektmanager)

1. **Repository klonen**  
   Klonen Sie das Projekt lokal auf Ihren Rechner:
   ```bash
   git clone <Ihre-Repository-URL>
   cd genesis-oracle
   ```

2. **Umgebung aufsetzen und Abhängigkeiten installieren**  
   Das Projekt nutzt `uv` zur Verwaltung der Abhängigkeiten und der virtuellen Umgebung.
   ```bash
   uv sync
   ```
   Dieser Befehl erstellt automatisch die virtuelle Umgebung und installiert alle benötigten Pakete (`jax`, `keras`, `matplotlib`, `numpy`, `scipy`) basierend auf der `pyproject.toml` und der `uv.lock`.

## Ausführung

Führen Sie die Skripte über `uv run` aus, um sicherzustellen, dass sie in der korrekten virtuellen Umgebung laufen.

**Keras/JAX-Setup testen:**
```bash
uv run python src/oracle_setup.py
```

**Datengenerator ausführen:**  
```bash
uv run python src/data_generator.py
```

**Dense-Architektur testen:**  
```bash
uv run python src/architecture.py
```

**Conv1D-Architektur testen:**  
```bash
uv run python src/architecture_conv1d.py
```

## Ergebnisse

Die wesentlichen Ergebnisse des Experiments sind in Form von Plots gesichert:

- **RC-Filter Signal (`data_feed.png`):** Zeigt den Lade- und Entladevorgang des Tiefpassfilters und dient als Ground Truth für das Training.
- **Ergebnis der Erkennung (`anomaly_detection.png`):** Zeigt das rekonstruierte Signal und den berechneten MAE (Reconstruction Loss). Der Plot veranschaulicht eindrucksvoll, dass der Fehler im normalen Signalverlauf sehr gering ist, aber beim Eintreten des simulierten Spikes signifikant über die rote Schwellenwert-Linie ausschlägt. Die Anomalie wurde korrekt erkannt.

## Hinweise zur Versionierung

- **`data/` Ordner:** Um die Größe des Repositories gering zu halten und Datenschutz sowie Sicherheit zu wahren, werden generierte Rohdaten im Ordner `data/` lokal gespeichert und über die `.gitignore` bewusst **nicht** versioniert.
- **`uv.lock`:** Diese Datei wird im Versionskontrollsystem mitgeführt, um sicherzustellen, dass das Projekt auf verschiedenen Systemen absolut reproduzierbar bleibt und dieselben Abhängigkeitsversionen verwendet werden.

## GitHub Pages

Eine visuelle und zusammenfassende Dokumentation der Experimente (inkl. Convolutional Horizon) ist über GitHub Pages verfügbar. Die zugrundeliegende Struktur finden Sie in der Datei `docs/index.md`, welche automatisch aus den Ergebnissen der Anomalieerkennung generiert und via GitHub Actions gerendert wird.

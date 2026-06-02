# Swarm Stress Report: Monte-Carlo-Simulation & Profiling

Dieses Dokument fasst die Ergebnisse der Monte-Carlo-Simulation und des Profilings für das Projekt zusammen. Es kombiniert den Stresstest der log-normalen Kostenverteilung (Subagent-Alpha) mit der Laufzeitanalyse (Subagent-Beta).

---

## Subagent-Alpha: Stress-Test der log-normalen Kostenverteilung

Diese Analyse untersucht die Auswirkung der Standardabweichung ($\sigma$) der log-normalen Produktionsanlagenkosten auf den Value-at-Risk (VaR 95%) in der Monte-Carlo-Simulation von `src/monte_carlo.py`.

### Simulationslogik & Methodik
In `src/monte_carlo.py` sind die Einnahmen (Revenue) wie folgt modelliert:
$$\text{Revenue} = (\text{Demand} \times 150.0) - \text{Cost} \times (1.0 - \text{Penalty Rate})$$
wobei:
- $\text{Demand} \sim \mathcal{N}(\mu=1000, \sigma=150)$
- $\text{Cost} = \exp(\text{Log-Cost})$ mit $\text{Log-Cost} \sim \mathcal{N}(\mu=5.5, \sigma_{\text{cost}})$
- $\text{Penalty Rate} \sim \mathcal{U}(0.05, 0.25)$

Der standardmäßige $\sigma_{\text{cost}}$-Wert (Kosten-Sigma) beträgt **0,3**.

Wir haben ein Parameter-Sweep-Skript unter `scratch/sweep_sigma.py` erstellt, um $\sigma_{\text{cost}}$ schrittweise zu erhöhen und den Wert zu bestimmen, bei dem der Value-at-Risk 95% (5. Perzentil der Einnahmen) unter Null fällt.

### Testergebnisse

#### Grober Sweep ($\sigma = 0,3$ bis $4,0$)
| Sigma ($\sigma$) | Erwartete Einnahmen | Value-at-Risk 95% (VaR 95%) | Status |
|---|---|---|---|
| **0,300** (Baseline) | 149.751,72 | 112.734,47 | Über Null |
| 0,500 | 149.733,66 | 112.715,05 | Über Null |
| 1,000 | 149.627,02 | 112.595,98 | Über Null |
| 1,500 | 149.332,66 | 112.188,72 | Über Null |
| 2,000 | 148.456,77 | 110.624,71 | Über Null |
| 2,500 | 145.252,73 | 106.694,98 | Über Null |
| 3,000 | 132.772,56 | 98.868,55 | Über Null |
| 3,500 | 71.026,35 | 80.125,98 | Über Null |
| 3,800 | -65.402,70 | 39.655,66 | Über Null |
| **4,000** | -282.180,97 | **-333,83** | **Unter Null** |

#### Feiner Sweep ($\sigma = 3,80$ bis $4,04$)
Hier ist die detaillierte Tabelle um den kritischen Punkt (Breaking Point) herum:

| Sigma ($\sigma$) | Erwartete Einnahmen | Value-at-Risk 95% (VaR 95%) | Status |
|---|---|---|---|
| 3,800 | -65.402,70 | 39.655,66 | Über Null |
| 3,850 | -105.819,16 | 30.962,24 | Über Null |
| 3,900 | -154.256,72 | 21.464,62 | Über Null |
| 3,950 | -212.373,23 | 11.059,92 | Über Null |
| 3,960 | -225.328,03 | 8.926,83 | Über Null |
| 3,970 | -238.767,14 | 6.646,32 | Über Null |
| 3,980 | -252.709,23 | 4.354,08 | Über Null |
| 3,990 | -267.173,78 | 2.047,15 | Über Null |
| **4,000** | **-282.180,97** | **-333,83** | **Unter Null** |
| 4,010 | -297.752,25 | -2.729,84 | Unter Null |
| 4,020 | -313.908,38 | -5.141,31 | Unter Null |
| 4,030 | -330.673,56 | -7.570,63 | Unter Null |
| 4,040 | -348.069,88 | -10.014,19 | Unter Null |

---

## Subagent-Beta: Profiler

In diesem Abschnitt werden die Leistungskennzahlen der Monte-Carlo-Simulation analysiert. Das Skript `src/monte_carlo.py` wurde zweimal ausgeführt, um die JAX-Kompilierungszeit und die reine Ausführungszeit der Berechnungen zu erfassen.

### Extrahierte Simulationsergebnisse (Baseline $\sigma = 0,3$)
- **Erwartete Einnahmen (Expected Revenue):** 149.751,72
- **Value-at-Risk 95% Schwellenwert (VaR 95%):** 112.734,47

### Gemessene Laufzeitkomponenten
Die folgende Tabelle zeigt die Ausführungszeiten aus zwei aufeinanderfolgenden Durchläufen:

| Durchlauf | JAX Warm-up / Kompilierungszeit | JAX Zweite Ausführungszeit (Warm Run) |
|---|---|---|
| **Durchlauf 1** | 0,619 Sekunden | 0,122 Sekunden |
| **Durchlauf 2** | 0,615 Sekunden | 0,121 Sekunden |

### Analyse der Laufzeitunterschiede (Kompilierung vs. Ausführung)
Der signifikante Laufzeitunterschied zwischen der ersten Ausführung (Warm-up) und der zweiten Ausführung (Warm Run) lässt sich durch die Funktionsweise von JAX erklären:

1. **JIT-Kompilierung (Just-In-Time):** Beim ersten Aufruf einer mit `@jax.jit` dekorierten Funktion (`run_monte_carlo`) analysiert JAX den Python-Code und kompiliert ihn mithilfe des **XLA-Compilers (Accelerated Linear Algebra)** in optimierten, maschinennah ausgeführten Code. Dieser Kompilierungsprozess benötigt einmalig Zeit (ca. 0,61 - 0,62 Sekunden).
2. **Warm Run:** Bei der zweiten Ausführung wird die bereits kompilierte Version direkt aus dem Cache aufgerufen, wodurch der Kompilierungsoverhead vollständig entfällt. Dies führt zu einer Reduktion der reinen Berechnungszeit auf ca. 0,12 Sekunden (eine Beschleunigung um ca. den Faktor 5).

---

## Gesamtfazit (Alpha & Beta)

Die Zusammenführung der Ergebnisse von Subagent-Alpha und Subagent-Beta liefert folgende Kernpunkte:

1. **Effizienz der Simulations-Pipeline:** JAX ermöglicht durch die Kombination aus Vektorisierung (`jax.vmap`) und JIT-Kompilierung eine extrem performante Simulation von 1.000.000 Pfaden. Nach der einmaligen Kompilierungszeit von ca. 0,62 s dauert die eigentliche Simulation nur noch rund 0,12 s. Diese hohe Geschwindigkeit ist essenziell für die Durchführung von rechenintensiven Parameter-Sweeps.
2. **Kritisches Risikoprofil (Breaking Point):** Während das System bei der Standardabweichung von $\sigma = 0,3$ äußerst stabile, positive Erträge (VaR 95% von 112.734,47) aufweist, stellt eine Erhöhung der Kostenvariabilität ein erhebliches wirtschaftliches Risiko dar. Bei **$\sigma \approx 4,00$** bricht das Modell zusammen und der VaR 95% fällt unter Null ($-333,83$).
3. **Risiko-Asymmetrie:** Aufgrund der Eigenschaften der Log-Normalverteilung der Kosten drückt eine höhere Standardabweichung die erwarteten Einnahmen (Expected Revenue) viel früher ins Negative (bereits bei $\sigma \approx 3,78$) als den VaR 95%. Dies liegt daran, dass seltene, aber astronomisch hohe Kosten den Erwartungswert stark dominieren, während die Einnahmen im 5. Perzentil (VaR 95%) erst bei extremen Sigmas ($\ge 4,00$) vollständig aufgezehrt werden.

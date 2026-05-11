# Ascension Report

## Exercise 2: JAX Speedup

Die klassische NumPy-Simulation wurde zuerst lokal als sequenzielle Ausgangsbasis ausgeführt. Danach wurde dieselbe Schwarm-Simulation mit JAX umgesetzt. Dabei wurden `vmap` zur Vektorisierung vieler Oszillatoren und `jit` zur Kompilierung mit XLA verwendet. Die JAX-Version wurde anschließend in Google Colab auf einer GPU ausgeführt.

- Legacy-Simulationszeit: `1.1252140998840332` Sekunden
- JAX Warm-up- / Kompilierungszeit: `0.25394392013549805` Sekunden
- JAX-Zeit im zweiten Lauf: `0.012471199035644531` Sekunden

Speedup-Faktor:

```text
Legacy Time / JAX Second Run Time = 1.1252140998840332 / 0.012471199035644531 = 90.23

Der Speedup-Faktor beträgt damit ungefähr 90.23. Die JAX-Version war im zweiten Lauf also ungefähr 90-mal schneller als die klassische NumPy-Version.

Der erste Lauf einer mit jit kompilierten JAX-Funktion ist langsamer, weil JAX die Funktion zunächst analysieren und mit XLA kompilieren muss. Dieser Vorgang wird auch als Tracing oder Warm-up bezeichnet. Beim zweiten Lauf kann JAX den bereits kompilierten Berechnungsgraphen wiederverwenden, wodurch die eigentliche Simulation deutlich schneller ausgeführt wird.
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

## Exercise 3: Gradient-Based Projectile Optimization

Für die Optimierungsaufgabe wurde eine reine JAX-Funktion `projectile_loss(v_initial)` erstellt. Diese berechnet die Entfernung eines Projektils nach 5 Sekunden und vergleicht sie mit der Zielentfernung von `150.0` Metern. Der Fehler wird als Mean Squared Error berechnet.

Mit `jax.grad(projectile_loss)` wurde automatisch eine Gradientenfunktion erzeugt. Diese gibt an, wie stark und in welche Richtung sich der Fehler verändert, wenn die Anfangsgeschwindigkeit verändert wird. Anschließend wurde eine einfache Gradient-Descent-Schleife verwendet, um die Anfangsgeschwindigkeit schrittweise zu optimieren.

Die optimierte Anfangsgeschwindigkeit lag bei ungefähr `29.999980926513672` m/s. Der theoretisch ideale Wert beträgt `30.0` m/s, da das Projektil bei konstanter Geschwindigkeit in 5 Sekunden genau `150.0` Meter zurücklegen muss.

`jax.grad` unterscheidet sich grundlegend von finiten Differenzen, weil JAX die Ableitung automatisch aus dem Berechnungsgraphen der Funktion bestimmt. Bei finiten Differenzen müsste man die Funktion mehrfach mit leicht veränderten Eingabewerten ausführen und daraus nur eine numerische Näherung der Steigung berechnen. `jax.grad` liefert dagegen eine automatische, präzisere und effizientere Ableitung der definierten mathematischen Funktion.

## Exercise 4: Flax and Explicit State Management

Für die Flax-Aufgabe wurde mit Hilfe eines Agenten ein einfaches Multi-Layer Perceptron mit `flax.linen` erstellt. Das Modell definiert dabei nur die Architektur, also die mathematischen Operationen der Schichten. Die Gewichte werden nicht automatisch im Modellobjekt gespeichert, sondern explizit mit `model.init(...)` erzeugt.

Der wichtigste Unterschied zu Keras liegt im State Management. In Keras sind Modelle typischerweise zustandsbehaftet: Nach der Initialisierung oder dem ersten Aufruf liegen die Gewichte direkt im Modellobjekt, zum Beispiel in `model.weights`. In Flax ist das Modell dagegen zustandsloser aufgebaut. Die Parameter werden als separates Objekt gespeichert und beim Forward Pass explizit mit `model.apply(variables, input_data)` übergeben.

Diese Trennung passt gut zum JAX-Paradigma, weil der Zustand nicht versteckt im Objekt liegt, sondern als klar sichtbare Eingabe verwendet wird. Dadurch lassen sich Funktionen besser mit JAX-Transformationen wie `jit`, `grad` und `vmap` kombinieren.
# Präsentationsskript – 5 Minuten

## 0:00–2:00 – Folien 1 bis 4

**Folie 1 – Problem des Single-Agenten (ca. 30 Sekunden)**  
„Sprachmodelle können wissenschaftlich plausibel klingende, aber physikalisch falsche Parameter erzeugen. Besonders kritisch sind Einheitenfehler: Titel, DOI und Zahlenwert können korrekt aussehen, während die physikalische Bedeutung falsch übertragen wurde.“

**Folie 2 – Funktionale Trennung (ca. 30 Sekunden)**  
„Unsere Antwort ist nicht ein größerer Prompt. Wir trennen Recherche, Prüfung und Ausführung. Scholar-Prime darf produzieren, aber nicht selbst freigeben. Der Auditor darf widersprechen, aber JAX nicht eigenmächtig starten. Das deterministische Gate entscheidet anhand objektiver Regeln.“

**Folie 3 – Adversarial Collaboration (ca. 30 Sekunden)**  
„Scholar-Prime produziert. Der Auditor widerspricht. Das Gate entscheidet. Diese Rollenverteilung ähnelt einem wissenschaftlichen Peer-Review: Der zweite Agent soll nicht höflich bestätigen, sondern aktiv nach Fehlern suchen.“

**Folie 4 – Audit-Trail in Antigravity (ca. 30 Sekunden)**  
„Antigravity zeigt Scholar-Prime und Auditor-Agent als getrennte native Subagenten. Im Trace können wir Quelle, strukturiertes JSON, Ablehnung, Audit-Feedback, Korrektur und erneute Prüfung nachvollziehen. Die Agenten laufen bewusst sequenziell, weil der Auditor das Scholar-Ergebnis benötigt.“

## 2:00–4:30 – Live-Demo

**Recherche und fehlerhaftes Payload (ca. 35 Sekunden)**  
„Scholar-Prime verwendet den installierten OpenAlex Science Skill. Die reale Quelle berichtet für AgGaS2 parallel 0,014 und senkrecht 0,015 Watt pro Zentimeter-Kelvin. Für das reproduzierbare Streit-Szenario wird transparent ein typischer Übertragungsfehler injiziert: Die Einheit wird auf Meter geändert, die Zahlen bleiben unverändert.“

**Ablehnung und Blockierung (ca. 35 Sekunden)**  
„Der Auditor-Agent fängt das JSON ab. Zusätzlich zum Agentenurteil prüfen Python-Validatoren Schema, Quelle, Matrixform, Symmetrie, positive Definitheit und jede Einheitenumrechnung. Das Ergebnis ist `REJECTED: unit conversion mismatch`. Das Gate blockiert, und JAX wurde nachweislich nicht gestartet.“

**Korrekturschleife (ca. 40 Sekunden)**  
„Das Audit-Feedback geht zurück an Scholar-Prime. Jede Leitfähigkeit wird mit hundert multipliziert. Aus der Matrix mit 0,014 und 0,015 wird 1,4 und 1,5 Watt pro Meter-Kelvin. Quelle und DOI bleiben erhalten; die Änderung wird in der Korrekturhistorie dokumentiert.“

**Erneutes Audit und Gate (ca. 25 Sekunden)**  
„Derselbe Auditor prüft erneut. Erst jetzt sind alle deterministischen Checks wahr: `VERIFIED: parameters accepted`. Das unabhängige Gate wiederholt die Kernprüfungen und meldet `APPROVED: JAX execution allowed`.“

**JAX-Simulation (ca. 15 Sekunden)**  
„Erst nach dieser Freigabe liest die zweidimensionale anisotrope JAX-Wärmesimulation die verifizierte Matrix aus dem JSON-Artefakt. Temperaturfeld und Ergebnisdaten werden gespeichert.“

## 4:30–5:00 – Abschluss

„Die Demo behauptet keine absolute Unfehlbarkeit. Sie zeigt robustere und nachvollziehbarere Validierung durch unabhängige Rollen, deterministische Regeln und eine harte Ausführungssperre.“

„Der entscheidende Sicherheitsgewinn entsteht nicht dadurch, dass Scholar-Prime niemals Fehler macht, sondern dadurch, dass kein ungeprüftes Ergebnis die physikalische Simulation erreicht.“

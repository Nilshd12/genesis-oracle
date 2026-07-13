# Abgleich mit den fünf Präsentationsfolien

| Folie | Technischer Demoteil | Terminalmeldung | Sichtbare Antigravity-Aktivität | Kurzer Sprechsatz |
|---|---|---|---|---|
| 1 | Kontrolliert falsch beschriftete Rohmatrix aus einer realen Quelle | `Reproduzierbarer Einheitenfehler wird für die Demonstration injiziert.` | Scholar-Prime recherchiert und erzeugt `extracted_payload.json` | „Auch ein plausibles Ergebnis kann physikalisch falsch sein, wenn Zahlenwert und Einheit auseinanderlaufen.“ |
| 2 | Funktionale Trennung von Recherche, Audit, Gate und Simulation | Phasen `[01]` bis `[04]`, danach `JAX wurde nicht gestartet.` | Erst Scholar-Prime, danach separater Auditor-Agent | „Robustheit entsteht hier durch getrennte Verantwortlichkeiten, nicht durch einen größeren Prompt.“ |
| 3 | Adversarial Collaboration plus unabhängiges Gate | `Scholar-Prime` / `REJECTED – Korrektur erforderlich` / `BLOCKED: JAX execution denied` | Auditor-Agent widerspricht; Orchestrator führt das Gate aus | „Scholar-Prime produziert. Der Auditor widerspricht. Das Gate entscheidet.“ |
| 4 | Strukturierter Audit-Trail und Rückkopplung | `REJECTED`, Korrekturauftrag, `VERIFIED`, `APPROVED` | JSON-Rückgabe, Feedback zurück zu Scholar-Prime, erneute Prüfung durch denselben Auditor | „Im Trace sehen wir nicht nur das Ergebnis, sondern Quelle, Ablehnung, Korrektur und erneute Freigabe.“ |
| 5 | Elementweise Umrechnung der anisotropen 2×2-Matrix und JAX-Ausführung | `[[0.014, 0.0], [0.0, 0.015]]` wird zu `[[1.4, 0.0], [0.0, 1.5]]`; `VERIFIED – Freigabe für JAX` | Auditor akzeptiert erst die korrigierte Matrix; JAX startet nach Gate | „Der Zentimeter-Meter-Fehler verändert beide Richtungswerte um den Faktor hundert – genau das fängt der Auditor ab.“ |

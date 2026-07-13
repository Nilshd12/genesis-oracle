# Live-Demo mit nativen Antigravity-Subagenten

Im Repository existiert keine unterstützte Antigravity-Workspace-Konfigurationsdatei. Deshalb werden keine versionsabhängigen Menübezeichnungen oder proprietären Konfigurationsdateien erfunden. Verwende funktional die Ansicht, die Eltern-/Kindagenten und deren Audit-Trail visualisiert.

1. Repository `genesis-oracle` in Antigravity öffnen.
2. Eine neue Konversation im Projekt erstellen.
3. `collective_lab_validation/antigravity/orchestrator_prompt.md` über die in der installierten Version verfügbare `@`-Dateireferenz einbinden.
4. Auftrag absenden.
5. Den von Antigravity angezeigten Implementation Plan prüfen und freigeben.
6. Beobachten, wie `Scholar-Prime` als nativer Subagent startet.
7. Science-Skill-/OpenAlex-Recherche sowie `source_record.json` und `extracted_payload.json` zeigen.
8. Beobachten, wie `Auditor-Agent` als zweiter, separater Subagent startet.
9. `REJECTED: unit conversion mismatch`, erwartete/erhaltene Matrix und Audit-Feedback zeigen.
10. Im ersten Gate-Lauf `BLOCKED: JAX execution denied` und `JAX wurde nicht gestartet.` zeigen.
11. Im Trace die Rückgabe des Feedbacks an denselben Scholar-Prime zeigen.
12. `corrected_payload.json` mit `[[1.4, 0.0], [0.0, 1.5]]` öffnen.
13. Die erneute Prüfung durch denselben Auditor-Agenten zeigen.
14. `VERIFIED: parameters accepted` und `VERIFIED – Freigabe für JAX` zeigen.
15. Das finale `APPROVED: JAX execution allowed` und erst danach den JAX-Start zeigen.
16. `simulation_result.json` und `jax_heat_result.png` öffnen.

Erwartete Struktur:

```text
Orchestrator
├── Scholar-Prime
└── Auditor-Agent
```

Die Rollen sind sequenziell abhängig und müssen nicht parallel laufen. Der lokale Runner prüft Code, Artefakte und Sperrlogik, ersetzt aber nicht dieses native visuelle Tracing.

## Optionaler OpenAlex-API-Key

Der vorhandene Science Skill funktioniert im kostenlosen, stark rate-limitierten OpenAlex-Pool ohne Schlüssel. Bei 401/429 kann ein eigener Schlüssel aus den OpenAlex-Kontoeinstellungen sicher lokal in `$HOME\.env` gespeichert werden. Den Schlüssel nie in Chat oder Git einfügen. In PowerShell verbirgt `Read-Host -AsSecureString` die Eingabe; die Datei muss die Zeile `OPENALEX_API_KEY=<lokaler Wert>` enthalten.

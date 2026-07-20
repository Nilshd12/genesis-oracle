# Problem Set 12 – The Great Spreeland Logistics Sync

This directory contains the complete, locally reproducible submission:

- `submission/ps12_design_Nils_Schoen.pdf`: design document for Exercise 1 and the complete architecture.
- `agent.py`: Google ADK `root_agent` with an MCP toolset and A2A negotiation tool.
- `bridge_mcp_server.py`: local stdio MCP server simulating the City PostgreSQL bridge data.
- `a2a_negotiation.py` and `agent_cards/`: A2A 1.0 Agent Card discovery and supplier negotiation.
- `a2ui_delivery_card.json`: A2UI v1.0 delivery status card.
- `agui_stream.py`: AG-UI-compatible JSONL lifecycle, progress, state, and A2UI events.
- `../run_spreeland_dispatcher.py`: end-to-end launcher.

## Reproducible local run

From the repository root:

```powershell
uv sync
uv run python run_spreeland_dispatcher.py
uv run python run_spreeland_dispatcher.py --format ag-ui
uv run pytest tests/test_spreeland_dispatcher.py -q
```

The default path needs no cloud key. It really starts an MCP stdio subprocess,
performs `tools/list`, calls `get_bridge_status`, reads and validates A2A Agent
Cards, queries two local supplier adapters, selects the valid landed-cost
winner, checks a clearly labelled demo authorization, and emits the A2UI card.

## Interactive Google ADK mode

The ADK agent calls a language model, so this optional mode requires a Gemini
key. In PowerShell:

```powershell
$env:GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
uv run adk web .
```

Open the URL printed by ADK and select `spreeland_dispatcher`. Ask:

> Check all bridges and negotiate an order for 2,000 kg of organic gherkins.

If the configured model is unavailable, set another model before starting:

```powershell
$env:SPREELAND_MODEL = "gemini-2.5-flash"
```

## What must change for production

The local demo intentionally avoids pretending that fictional course services
or payment credentials exist.

1. Replace the records in `bridge_mcp_server.py` with parameterized, read-only
   PostgreSQL queries and inject database credentials through a secret manager.
2. Replace each `local://` URL in `agent_cards/*.json` with an authenticated
   HTTPS A2A endpoint. Replace `_local_response` with an A2A 1.0 client using
   `SendMessage` or `SendStreamingMessage`, and verify Agent Card signatures.
3. Connect the selected supplier to its UCP profile and checkout endpoints.
4. Replace `COURSE_DEMO_ONLY_NOT_AP2_CONFORMANT` with AP2 Checkout and Payment
   Mandates signed on an owner-controlled Trusted Surface. Never reuse the
   included demo HMAC key.
5. Connect the AG-UI JSONL stream to SSE/WebSocket transport and an A2UI v1.0
   renderer supporting the Basic Catalog.

Protocol versions are stated explicitly because these standards evolve. The
submission uses A2A 1.0 and A2UI v1.0 Candidate as available on 20 July 2026.

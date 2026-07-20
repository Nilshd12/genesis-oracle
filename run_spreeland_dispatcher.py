"""Run the complete Problem Set 12 dispatcher demonstration."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from spreeland_dispatcher.a2ui import render_card_payload
from spreeland_dispatcher.agui_stream import AGUIStream
from spreeland_dispatcher.workflow import run_dispatch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spreeland MCP/A2A dispatcher demonstration"
    )
    parser.add_argument(
        "--format",
        choices=("human", "ag-ui"),
        default="human",
        help="human-readable progress or AG-UI JSONL events",
    )
    return parser


async def _run(output_format: str) -> int:
    if output_format == "ag-ui":
        stream = AGUIStream(sys.stdout)
        stream.start()
        result = await run_dispatch(progress=stream.progress)
        state = result.to_dict()
        stream.finish(state, render_card_payload(state))
        return 0

    def progress(step: str, summary: str) -> None:
        print(f"[{step}] {summary}", flush=True)

    result = await run_dispatch(progress=progress)
    state = result.to_dict()
    print("\nFinal dispatch result:")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parser().parse_args().format)))

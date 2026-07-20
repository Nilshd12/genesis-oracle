"""Small AG-UI-compatible JSONL event emitter for the CLI demo."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any, TextIO


class AGUIStream:
    def __init__(self, output: TextIO):
        self.output = output
        self.thread_id = f"spreeland-{uuid.uuid4()}"
        self.run_id = f"run-{uuid.uuid4()}"
        self.message_id = f"msg-{uuid.uuid4()}"

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("timestamp", datetime.now(UTC).timestamp() * 1000)
        self.output.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.output.flush()

    def start(self) -> None:
        self._emit(
            {
                "type": "RUN_STARTED",
                "threadId": self.thread_id,
                "runId": self.run_id,
            }
        )
        self._emit(
            {
                "type": "TEXT_MESSAGE_START",
                "messageId": self.message_id,
                "role": "assistant",
            }
        )

    def progress(self, step: str, summary: str) -> None:
        step_id = f"{self.run_id}-{step}"
        self._emit({"type": "STEP_STARTED", "stepName": step, "stepId": step_id})
        self._emit(
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": self.message_id,
                "delta": f"[{step}] {summary}\n",
            }
        )
        self._emit({"type": "STEP_FINISHED", "stepName": step, "stepId": step_id})

    def finish(
        self, state: dict[str, Any], a2ui_sample: dict[str, Any]
    ) -> None:
        self._emit({"type": "STATE_SNAPSHOT", "snapshot": state})
        for message in a2ui_sample["messages"]:
            self._emit({"type": "CUSTOM", "name": "a2ui", "value": message})
        self._emit({"type": "TEXT_MESSAGE_END", "messageId": self.message_id})
        self._emit(
            {
                "type": "RUN_FINISHED",
                "threadId": self.thread_id,
                "runId": self.run_id,
            }
        )

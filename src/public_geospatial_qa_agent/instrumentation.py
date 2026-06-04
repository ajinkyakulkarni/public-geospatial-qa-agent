"""Per-call JSONL log. One JSON object per LLM call, one call per line.

Schema, every field always present:

    ts                              ISO-8601 timestamp
    session_id                      groups calls in one cycle
    archetype                       which canned question
    mode                            templated or freeform
    stage_idx                       1..6
    stage_name                      e.g. parse_datetime
    user_query                      what the user actually asked
    prompt_tokens                   total input tokens this call
    cached_tokens                   served from the cache
    completion_tokens               produced by the model
    fresh_input_tokens              prompt minus cached
    messages_count                  length of the messages list
    tool_messages_chars             this call's tool message size
    tool_messages_chars_running     cumulative across the cycle
    state_size_chars                what stayed server-side
    call_cost_usd                   billed against the rate card
    cumulative_cost_usd             cycle-so-far
    openai_response_id              cross-reference with the dashboard

cli.py:cmd_analyze reads these files. This module only writes.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CallRecord:
    """One LLM call's worth of telemetry. The runner fills this and
    appends to a JSONL log."""
    ts: str = ""
    session_id: str = ""
    archetype: str = ""
    mode: str = ""
    stage_idx: int = 0
    stage_name: str = ""
    user_query: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    fresh_input_tokens: int = 0
    messages_count: int = 0
    tool_messages_chars: int = 0
    tool_messages_chars_running: int = 0
    state_size_chars: int = 0
    call_cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    openai_response_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        # Return all fields including zero-valued ints. The analyzer
        # reads `cached_tokens` directly, and stage-1 calls legitimately
        # have cached_tokens=0 (cold cache) — dropping the field there
        # produces KeyError downstream.
        return asdict(self)


class JsonlLogger:
    """Append-only writer. One file per measurement run.

    Usage:
        with JsonlLogger("run.jsonl") as logger:
            logger.write(record)

    NOT thread- or process-safe. For the planned web UI, give each
    request its own logger instance (one file per session) or route
    writes through a single consumer thread/queue. Concurrent calls
    to `write` from multiple threads against the same instance can
    interleave half-lines and produce malformed JSONL.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "JsonlLogger":
        self._fh = self.path.open("a")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, record: CallRecord) -> None:
        if not self._fh:
            raise RuntimeError("logger used outside `with` block")
        d = record.to_dict()
        # `ts` is always present in the asdict output, but may be "" if the
        # caller left it defaulted. Stamp it here so every JSONL line has a
        # parseable timestamp regardless.
        if not d.get("ts"):
            d["ts"] = iso_now()
        self._fh.write(json.dumps(d) + "\n")
        self._fh.flush()


def iso_now() -> str:
    """UTC timestamp string suitable for the `ts` field. Avoids
    importing datetime in the hot path."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

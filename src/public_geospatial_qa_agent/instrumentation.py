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


# ---------------------------------------------------------------------
# Full-detail trace records for publishable audits
# ---------------------------------------------------------------------

import hashlib


@dataclass
class TraceRecord:
    """Publishable per-call trace.

    Carries everything an independent reader needs to verify the cost
    arithmetic and inspect what the model actually saw and produced.
    Pair with a TraceMeta sidecar for the full system prompt and tool
    schemas (referenced by sha256 here to avoid repeating ~14 KB per
    record).
    """
    ts: str = ""
    session_id: str = ""
    archetype: str = ""
    mode: str = ""             # templated | freeform
    pattern: str = ""          # single-turn | per-stage-confirm
    stage_idx: int = 0
    stage_name: str = ""
    user_query: str = ""

    # Reproducibility — the meta sidecar resolves these hashes back
    # to the full content so any reader can replay the call.
    model: str = ""
    prompt_cache_key: str = ""
    sysprompt_sha256: str = ""
    tools_sha256: str = ""

    # Input snapshot — every message that went into the call, in
    # order, excluding the system prompt (referenced above).
    messages_in: list[dict[str, Any]] = field(default_factory=list)

    # OpenAI response
    openai_response_id: str = ""
    assistant_content: str = ""              # text content of model reply (often empty for tool calls)
    assistant_tool_call: dict[str, Any] | None = None
                                              # {name, arguments_json} the model chose

    # Runner-forced behaviour. When the runner overrides the model's
    # pick (the standard single-turn / per-stage-confirm pattern does
    # this for comparability), these record what was actually run.
    # Empty for stages where the model's call was honoured verbatim.
    forced_tool_name: str = ""
    forced_tool_args_json: str = ""

    # The tool layer's output that was appended back to messages.
    tool_response_content: str = ""

    # Cost / token accounting (mirrors CallRecord so analyze --corpus
    # can consume the trace directly).
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    call_cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0

    # Wall-clock observed by the runner (request roundtrip).
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceMeta:
    """Sidecar metadata for a TraceLogger run. Resolves the sha256
    references in TraceRecord back to full content and records the
    environment so the trace is self-contained."""
    schema_version: str = "2"
    started_at: str = ""
    finished_at: str = ""
    model: str = ""
    rate_card: dict[str, Any] = field(default_factory=dict)
    sysprompt_sha256: str = ""
    sysprompt: str = ""
    tools_sha256: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    corpus_file: str | None = None
    corpus_sha256: str | None = None
    notes: str = ""
    # Package + library versions, populated at open() time so a
    # reviewer can pin the runner code and the OpenAI SDK version.
    package_versions: dict[str, str] = field(default_factory=dict)
    # Git commit hash + dirty flag, when the run is inside a git
    # working tree.
    git_commit: dict[str, str] = field(default_factory=dict)


def _collect_versions() -> dict[str, str]:
    """Return a {package: version} dict for everything that affects
    the measurement: the agent itself, the OpenAI SDK, tiktoken,
    pystac-client (live backend), python."""
    import importlib
    import importlib.metadata as _md
    import platform as _pf
    out: dict[str, str] = {"python": _pf.python_version()}
    for name in ("public_geospatial_qa_agent", "openai", "tiktoken",
                  "pystac-client", "fastapi", "uvicorn"):
        try:
            out[name] = _md.version(name)
        except Exception:
            out[name] = "unknown"
    return out


def _resolve_git_commit() -> dict[str, str]:
    """Return {sha, dirty}. Empty dict when not in a git work tree."""
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        diff = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return {"sha": sha, "dirty": "yes" if diff.returncode != 0 else "no"}
    except Exception:
        return {}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


class TraceLogger:
    """Append-only logger for TraceRecord. Writes JSONL to a `.trace.jsonl`
    file and a `.trace.meta.json` sidecar with the resolved sysprompt /
    tools / rate card so traces can be republished and validated
    independently of this repo.

        with TraceLogger("runs/x.trace.jsonl", sysprompt=..., tools=...) as t:
            t.write(record)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sysprompt: str,
        tools: list[dict[str, Any]],
        model: str,
        rate_card: dict[str, Any],
        corpus_file: str | None = None,
        notes: str = "",
    ):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.sysprompt = sysprompt
        self.sysprompt_sha256 = sha256_text(sysprompt)
        self.tools = tools
        self.tools_sha256 = sha256_json(tools)
        self.model = model
        self.rate_card = rate_card
        self.corpus_file = corpus_file
        # Hash the corpus content if a file path was provided so the
        # meta sidecar can prove which queries were run.
        if corpus_file:
            try:
                self.corpus_sha256 = sha256_text(
                    Path(corpus_file).read_text()
                )
            except Exception:
                self.corpus_sha256 = None
        else:
            self.corpus_sha256 = None
        self.notes = notes
        self._fh = None

    def __enter__(self) -> "TraceLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")
        # Resolve package + library versions and the git commit. All
        # written to TraceMeta so a reviewer can exactly tie the
        # numbers to a code revision.
        meta = TraceMeta(
            started_at=iso_now(),
            model=self.model,
            rate_card=self.rate_card,
            sysprompt_sha256=self.sysprompt_sha256,
            sysprompt=self.sysprompt,
            tools_sha256=self.tools_sha256,
            tools=self.tools,
            corpus_file=self.corpus_file,
            corpus_sha256=self.corpus_sha256,
            notes=self.notes,
            package_versions=_collect_versions(),
            git_commit=_resolve_git_commit(),
        )
        self.meta_path.write_text(json.dumps(asdict(meta), indent=2))
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
        # Stamp finished_at by reloading and overwriting the meta file.
        try:
            d = json.loads(self.meta_path.read_text())
            d["finished_at"] = iso_now()
            self.meta_path.write_text(json.dumps(d, indent=2))
        except Exception:
            pass

    def hashes(self) -> tuple[str, str]:
        """Return (sysprompt_sha256, tools_sha256) for runner use."""
        return (self.sysprompt_sha256, self.tools_sha256)

    def write(self, record: TraceRecord) -> None:
        if not self._fh:
            raise RuntimeError("trace logger used outside `with` block")
        d = record.to_dict()
        if not d.get("ts"):
            d["ts"] = iso_now()
        if not d.get("model"):
            d["model"] = self.model
        if not d.get("sysprompt_sha256"):
            d["sysprompt_sha256"] = self.sysprompt_sha256
        if not d.get("tools_sha256"):
            d["tools_sha256"] = self.tools_sha256
        self._fh.write(json.dumps(d) + "\n")
        self._fh.flush()

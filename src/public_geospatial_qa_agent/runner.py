"""The six-stage cycle.

run_cycle() owns one cycle. Per stage: assemble messages, call
chat.completions, invoke the matching tool wrapper, append the
assistant tool_call and the tool response to messages, record a Stage,
hand the Stage to the optional on_stage callback, loop. Return a
CycleTrace at the end (or sooner, if the OpenAI call raises).

Plain function, no globals, no prints. The CLI calls it once per
archetype. The web app calls it from a worker thread and uses
on_stage to push SSE events to the browser.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .archetypes import Archetype
from .backends import Backend, CannedBackend
from .cost import GPT_5_2_STANDARD, RateCard, cost_for_call
from .instrumentation import (
    CallRecord,
    JsonlLogger,
    TraceLogger,
    TraceRecord,
    iso_now,
)
from .state import AgentState
from .tools import make_tools


@dataclass
class Stage:
    """One stage of a cycle — what the LLM saw, what the tool did,
    what got billed."""
    idx: int
    name: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    call_cost_usd: float = 0.0
    tool_message_chars: int = 0   # what the LLM saw from the tool
    state_size_chars: int = 0     # what stayed server-side
    assistant_tool_call_args: str = ""  # what the LLM decided
    tool_response_preview: str = ""     # first 200 chars of tool msg
    openai_response_id: str = ""


@dataclass
class CycleTrace:
    """One complete cycle — six stages plus aggregates.

    The `error` field is populated when an OpenAI call raises mid-cycle
    (rate limit, auth failure, transient network). Stages completed
    before the error are preserved in `stages` so the caller (CLI,
    future web UI) can render a partial result instead of getting only
    an exception.
    """
    session_id: str
    archetype_id: str
    user_query: str
    mode: str
    stages: list[Stage] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    final_state_size_chars: int = 0
    error: str | None = None  # set if the cycle was cut short mid-flight


def load_sysprompt() -> str:
    """Read the system prompt from data/sysprompt.txt."""
    p = Path(__file__).resolve().parent.parent.parent / "data" / "sysprompt.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return p.read_text()


def load_tool_schemas() -> list[dict[str, Any]]:
    """Read tool schemas from data/tool_schemas.json."""
    import json
    p = Path(__file__).resolve().parent.parent.parent / "data" / "tool_schemas.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return json.loads(p.read_text())


def run_cycle(
    client: OpenAI,
    archetype: Archetype,
    mode: str,
    *,
    model: str = "gpt-5.2",
    rate: RateCard = GPT_5_2_STANDARD,
    prompt_cache_key: str = "public-geospatial-qa-agent",
    sysprompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    logger: JsonlLogger | None = None,
    trace_logger: TraceLogger | None = None,
    session_id: str | None = None,
    user_query: str | None = None,
    backend: Backend | None = None,
    on_stage: Callable[["Stage", "AgentState"], None] | None = None,
) -> CycleTrace:
    """Run one full cycle (6 stages) for one archetype in one mode.

    Args:
        client: an instantiated openai.OpenAI client. The caller owns
                the API key and any rate-limit handling.
        archetype: the user query + pipeline sequence.
        mode: 'templated' or 'freeform'.
        model: OpenAI model id to use.
        rate: rate card to bill against (defaults to gpt-5.2 standard).
        prompt_cache_key: passed to OpenAI as `extra_body.prompt_cache_key`;
                          all calls in this run that share this key share
                          a cache. Kept stable across the whole session
                          so the cache warms in the expected way.
        sysprompt: optional override (defaults to load_sysprompt()).
        tools: optional override (defaults to load_tool_schemas()).
        logger: optional JsonlLogger to append per-call records to.
        session_id: optional override for grouping records.
        user_query: optional override for the user's question text.
                    Defaults to archetype.query (the canned examples).
                    The web UI passes the user's typed-in query here.
        on_stage: optional callback invoked after each stage completes,
                  with the newly-completed Stage and the current
                  AgentState. The web UI uses this to push SSE events
                  to the browser; the CLI doesn't pass anything.

    Returns:
        CycleTrace describing the whole cycle.
    """
    if sysprompt is None:
        sysprompt = load_sysprompt()
    if tools is None:
        tools = load_tool_schemas()
    if session_id is None:
        session_id = str(uuid.uuid4())
    if backend is None:
        backend = CannedBackend()

    state = AgentState()
    tool_wrappers = make_tools(mode, state, backend)
    effective_query = user_query if user_query is not None else archetype.query

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": effective_query},
    ]

    trace = CycleTrace(
        session_id=session_id,
        archetype_id=archetype.id,
        user_query=effective_query,
        mode=mode,
    )
    tool_chars_running = 0

    derived = _derive_args_from_query(effective_query)

    for stage_idx, stage_name in enumerate(archetype.pipeline, start=1):
        # ── Call OpenAI ────────────────────────────────────────────
        # Snapshot the messages BEFORE the call so the trace records
        # exactly what the model saw. messages excludes the system
        # prompt here because the trace logger stores that once in
        # its meta sidecar and refers to it by sha256.
        trace_messages_in = (
            [dict(m) for m in messages[1:]] if trace_logger else []
        )
        t_call_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_body={"prompt_cache_key": prompt_cache_key},
            )
        except Exception as e:
            trace.error = f"{type(e).__name__} at stage {stage_idx}/{stage_name}: {e}"
            trace.final_state_size_chars = sum(state.snapshot_sizes().values())
            return trace
        latency_ms = int((time.perf_counter() - t_call_start) * 1000)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        cached_tokens = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details else 0
        )
        completion_tokens = usage.completion_tokens

        call_cost = cost_for_call(
            rate, prompt_tokens, cached_tokens, completion_tokens
        )

        # ── Invoke the tool wrapper ────────────────────────────────
        # Force the pre-decided stage_name rather than honour the
        # LLM's tool_calls. Per-stage token counts only line up across
        # samples if the sequence is identical; the cached_tokens
        # numbers are what we're measuring, not the model's routing.
        tool_method = getattr(tool_wrappers, stage_name)
        tool_args = _default_args_for_stage(stage_name, archetype, state, derived)
        tool_message_content = tool_method(**tool_args)

        # Append the assistant's tool_call + the tool response so the
        # next stage sees the accumulated context.
        tool_call_id = f"call_{stage_idx}_{uuid.uuid4().hex[:8]}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": stage_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_message_content,
        })

        tool_chars_running += len(tool_message_content)

        # ── Record the stage ───────────────────────────────────────
        st = Stage(
            idx=stage_idx,
            name=stage_name,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            call_cost_usd=call_cost,
            tool_message_chars=len(tool_message_content),
            state_size_chars=sum(state.snapshot_sizes().values()),
            assistant_tool_call_args=json.dumps(tool_args),
            tool_response_preview=tool_message_content[:200],
            openai_response_id=getattr(response, "id", "") or "",
        )
        trace.stages.append(st)
        trace.total_prompt_tokens += prompt_tokens
        trace.total_cached_tokens += cached_tokens
        trace.total_completion_tokens += completion_tokens
        trace.total_cost_usd += call_cost

        # Push the per-stage event to the caller (web UI uses this
        # for SSE streaming; CLI doesn't pass on_stage).
        if on_stage is not None:
            try:
                on_stage(st, state)
            except Exception as e:
                # A misbehaving callback shouldn't tank the cycle.
                # Record the error and continue; the trace still
                # captures every stage we successfully measured.
                # If trace.error is already set (e.g., an OpenAI
                # error this same iteration), preserve the original
                # error — the callback's secondary failure is less
                # actionable than the root cause.
                if trace.error is None:
                    trace.error = f"on_stage callback raised: {type(e).__name__}: {e}"

        if logger:
            logger.write(CallRecord(
                ts=iso_now(),
                session_id=session_id,
                archetype=archetype.id,
                mode=mode,
                stage_idx=stage_idx,
                stage_name=stage_name,
                user_query=archetype.query,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                fresh_input_tokens=max(0, prompt_tokens - cached_tokens),
                messages_count=len(messages),
                tool_messages_chars=len(tool_message_content),
                tool_messages_chars_running=tool_chars_running,
                state_size_chars=st.state_size_chars,
                call_cost_usd=round(call_cost, 6),
                cumulative_cost_usd=round(trace.total_cost_usd, 6),
                openai_response_id=st.openai_response_id,
            ))

        if trace_logger:
            model_msg = response.choices[0].message
            model_tool_call = None
            if getattr(model_msg, "tool_calls", None):
                tc = model_msg.tool_calls[0]
                model_tool_call = {
                    "name": tc.function.name,
                    "arguments_json": tc.function.arguments,
                }
            trace_logger.write(TraceRecord(
                ts=iso_now(),
                session_id=session_id,
                archetype=archetype.id,
                mode=mode,
                pattern="single-turn",
                stage_idx=stage_idx,
                stage_name=stage_name,
                user_query=effective_query,
                model=model,
                prompt_cache_key=prompt_cache_key,
                messages_in=trace_messages_in,
                openai_response_id=st.openai_response_id,
                assistant_content=(model_msg.content or "") if model_msg else "",
                assistant_tool_call=model_tool_call,
                forced_tool_name=stage_name,
                forced_tool_args_json=json.dumps(tool_args),
                tool_response_content=tool_message_content,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                call_cost_usd=round(call_cost, 6),
                cumulative_cost_usd=round(trace.total_cost_usd, 6),
                latency_ms=latency_ms,
            ))

    trace.final_state_size_chars = sum(state.snapshot_sizes().values())
    return trace


# Stages whose tool wrapper returns status="pending_confirmation".
# Used by simulate_per_stage_confirm_cycle to decide which stage
# transitions get an interstitial confirmation round.
PENDING_CONFIRMATION_STAGES = frozenset({
    "parse_datetime",
    "geocode",
    "select_collection",
})


CONFIRM_USER_TEXT = "Confirm."


def simulate_per_stage_confirm_cycle(
    client: OpenAI,
    archetype: Archetype,
    mode: str,
    *,
    model: str = "gpt-5.2",
    rate: RateCard = GPT_5_2_STANDARD,
    prompt_cache_key: str = "public-geospatial-qa-agent",
    sysprompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    logger: JsonlLogger | None = None,
    trace_logger: TraceLogger | None = None,
    session_id: str | None = None,
    user_query: str | None = None,
    backend: Backend | None = None,
    on_stage: Callable[["Stage", "AgentState"], None] | None = None,
) -> CycleTrace:
    """Multi-turn confirmation pattern: the cycle pauses at each input-
    resolution stage so the user can confirm or correct the parsed
    value before the pipeline continues.

    Two cost-relevant differences from run_cycle:

    1. Each `parse_datetime`, `geocode`, and `select_collection` stage
       is followed by an additional OpenAI call that produces the
       confirmation prose the model would write at a real turn boundary
       (e.g. "Time set to 2020-01-01/2020-12-31. Confirm to continue.").
       That call is billed and recorded as its own Stage with idx=N.5
       so analyze can split the prose cost from the tool-call cost.

    2. After the confirmation prose, a synthetic user "Confirm." message
       is appended to the history. Subsequent stages see the extended
       conversation, so cumulative input tokens grow faster than under
       run_cycle.

    Returns a CycleTrace with the same shape run_cycle returns. The
    tool-message and state recording semantics are identical, so the
    per-cycle aggregates from analyze --corpus stack against the
    single-turn baseline directly.
    """
    if sysprompt is None:
        sysprompt = load_sysprompt()
    if tools is None:
        tools = load_tool_schemas()
    if session_id is None:
        session_id = str(uuid.uuid4())
    if backend is None:
        backend = CannedBackend()

    state = AgentState()
    tool_wrappers = make_tools(mode, state, backend)
    effective_query = user_query if user_query is not None else archetype.query

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": effective_query},
    ]

    trace = CycleTrace(
        session_id=session_id,
        archetype_id=archetype.id,
        user_query=effective_query,
        mode=mode,
    )
    tool_chars_running = 0
    derived = _derive_args_from_query(effective_query)

    for stage_idx, stage_name in enumerate(archetype.pipeline, start=1):
        # ── Stage's tool call (same as run_cycle) ──────────────────
        trace_messages_in = (
            [dict(m) for m in messages[1:]] if trace_logger else []
        )
        t_call_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_body={"prompt_cache_key": prompt_cache_key},
            )
        except Exception as e:
            trace.error = f"{type(e).__name__} at stage {stage_idx}/{stage_name}: {e}"
            trace.final_state_size_chars = sum(state.snapshot_sizes().values())
            return trace
        latency_ms = int((time.perf_counter() - t_call_start) * 1000)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        cached_tokens = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details else 0
        )
        completion_tokens = usage.completion_tokens
        call_cost = cost_for_call(
            rate, prompt_tokens, cached_tokens, completion_tokens
        )

        tool_method = getattr(tool_wrappers, stage_name)
        tool_args = _default_args_for_stage(stage_name, archetype, state, derived)
        tool_message_content = tool_method(**tool_args)

        tool_call_id = f"call_{stage_idx}_{uuid.uuid4().hex[:8]}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": stage_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_message_content,
        })

        tool_chars_running += len(tool_message_content)
        st = Stage(
            idx=stage_idx,
            name=stage_name,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            call_cost_usd=call_cost,
            tool_message_chars=len(tool_message_content),
            state_size_chars=sum(state.snapshot_sizes().values()),
            assistant_tool_call_args=json.dumps(tool_args),
            tool_response_preview=tool_message_content[:200],
            openai_response_id=getattr(response, "id", "") or "",
        )
        trace.stages.append(st)
        trace.total_prompt_tokens += prompt_tokens
        trace.total_cached_tokens += cached_tokens
        trace.total_completion_tokens += completion_tokens
        trace.total_cost_usd += call_cost

        if on_stage is not None:
            try:
                on_stage(st, state)
            except Exception as e:
                if trace.error is None:
                    trace.error = f"on_stage callback raised: {type(e).__name__}: {e}"

        if logger:
            logger.write(CallRecord(
                ts=iso_now(),
                session_id=session_id,
                archetype=archetype.id,
                mode=mode,
                stage_idx=stage_idx,
                stage_name=stage_name,
                user_query=archetype.query,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                fresh_input_tokens=max(0, prompt_tokens - cached_tokens),
                messages_count=len(messages),
                tool_messages_chars=len(tool_message_content),
                tool_messages_chars_running=tool_chars_running,
                state_size_chars=st.state_size_chars,
                call_cost_usd=round(call_cost, 6),
                cumulative_cost_usd=round(trace.total_cost_usd, 6),
                openai_response_id=st.openai_response_id,
            ))

        if trace_logger:
            model_msg = response.choices[0].message
            model_tool_call = None
            if getattr(model_msg, "tool_calls", None):
                tc = model_msg.tool_calls[0]
                model_tool_call = {
                    "name": tc.function.name,
                    "arguments_json": tc.function.arguments,
                }
            trace_logger.write(TraceRecord(
                ts=iso_now(),
                session_id=session_id,
                archetype=archetype.id,
                mode=mode,
                pattern="per-stage-confirm",
                stage_idx=stage_idx,
                stage_name=stage_name,
                user_query=effective_query,
                model=model,
                prompt_cache_key=prompt_cache_key,
                messages_in=trace_messages_in,
                openai_response_id=st.openai_response_id,
                assistant_content=(model_msg.content or "") if model_msg else "",
                assistant_tool_call=model_tool_call,
                forced_tool_name=stage_name,
                forced_tool_args_json=json.dumps(tool_args),
                tool_response_content=tool_message_content,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                call_cost_usd=round(call_cost, 6),
                cumulative_cost_usd=round(trace.total_cost_usd, 6),
                latency_ms=latency_ms,
            ))

        # ── Confirmation round, if this is a pending stage ────────
        if stage_name not in PENDING_CONFIRMATION_STAGES:
            continue

        # Ask the model to write the confirmation prose. No tools
        # available on this call — the model writes plain text, which
        # is what would actually appear to the user mid-conversation.
        confirm_messages_in = (
            [dict(m) for m in messages[1:]] if trace_logger else []
        )
        t_confirm_start = time.perf_counter()
        try:
            confirm_response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"prompt_cache_key": prompt_cache_key},
            )
        except Exception as e:
            trace.error = (
                f"{type(e).__name__} at confirmation for stage "
                f"{stage_idx}/{stage_name}: {e}"
            )
            trace.final_state_size_chars = sum(state.snapshot_sizes().values())
            return trace
        confirm_latency_ms = int((time.perf_counter() - t_confirm_start) * 1000)

        confirm_usage = confirm_response.usage
        c_pt = confirm_usage.prompt_tokens
        c_ct = (
            confirm_usage.prompt_tokens_details.cached_tokens
            if confirm_usage.prompt_tokens_details else 0
        )
        c_ot = confirm_usage.completion_tokens
        c_cost = cost_for_call(rate, c_pt, c_ct, c_ot)
        confirm_prose = confirm_response.choices[0].message.content or ""

        messages.append({"role": "assistant", "content": confirm_prose})
        messages.append({"role": "user", "content": CONFIRM_USER_TEXT})

        # Record the confirmation prose as its own Stage with a
        # half-step idx so it stays distinguishable from the tool
        # stages in the JSONL log.
        confirm_st = Stage(
            idx=stage_idx,
            name=f"{stage_name}__confirm",
            prompt_tokens=c_pt,
            cached_tokens=c_ct,
            completion_tokens=c_ot,
            call_cost_usd=c_cost,
            tool_message_chars=0,
            state_size_chars=sum(state.snapshot_sizes().values()),
            assistant_tool_call_args="",
            tool_response_preview=confirm_prose[:200],
            openai_response_id=getattr(confirm_response, "id", "") or "",
        )
        trace.stages.append(confirm_st)
        trace.total_prompt_tokens += c_pt
        trace.total_cached_tokens += c_ct
        trace.total_completion_tokens += c_ot
        trace.total_cost_usd += c_cost

        if on_stage is not None:
            try:
                on_stage(confirm_st, state)
            except Exception:
                pass

        if logger:
            logger.write(CallRecord(
                ts=iso_now(),
                session_id=session_id,
                archetype=archetype.id,
                mode=mode,
                stage_idx=stage_idx,
                stage_name=f"{stage_name}__confirm",
                user_query=archetype.query,
                prompt_tokens=c_pt,
                cached_tokens=c_ct,
                completion_tokens=c_ot,
                fresh_input_tokens=max(0, c_pt - c_ct),
                messages_count=len(messages),
                tool_messages_chars=0,
                tool_messages_chars_running=tool_chars_running,
                state_size_chars=confirm_st.state_size_chars,
                call_cost_usd=round(c_cost, 6),
                cumulative_cost_usd=round(trace.total_cost_usd, 6),
                openai_response_id=confirm_st.openai_response_id,
            ))

        if trace_logger:
            trace_logger.write(TraceRecord(
                ts=iso_now(),
                session_id=session_id,
                archetype=archetype.id,
                mode=mode,
                pattern="per-stage-confirm",
                stage_idx=stage_idx,
                stage_name=f"{stage_name}__confirm",
                user_query=effective_query,
                model=model,
                prompt_cache_key=prompt_cache_key,
                messages_in=confirm_messages_in,
                openai_response_id=confirm_st.openai_response_id,
                assistant_content=confirm_prose,
                assistant_tool_call=None,
                forced_tool_name="",
                forced_tool_args_json="",
                tool_response_content="",
                prompt_tokens=c_pt,
                cached_tokens=c_ct,
                completion_tokens=c_ot,
                call_cost_usd=round(c_cost, 6),
                cumulative_cost_usd=round(trace.total_cost_usd, 6),
                latency_ms=confirm_latency_ms,
            ))

    trace.final_state_size_chars = sum(state.snapshot_sizes().values())
    return trace


def _default_args_for_stage(
    stage_name: str,
    archetype: Archetype,
    state: "AgentState",
    derived: dict[str, str],
) -> dict[str, Any]:
    """Choose tool arguments for a stage.

    For canned-backend measurement runs the args don't have to be
    accurate — the wrappers don't parse them — but with the live
    backend they have to be plausible. select_collection has to pick
    something the rag stage actually returned, and stac_search has to
    use that same collection_id so the catalog query lines up.
    """
    if stage_name == "select_collection" and state.collections_result:
        matches = state.collections_result.get("matches", [])
        if matches:
            cid = matches[0]["id"]
            return {"collection_id": cid}
    if stage_name == "stac_search" and state.selected_collection_id:
        return {"collection_id": state.selected_collection_id, "limit": 15}
    return {
        "parse_datetime":   {"value": derived["datetime"]},
        "geocode":          {"query": derived["place"]},
        "collections_rag":  {"query": archetype.query, "top_k": 5},
        "select_collection": {"collection_id": "no2-monthly"},
        "stac_search":      {"collection_id": "no2-monthly", "limit": 15},
        "compute_stats":    {},
        "build_viz_tiles":  {},
    }[stage_name]


# Very small heuristic extractor. Picks a date range and a place name
# out of the user's query string. Good enough to make the live backend
# resolve a plausible bbox; not a substitute for an actual extractor
# stage.
_MONTH_RX = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
_YEAR_RX = r"(?:19|20)\d{2}"


def _derive_args_from_query(query: str) -> dict[str, str]:
    import re
    years = re.findall(_YEAR_RX, query)
    if len(years) >= 2:
        dt = f"{years[0]}-01-01/{years[1]}-12-31"
    elif years:
        dt = f"{years[0]}-01-01/{years[0]}-12-31"
    else:
        dt = "2020-01-01/2020-12-31"

    place_match = re.search(
        r"(?:over|in|for|near|across)\s+(?:the\s+)?"
        r"([A-Z][\w]+(?:\s+[A-Z][\w]+){0,3})",
        query,
    )
    place = place_match.group(1) if place_match else "Los Angeles County, California"
    return {"datetime": dt, "place": place}


CLARIFY_SYS_PROMPT = (
    "You are a query intake assistant for an Earth-observation Q&A agent. "
    "The downstream agent answers questions about which datasets cover a "
    "place over a time window, then summarises catalog items in that area. "
    "Decide whether the user query has all three of:\n"
    "(a) AN EXPLICIT TIME REFERENCE — a year, month, season, or date range. "
    "Words like 'recent', 'latest', or 'now' do NOT count.\n"
    "(b) A PLACE — a city, county, state, country, region, or bbox.\n"
    "(c) A DATASET HINT — either a variable name (NO2, NDVI, SST, methane, "
    "precipitation, sea ice, land cover, etc.) or a collection name.\n"
    "\n"
    "If ALL THREE are present, respond with the literal token OK and nothing "
    "else.\n"
    "\n"
    "If any are missing, do NOT demand them blankly. Instead, PROPOSE a "
    "reasonable default for what's missing and ask the user to confirm or "
    "correct it. The proposed default for time is 'the most recent full "
    "year' (currently 2023). The proposed default for dataset is the most "
    "obvious variable given the place or context (NO2 for air quality, NDVI "
    "for vegetation health, SST for ocean temperature, etc.). The proposed "
    "default for place is whatever the user mentioned most recently or, if "
    "nothing, a global / country-scale view. Pose your reply as: "
    "'Going to <default interpretation>. Want a different <missing field>?'\n"
    "\n"
    "Be terse — one sentence, no preamble, no pleasantries.\n"
    "\n"
    "Examples:\n"
    "  User: 'Show NO2 over Houston'\n"
    "       → 'Going to use 2023 (most recent full year). Want a different year?'\n"
    "  User: 'Show NO2 over Houston for 2021'\n"
    "       → 'OK'\n"
    "  User: 'Show air quality data for LA County'\n"
    "       → 'Going to use NO2 for 2023. Want a different variable or year?'\n"
    "  User: 'What datasets do you have for NO2'\n"
    "       → 'Going to search globally for the most recent year. Narrow the region or year?'\n"
    "  User: 'Show me data for Los Angeles in 2020'\n"
    "       → 'Going to show NO2 air quality. Want vegetation, temperature, or another variable instead?'\n"
    "  User: 'What can you tell me about India in 2021'\n"
    "       → 'Going to show NO2 for India in 2021. Want a different variable?'\n"
    "  User: 'Show me a map'\n"
    "       → 'Need a place and a variable. Try \"NO2 over Houston for 2021\" or tell me the topic and area.'\n"
    "\n"
    "Do not call any tools. Do not add explanations."
)


def needs_clarification(query: str) -> str | None:
    """Cheap rule-based fallback used when no OpenAI client is available
    (offline tests, dry runs). Returns a question or None. The web UI
    and run-corpus prefer llm_clarification_check, which is more
    forgiving and measurable.
    """
    import re
    has_year = bool(re.search(_YEAR_RX, query))
    has_month = bool(re.search(_MONTH_RX, query))
    has_date = has_year or has_month
    has_place = bool(re.search(
        r"(?:over|in|for|near|across)\s+(?:the\s+)?"
        r"[A-Z][\w]+(?:\s+[A-Z][\w]+){0,3}",
        query,
    ))
    if not has_date and not has_place:
        return ("What place and date range should I look at? "
                "For example: \"NO2 over Houston for 2021\".")
    if not has_date:
        return "What date range or year do you want?"
    if not has_place:
        return "What place should I search? A city, county, or region works."
    return None


@dataclass
class ClarifyResult:
    """One LLM clarification call's worth of state. Surfaces the
    question (or None if the query was OK), plus the cost and token
    telemetry so corpus runs can report the clarify-cost separately
    from the cycle cost."""
    question: str | None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    openai_response_id: str = ""


def llm_clarification_check(
    client: OpenAI,
    query: str,
    *,
    model: str = "gpt-5.2",
    rate: RateCard = GPT_5_2_STANDARD,
    prompt_cache_key: str = "public-geospatial-qa-agent-clarify",
) -> ClarifyResult:
    """One small LLM call that decides whether to ask back.

    Returns ClarifyResult.question = None when the query has enough
    context to proceed; otherwise a single short follow-up question.
    Either way, the call's token counts and cost are recorded so the
    corpus aggregator can report clarification overhead.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLARIFY_SYS_PROMPT},
                {"role": "user", "content": query},
            ],
            extra_body={"prompt_cache_key": prompt_cache_key},
            max_completion_tokens=80,
        )
    except Exception as e:
        # If the gate call fails (rate-limit, auth, network), let the
        # downstream cycle run rather than blocking the user. Record
        # zero cost so corpus runs don't penalise a transient failure.
        return ClarifyResult(question=None, openai_response_id=f"error:{type(e).__name__}")

    msg = (response.choices[0].message.content or "").strip()
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    cached_tokens = (
        usage.prompt_tokens_details.cached_tokens
        if usage.prompt_tokens_details else 0
    )
    completion_tokens = usage.completion_tokens
    cost = cost_for_call(rate, prompt_tokens, cached_tokens, completion_tokens)

    # The system prompt asks for exactly "OK" when no follow-up is
    # needed. Tolerate trailing punctuation / quoting / model gloss
    # ("OK.", "OK\n", "Sure, OK").
    is_ok = (
        msg.upper().startswith("OK")
        and len(msg) <= 4
    ) or msg.upper().strip(" .\"'") == "OK"
    question = None if is_ok else msg

    return ClarifyResult(
        question=question,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        openai_response_id=getattr(response, "id", "") or "",
    )

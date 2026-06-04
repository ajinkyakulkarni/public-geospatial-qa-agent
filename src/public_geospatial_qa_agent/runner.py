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
from .instrumentation import CallRecord, JsonlLogger, iso_now
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
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_body={"prompt_cache_key": prompt_cache_key},
            )
        except Exception as e:
            # Preserve stages completed so far; the caller decides
            # whether to retry, render the partial, or escalate.
            trace.error = f"{type(e).__name__} at stage {stage_idx}/{stage_name}: {e}"
            trace.final_state_size_chars = sum(state.snapshot_sizes().values())
            return trace

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
        "stats":            {},
        "viz":              {},
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


def needs_clarification(query: str) -> str | None:
    """Return a follow-up question if the query is missing a date range
    or a place; otherwise None.

    The check is rule-based on purpose — it should be cheap, predictable,
    and have an off switch. An LLM-based intent extractor would be more
    forgiving but adds an extra billable call per turn before the cycle
    even starts.
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

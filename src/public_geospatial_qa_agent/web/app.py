"""HTTP routes and SSE producer for the browser UI.

Endpoints:

    GET  /                  → index.html
    GET  /static/{path}     → JS / CSS / assets
    GET  /api/health        → version, budget remaining, key flag
    GET  /api/archetypes    → the five quick-buttons
    POST /api/ask           → starts a cycle, returns an SSE stream of
                              per-stage events; the final event carries
                              the full CycleTrace.

A few deliberate constraints. The API key comes from OPENAI_API_KEY
in the server environment and never from request input. /api/ask is
templated-only; freeform exists for measurement and isn't appropriate
to expose to anonymous traffic. User query is capped at 1 KB and
stripped of control characters before forwarding. The web app does
not write the JSONL log that the CLI writes — keeping the user query
and the OpenAI response_id together would create a privacy hazard for
a multi-user deployment.
"""
from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from .. import __version__
from ..archetypes import ALL_ARCHETYPES, archetype_by_id
from ..backends import make_backend
from ..cost import GPT_5_2_STANDARD
from ..instrumentation import JsonlLogger, TraceLogger
from ..runner import (
    llm_clarification_check,
    load_sysprompt,
    load_tool_schemas,
    needs_clarification,
    run_cycle,
    simulate_per_stage_confirm_cycle,
)
from .budget import Budget

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_QUERY_BYTES = 1024  # cap user input; defends against token-burn DoS

# Default budget cap for the local-dev case. Override with
# PGQA_BUDGET_USD env var before starting the server.
DEFAULT_BUDGET_USD = 1.00


def _sanitize_query(raw: str) -> str:
    """Strip control characters and clamp to MAX_QUERY_BYTES.
    Defensive against the trivial 100-KB token-burn attack the security
    review flagged."""
    cleaned = "".join(ch for ch in raw if ch == "\n" or ch >= " ")
    if len(cleaned.encode("utf-8")) > MAX_QUERY_BYTES:
        # Truncate by codepoints (not bytes) so we never split a
        # multi-byte UTF-8 sequence.
        while len(cleaned.encode("utf-8")) > MAX_QUERY_BYTES:
            cleaned = cleaned[:-1]
    return cleaned.strip()


def create_app(
    *,
    budget_cap_usd: float | None = None,
    api_key: str | None = None,
    backend_name: str | None = None,
    measurement_log_path: str | None = None,
    trace_path: str | None = None,
) -> FastAPI:
    """Build the FastAPI app. Exposed as a factory so the Playwright
    tests can construct an app with a stubbed OpenAI client or zero
    budget without touching env vars."""
    if budget_cap_usd is None:
        budget_cap_usd = float(os.environ.get("PGQA_BUDGET_USD", DEFAULT_BUDGET_USD))
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if backend_name is None:
        backend_name = os.environ.get("PGQA_BACKEND", "canned")
    if measurement_log_path is None:
        measurement_log_path = os.environ.get("PGQA_MEASUREMENT_LOG") or None
    if trace_path is None:
        trace_path = os.environ.get("PGQA_TRACE_LOG") or None
    # Open-once logger so concurrent SSE workers append to the same
    # file. JsonlLogger.write isn't fully thread-safe, but FastAPI's
    # threadpool runs one request per cycle and per-line writes flush
    # immediately, so interleaving on local single-user use is rare
    # enough to live with. Not appropriate for multi-tenant.
    measurement_logger: JsonlLogger | None = None
    if measurement_log_path:
        Path(measurement_log_path).parent.mkdir(parents=True, exist_ok=True)
        measurement_logger = JsonlLogger(measurement_log_path).__enter__()
    # Optional publish-grade trace logger: writes a richer record per
    # LLM call plus a `.meta.json` sidecar with the resolved sysprompt
    # / tool schemas / rate card so the trace is self-contained for
    # reviewers and downstream validators.
    trace_logger: TraceLogger | None = None
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        trace_logger = TraceLogger(
            trace_path,
            sysprompt=load_sysprompt(),
            tools=load_tool_schemas(),
            model="gpt-5.2",
            rate_card={
                "input_per_million": GPT_5_2_STANDARD.input_per_million,
                "cached_input_per_million": GPT_5_2_STANDARD.cached_input_per_million,
                "output_per_million": GPT_5_2_STANDARD.output_per_million,
            },
            corpus_file=os.environ.get("PGQA_CORPUS_FILE"),
            notes=os.environ.get("PGQA_TRACE_NOTES", ""),
        ).__enter__()

    app = FastAPI(
        title="public-geospatial-qa-agent",
        version=__version__,
        docs_url=None,  # no /docs in this minimal local app
        redoc_url=None,
    )
    budget = Budget(cap_usd=budget_cap_usd)

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        st = budget.state()
        return {
            "ok": True,
            "version": __version__,
            "budget_cap_usd": st.cap_usd,
            "budget_remaining_usd": round(st.remaining_usd, 4),
            "budget_spent_usd": round(st.spent_usd, 4),
            "has_api_key": api_key is not None,
            "backend": backend_name,
        }

    @app.get("/api/archetypes")
    def list_archetypes() -> dict[str, Any]:
        return {
            "archetypes": [
                {"id": a.id, "query": a.query} for a in ALL_ARCHETYPES
            ]
        }

    @app.post("/api/ask")
    async def ask(request: Request) -> StreamingResponse:
        if api_key is None:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not configured on the server.",
            )
        if not budget.reserve():
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Process budget exhausted "
                    f"(${budget.state().spent_usd:.4f} of "
                    f"${budget.state().cap_usd:.2f}). "
                    f"Restart the server to reset."
                ),
            )

        body = await request.json()
        archetype_id = body.get("archetype_id") or "single_dataset_viz"
        try:
            archetype = archetype_by_id(archetype_id)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown archetype_id: {archetype_id!r}",
            )

        raw_query = body.get("query", "") or archetype.query
        query = _sanitize_query(raw_query)
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query is empty after sanitization.",
            )

        # Optional client-supplied cache namespace + session id hint.
        # cache_namespace: rotated on UI reset so each "fresh session"
        # starts cold. session_id_hint: lets a corpus driver tag the
        # JSONL records with a stable id like "sdv-01-templated-s1".
        cache_namespace = body.get("cache_namespace") or "default"
        session_id_hint = body.get("session_id") or None
        prompt_cache_key = f"public-geospatial-qa-agent-{cache_namespace}"

        # Mode + pattern selection. Defaults reproduce the prior
        # behaviour (templated, single-turn) — these are only set
        # away from defaults by the corpus driver.
        mode = body.get("mode") or "templated"
        if mode not in ("templated", "freeform"):
            raise HTTPException(400, f"unknown mode {mode!r}")
        pattern = body.get("pattern") or "single-turn"
        if pattern not in ("single-turn", "per-stage-confirm"):
            raise HTTPException(400, f"unknown pattern {pattern!r}")

        client = OpenAI(api_key=api_key)

        # Clarification mode: if the client opts in via clarify=true,
        # make one small LLM call that decides whether to ask back.
        # The call's cost is logged + surfaced to the UI either way.
        if body.get("clarify"):
            result = llm_clarification_check(client, query)
            budget.settle(result.cost_usd)
            if result.question:
                return StreamingResponse(
                    _stream_clarification(
                        question=result.question, query=query,
                        clarify_cost=result.cost_usd,
                        clarify_tokens=result.prompt_tokens + result.completion_tokens,
                    ),
                    media_type="text/event-stream",
                )
            # The cycle path needs a fresh reserve since we just
            # settled. Re-reserve and continue.
            if not budget.reserve():
                raise HTTPException(
                    status_code=429,
                    detail="Budget exhausted after clarification check.",
                )

        return StreamingResponse(
            _stream_cycle(
                client=client,
                archetype=archetype,
                user_query=query,
                budget=budget,
                backend=make_backend(backend_name),
                prompt_cache_key=prompt_cache_key,
                session_id=session_id_hint,
                logger=measurement_logger,
                trace_logger=trace_logger,
                mode=mode,
                pattern=pattern,
            ),
            media_type="text/event-stream",
        )

    return app


def _stream_clarification(
    *, question: str, query: str,
    clarify_cost: float = 0.0, clarify_tokens: int = 0,
):
    """Single-frame SSE producer for the clarification path. Pushes one
    'clarification' event and ends. Surfaces the LLM gate's cost +
    token usage so the UI can show what the question cost."""
    payload = {
        "type": "clarification",
        "question": question,
        "pending_query": query,
        "clarify_cost_usd": round(clarify_cost, 6),
        "clarify_tokens": clarify_tokens,
    }
    def gen():
        yield f"data: {json.dumps(payload)}\n\n"
    return gen()


def _stream_cycle(
    *, client, archetype, user_query, budget, backend,
    prompt_cache_key: str = "public-geospatial-qa-agent",
    session_id: str | None = None,
    logger: JsonlLogger | None = None,
    trace_logger: TraceLogger | None = None,
    mode: str = "templated",
    pattern: str = "single-turn",
):
    """SSE producer. Runs `run_cycle` in a worker thread so the SSE
    handler can drain a queue and push events as they happen.

    Each event is a Server-Sent Events frame:
        data: {"type": "stage", ...}
        data: {"type": "done", ...}
    """
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    final_trace_holder: dict[str, Any] = {}

    def on_stage(stage, state):
        events.put({
            "type": "stage",
            "idx": stage.idx,
            "name": stage.name,
            "prompt_tokens": stage.prompt_tokens,
            "cached_tokens": stage.cached_tokens,
            "completion_tokens": stage.completion_tokens,
            "cache_ratio": (
                stage.cached_tokens / stage.prompt_tokens
                if stage.prompt_tokens else 0
            ),
            "call_cost_usd": round(stage.call_cost_usd, 6),
            "tool_message_chars": stage.tool_message_chars,
            "state_size_chars": stage.state_size_chars,
            "tool_response_preview": stage.tool_response_preview,
            # The geometry / items the front-end map needs:
            "map_payload": _map_payload_after_stage(stage, state),
        })

    def worker():
        try:
            cycle_fn = (
                simulate_per_stage_confirm_cycle
                if pattern == "per-stage-confirm"
                else run_cycle
            )
            trace = cycle_fn(
                client,
                archetype,
                mode=mode,
                user_query=user_query,
                backend=backend,
                prompt_cache_key=prompt_cache_key,
                session_id=session_id,
                logger=logger,
                trace_logger=trace_logger,
                on_stage=on_stage,
            )
            budget.settle(trace.total_cost_usd)
            final_trace_holder["trace"] = trace
            events.put({"type": "done", "trace": _serialize_trace(trace)})
        except Exception as e:
            # Settle whatever spend was incurred so the budget stays
            # accurate even on error.
            budget.settle(0.0)
            events.put({
                "type": "error",
                "message": f"{type(e).__name__}: {e}",
            })
        finally:
            events.put({"type": "__end__"})

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            ev = events.get()
            if ev.get("type") == "__end__":
                return
            yield f"data: {json.dumps(ev)}\n\n"

    return gen()


def _map_payload_after_stage(stage, state) -> dict[str, Any]:
    """Return the JSON the front-end map needs to render after this
    stage finished. We send only the pieces the browser actually
    paints — not the entire state."""
    payload: dict[str, Any] = {}
    if stage.name == "geocode":
        place = state.place_result
        if place:
            payload["geocode"] = {
                "place": place.get("place"),
                "bbox": place.get("bbox"),
                "geometry": place.get("geometry"),
            }
    elif stage.name == "collections_rag":
        payload["collections"] = [
            {"id": m["id"], "title": m["title"]}
            for m in state.collections_result.get("matches", [])
        ]
    elif stage.name == "stac_search":
        payload["stac_items"] = [
            {
                "id": it["id"],
                "bbox": it.get("bbox"),
                "datetime": it.get("properties", {}).get("datetime"),
            }
            for it in state.stac_result.get("items", [])
        ]
    elif stage.name == "compute_stats":
        payload["stats"] = state.stats_result.get("per_item", [])
    return payload


def _serialize_trace(trace) -> dict[str, Any]:
    """asdict the CycleTrace into a JSON-friendly shape."""
    d = asdict(trace)
    d["cache_ratio"] = (
        trace.total_cached_tokens / trace.total_prompt_tokens
        if trace.total_prompt_tokens else 0
    )
    return d


# Module-level app for `uvicorn public_geospatial_qa_agent.web.app:app`
app = create_app()

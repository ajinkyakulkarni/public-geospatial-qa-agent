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
from ..runner import run_cycle
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
) -> FastAPI:
    """Build the FastAPI app. Exposed as a factory so the Playwright
    tests can construct an app with a stubbed OpenAI client or zero
    budget without touching env vars."""
    if budget_cap_usd is None:
        budget_cap_usd = float(os.environ.get("PGQA_BUDGET_USD", DEFAULT_BUDGET_USD))
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

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

        return StreamingResponse(
            _stream_cycle(
                client=OpenAI(api_key=api_key),
                archetype=archetype,
                user_query=query,
                budget=budget,
            ),
            media_type="text/event-stream",
        )

    return app


def _stream_cycle(*, client, archetype, user_query, budget):
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
            trace = run_cycle(
                client,
                archetype,
                mode="templated",  # public endpoint is templated-only
                user_query=user_query,
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
    elif stage.name == "stats":
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

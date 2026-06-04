"""Two tool-wrapper modes wired around a Backend.

Templated keeps the full structured payload in agent-side state and
returns a short status message to the model. Freeform serialises the
full payload back to the model. Same signatures in both modes; both
write the same data to state, so downstream stages don't care which
mode they're in.

The actual work — geocoding, STAC search, statistics — is done by
the Backend instance. Swapping CannedBackend for LiveBackend changes
the absolute token sizes but preserves the templated/freeform
contrast.

The default canned sizes are picked to match the calibrated
public-geospatial-qa preset on the calculator side: ~119 templated
tokens total across the six stages of a cycle, ~20,926 freeform.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backends import Backend, CannedBackend
from .state import AgentState

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_response_templates() -> dict[str, dict[str, str]]:
    """Load the response strings from data/response_templates.json."""
    p = DATA_DIR / "response_templates.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return json.loads(p.read_text())


# Status values returned in the JSON the model sees.
STATUS_COMPLETE = "complete"
STATUS_PENDING = "pending_confirmation"
STATUS_ERROR = "error"
STATUS_OUT_OF_SCOPE = "out_of_scope"


def build_tool_response(status: str, message: str, **extra: Any) -> str:
    """Return the JSON string that becomes the model-visible tool message."""
    return json.dumps({"status": status, "message": message, **extra})


class TemplatedTools:
    """Short-status-string mode. Each tool returns roughly 16-50 tokens
    to the model regardless of how big the underlying payload is."""

    def __init__(
        self,
        state: AgentState,
        response_templates: dict,
        backend: Backend,
    ):
        self.state = state
        self.templates = response_templates
        self.backend = backend

    def parse_datetime(self, value: str) -> str:
        self.state.datetime_range = value
        msg = self.templates["DATETIME_RESPONSES"]["success"].format(datetime=value)
        return build_tool_response(STATUS_PENDING, msg)

    def geocode(self, query: str) -> str:
        result = self.backend.geocode(query)
        if "error" in result:
            return build_tool_response(STATUS_ERROR, result["error"])
        self.state.place_result = result
        msg = self.templates["PLACE_RESPONSES"]["success"].format(
            place=result.get("place", query)
        )
        return build_tool_response(STATUS_PENDING, msg)

    def collections_rag(self, query: str, top_k: int = 5) -> str:
        result = self.backend.collections_rag(query, top_k)
        if "error" in result:
            return build_tool_response(STATUS_ERROR, result["error"])
        full_matches = result["matches"]
        self.state.collections_result = {
            "matches": full_matches,
            "collections": [m["id"] for m in full_matches],
        }
        options = [
            {"id": m["id"], "label": m.get("title", m["id"]),
             "is_cmr_backed": m.get("is_cmr_backed", False)}
            for m in full_matches
        ]
        msg = self.templates["COLLECTIONS_RESPONSES"]["success_multiple"].format(
            count=len(full_matches)
        )
        return build_tool_response(STATUS_PENDING, msg, options=options)

    def select_collection(self, collection_id: str,
                          selected_variable: str | None = None) -> str:
        self.state.selected_collection_id = collection_id
        self.state.selected_variable = selected_variable
        title = collection_id.replace("-", " ").title()
        if selected_variable:
            msg = self.templates["SELECT_COLLECTION_RESPONSES"][
                "success_with_variable"
            ].format(title=title, variable=selected_variable)
        else:
            msg = self.templates["SELECT_COLLECTION_RESPONSES"][
                "success"
            ].format(title=title)
        return build_tool_response(STATUS_COMPLETE, msg)

    def stac_search(self, collection_id: str, limit: int | None = None) -> str:
        n_items = limit or 15
        bbox = self.state.place_result.get("bbox") if self.state.place_result else None
        dt_range = self.state.datetime_range
        result = self.backend.stac_search(collection_id, bbox, dt_range, n_items)
        if "error" in result:
            return build_tool_response(STATUS_ERROR, result["error"])
        self.state.stac_result = result
        actual = result.get("matched", len(result.get("items", [])))
        msg = self.templates["STAC_RESPONSES"]["success"].format(
            retrieved=actual, plural="" if actual == 1 else "s"
        )
        return build_tool_response(STATUS_COMPLETE, msg, retrieved=actual)

    def stats(self) -> str:
        items = self.state.stac_result.get("items", [])
        bbox = self.state.place_result.get("bbox") if self.state.place_result else None
        result = self.backend.stats(items, bbox)
        if "error" in result:
            return build_tool_response(STATUS_ERROR, result["error"])
        self.state.stats_result = result
        count = result.get("count", len(items))
        msg = self.templates["STATS_RESPONSES"]["success"].format(
            count=count, plural="" if count == 1 else "s"
        )
        return build_tool_response(STATUS_COMPLETE, msg)

    def viz(self) -> str:
        items = self.state.stac_result.get("items", [])
        collection_id = self.state.selected_collection_id or ""
        result = self.backend.viz(items, collection_id)
        if "error" in result:
            return build_tool_response(STATUS_ERROR, result["error"])
        self.state.viz_result = result
        count = result.get("count", len(items))
        msg = self.templates["VIZ_RESPONSES"]["success"].format(
            count=count, plural="" if count == 1 else "s"
        )
        return build_tool_response(STATUS_COMPLETE, msg)


class FreeformTools:
    """The 'no templating layer' variant. Each tool returns the full
    structured payload to the LLM. This is what makes the per-stage
    input balloon to ~20-25K tokens by stage 6."""

    def __init__(
        self,
        state: AgentState,
        response_templates: dict,
        backend: Backend,
    ):
        self.state = state
        self.templates = response_templates
        self.backend = backend
        # Templated path populates state identically; freeform reads
        # from state and inlines the payload into the model message.
        self._templated = TemplatedTools(state, response_templates, backend)

    def parse_datetime(self, value: str) -> str:
        return self._templated.parse_datetime(value)

    def geocode(self, query: str) -> str:
        self._templated.geocode(query)
        if not self.state.place_result:
            return build_tool_response(STATUS_ERROR, "geocode failed")
        return json.dumps({
            "status": STATUS_PENDING,
            "message": f"Resolved '{query}'.",
            "geometry": self.state.place_result.get("geometry"),
            "bbox": self.state.place_result.get("bbox"),
            "place_id": self.state.place_result.get("place_id"),
        })

    def collections_rag(self, query: str, top_k: int = 5) -> str:
        self._templated.collections_rag(query, top_k)
        return json.dumps({
            "status": STATUS_PENDING,
            "message": "Top-K collections retrieved with full metadata.",
            "matches": self.state.collections_result.get("matches", []),
        })

    def select_collection(self, collection_id: str,
                          selected_variable: str | None = None) -> str:
        return self._templated.select_collection(collection_id, selected_variable)

    def stac_search(self, collection_id: str, limit: int | None = None) -> str:
        self._templated.stac_search(collection_id, limit)
        return json.dumps({
            "status": STATUS_COMPLETE,
            "message": "STAC search complete — see items below.",
            "items": self.state.stac_result.get("items", []),
            "matched": self.state.stac_result.get("matched", 0),
        })

    def stats(self) -> str:
        self._templated.stats()
        return json.dumps({
            "status": STATUS_COMPLETE,
            "message": "Statistics computed — full per-item table below.",
            "per_item": self.state.stats_result.get("per_item", []),
            "count": self.state.stats_result.get("count", 0),
        })

    def viz(self) -> str:
        self._templated.viz()
        return json.dumps({
            "status": STATUS_COMPLETE,
            "message": "Visualization layers built — see metadata below.",
            "layers": self.state.viz_result.get("layers", []),
            "count": self.state.viz_result.get("count", 0),
        })


def make_tools(
    mode: str,
    state: AgentState,
    backend: Backend | None = None,
):
    """Return the right tool set for the given mode.

    `mode` is 'templated' or 'freeform'. `backend` defaults to
    CannedBackend, which is what tests/ and the published measurement
    runs use.
    """
    if backend is None:
        backend = CannedBackend()
    templates = load_response_templates()
    if mode == "templated":
        return TemplatedTools(state, templates, backend)
    if mode == "freeform":
        return FreeformTools(state, templates, backend)
    raise ValueError(f"Unknown mode {mode!r} — use 'templated' or 'freeform'.")

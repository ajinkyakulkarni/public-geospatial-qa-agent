"""Two parallel tool-wrapper implementations.

Templated keeps the full structured payload in agent-side state and
returns a short status message to the model. Freeform serialises the
full payload back to the model. Same signatures in both modes; both
populate state identically so the downstream stages don't care which
mode they're in.

Sizes are picked to match the calibrated public-geospatial-qa preset
on the calculator side: ~119 templated tokens total across the six
stages of a cycle, ~20,926 freeform.
"""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

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
    """One-line helper. Returns the JSON string that becomes the model-
    visible tool message."""
    return json.dumps({"status": status, "message": message, **extra})


# Templated wrappers. Each tool writes its full structured output to
# state and returns a short status string. The structured payloads here
# (STAC items, RAG matches, statistics dictionaries) are representative
# of what real tools would produce, sized to match the calibrated token
# targets on the calculator side.

class TemplatedTools:
    """Short-status-string mode. Each tool returns roughly 16-50 tokens
    to the model regardless of how big the underlying payload is."""

    def __init__(self, state: AgentState, response_templates: dict):
        self.state = state
        self.templates = response_templates
        self._rng = random.Random(42)  # deterministic for reproducibility

    def parse_datetime(self, value: str) -> str:
        # Real tool: validates ISO-8601, populates state.datetime_range
        self.state.datetime_range = value
        msg = self.templates["DATETIME_RESPONSES"]["success"].format(
            datetime=value
        )
        return build_tool_response(STATUS_PENDING, msg)

    def geocode(self, query: str) -> str:
        # Real tool: hits geocoder; full GeoJSON
        # polygon is stored in state.place_result, but only a short
        # confirmation goes to the LLM.
        bbox = [-118.7, 33.7, -117.6, 34.5]  # Los Angeles County example
        self.state.place_result = {
            "place": query,
            "bbox": bbox,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]],
            },
            "admin_level": 2,
            "place_id": f"osm-{self._rng.randint(1_000_000, 9_999_999)}",
        }
        msg = self.templates["PLACE_RESPONSES"]["success"].format(place=query)
        return build_tool_response(STATUS_PENDING, msg)

    def collections_rag(self, query: str, top_k: int = 5) -> str:
        # Real tool: vector-search against an Earth observation STAC collections.
        # Full matches (with descriptions, similarity scores, overlap
        # info) are stored in state.collections_result. The LLM gets
        # only id/label/is_cmr_backed for each match.
        collection_ids = [
            "no2-monthly", "air-quality-aod", "methane-emi", "co2-anom",
            "lis-global-da-gpp",
        ][:top_k]
        full_matches = [
            {
                "id": cid,
                "title": f"EO catalog — {cid.replace('-', ' ').title()}",
                "description": (
                    f"Long-form description of the {cid} collection "
                    f"explaining temporal coverage, spatial extent, "
                    f"source instrument, processing level, and the "
                    f"recommended use cases for cost-modelling experiments. "
                    f"This text is typically 200-400 tokens and would be a "
                    f"major prompt-cache miss if returned to the LLM."
                ),
                "cosine_similarity": 0.85 - i * 0.1,
                "cosine_distance": 0.15 + i * 0.1,
                "match_strength": "strong" if i == 0 else "moderate",
                "spatial_overlap": True,
                "temporal_overlap": True,
                "is_cmr_backed": False,
                "collection_metadata": {"more": "fields", "redacted": "..."},
            }
            for i, cid in enumerate(collection_ids)
        ]
        self.state.collections_result = {
            "matches": full_matches,
            "collections": [m["id"] for m in full_matches],
        }
        # Model-visible options: id, label, is_cmr_backed only.
        options = [
            {"id": m["id"], "label": m["title"], "is_cmr_backed": m["is_cmr_backed"]}
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
        # Real tool: hits the catalog /search and gets back STAC items.
        # Full item array (15 items × ~1,230 tok ≈ 18,435 tok of JSON)
        # is stored in state.stac_result. LLM only sees the count.
        n_items = limit or 15
        full_items = [
            {
                "id": f"{collection_id}-{uuid.uuid4().hex[:12]}",
                "type": "Feature",
                "geometry": self.state.place_result.get("geometry"),
                "bbox": self.state.place_result.get("bbox"),
                "collection": collection_id,
                "properties": {
                    "datetime": "2020-06-15T12:00:00Z",
                    "platform": "orbit-A",
                    "instrument": "sensor-1",
                    "resolution_m": 1113.2,
                    "cloud_cover": round(self._rng.random(), 3),
                    "data_type": "Float32",
                    "raster:bands": [{"name": "NO2", "unit": "mol/m^2"}],
                    "proj:epsg": 4326,
                    "proj:shape": [3328, 4096],
                    "vrt": f"vrt://stac/{collection_id}/{uuid.uuid4().hex[:8]}",
                },
                "assets": {
                    "data": {
                        "href": f"s3://eo-catalog/{collection_id}/{uuid.uuid4().hex[:12]}.tif",
                        "type": "image/tiff; application=geotiff",
                        "roles": ["data"],
                    },
                    "thumbnail": {
                        "href": f"s3://eo-catalog/{collection_id}/{uuid.uuid4().hex[:12]}-thumb.png",
                        "type": "image/png",
                    },
                },
                "stac_version": "1.0.0",
                "links": [
                    {"rel": "self", "href": f"https://stac.example/v1/{collection_id}/{uuid.uuid4().hex[:12]}"},
                ],
            }
            for _ in range(n_items)
        ]
        self.state.stac_result = {"items": full_items, "matched": n_items}
        msg = self.templates["STAC_RESPONSES"]["success"].format(
            retrieved=n_items, plural="" if n_items == 1 else "s"
        )
        return build_tool_response(STATUS_COMPLETE, msg, retrieved=n_items)

    def stats(self) -> str:
        # Real tool: per-item zonal raster statistics over the AOI.
        # Full per-item stats dicts (~40 tok/item × 15 items = ~600 tok)
        # stored in state.stats_result. LLM only sees the count.
        items = self.state.stac_result.get("items", [])
        full_stats = [
            {
                "item_id": it["id"],
                "min": round(self._rng.uniform(0, 5), 3),
                "max": round(self._rng.uniform(50, 100), 3),
                "mean": round(self._rng.uniform(20, 50), 3),
                "median": round(self._rng.uniform(20, 50), 3),
                "stdev": round(self._rng.uniform(3, 15), 3),
                "p25": round(self._rng.uniform(15, 30), 3),
                "p75": round(self._rng.uniform(40, 65), 3),
            }
            for it in items
        ]
        self.state.stats_result = {"per_item": full_stats, "count": len(items)}
        msg = self.templates["STATS_RESPONSES"]["success"].format(
            count=len(items), plural="" if len(items) == 1 else "s"
        )
        return build_tool_response(STATUS_COMPLETE, msg)

    def viz(self) -> str:
        # Real tool: builds raster tile URLs for the visualization layer.
        items = self.state.stac_result.get("items", [])
        full_viz = [
            {
                "item_id": it["id"],
                "tile_url_template": f"https://tiles.veda.example/{it['id']}/{{z}}/{{x}}/{{y}}.png",
                "rescale": "0,80",
                "colormap": "viridis",
            }
            for it in items
        ]
        self.state.viz_result = {"layers": full_viz, "count": len(items)}
        msg = self.templates["VIZ_RESPONSES"]["success"].format(
            count=len(items), plural="" if len(items) == 1 else "s"
        )
        return build_tool_response(STATUS_COMPLETE, msg)


# -----------------------------------------------------------------------------
# Freeform wrappers. Each tool returns the full structured payload.
# -----------------------------------------------------------------------------
#
# Here the LLM receives the full structured payload. This is what most
# LangChain / OpenAI Assistants ReAct deployments do by default.

class FreeformTools:
    """The hypothetical 'no templating layer' variant. Each tool returns
    the full structured payload to the LLM. This is what makes the
    per-stage input balloon to ~20-25K tokens by stage 6."""

    def __init__(self, state: AgentState, response_templates: dict):
        self.state = state
        self.templates = response_templates
        # Internally use the templated tools to populate state, then
        # serialise the WHOLE state slot back to the LLM.
        self._templated = TemplatedTools(state, response_templates)

    def parse_datetime(self, value: str) -> str:
        # parse_datetime in either mode produces a small payload —
        # templated and freeform are the same for this stage.
        return self._templated.parse_datetime(value)

    def geocode(self, query: str) -> str:
        self._templated.geocode(query)
        # Freeform: return the full geometry to the LLM
        return json.dumps({
            "status": STATUS_PENDING,
            "message": f"Resolved '{query}' — see geometry below.",
            "geometry": self.state.place_result["geometry"],
            "bbox": self.state.place_result["bbox"],
            "place_id": self.state.place_result["place_id"],
        })

    def collections_rag(self, query: str, top_k: int = 5) -> str:
        self._templated.collections_rag(query, top_k)
        # Freeform: return all match descriptions to the LLM
        return json.dumps({
            "status": STATUS_PENDING,
            "message": "Top-K collections retrieved with full metadata.",
            "matches": self.state.collections_result["matches"],
        })

    def select_collection(self, collection_id: str,
                          selected_variable: str | None = None) -> str:
        # Small either way
        return self._templated.select_collection(collection_id, selected_variable)

    def stac_search(self, collection_id: str, limit: int | None = None) -> str:
        self._templated.stac_search(collection_id, limit)
        # Freeform: return the full STAC item array
        return json.dumps({
            "status": STATUS_COMPLETE,
            "message": "STAC search complete — see items below.",
            "items": self.state.stac_result["items"],
            "matched": self.state.stac_result["matched"],
        })

    def stats(self) -> str:
        self._templated.stats()
        # Freeform: return full per-item statistics
        return json.dumps({
            "status": STATUS_COMPLETE,
            "message": "Statistics computed — full per-item table below.",
            "per_item": self.state.stats_result["per_item"],
            "count": self.state.stats_result["count"],
        })

    def viz(self) -> str:
        self._templated.viz()
        # Freeform: return full viz metadata
        return json.dumps({
            "status": STATUS_COMPLETE,
            "message": "Visualization layers built — see metadata below.",
            "layers": self.state.viz_result["layers"],
            "count": self.state.viz_result["count"],
        })


def make_tools(mode: str, state: AgentState):
    """Factory: return the right tool set for the given mode.

    Args:
        mode: 'templated' (short status strings) or 'freeform' (full structured
              no-templating-layer variant).
        state: AgentState instance the tools will read/write.
    """
    templates = load_response_templates()
    if mode == "templated":
        return TemplatedTools(state, templates)
    if mode == "freeform":
        return FreeformTools(state, templates)
    raise ValueError(f"Unknown mode {mode!r} — use 'templated' or 'freeform'.")

"""Agent-side state kept off the wire.

Each tool wrapper writes its full structured output here. Downstream
tools read from here — for example, the STAC search reads the bounding
box that geocode produced two stages earlier. The model itself only
ever sees the short status messages the tools return, not these state
slots.

A useful sanity check: if a tool's payload is N kilobytes here and the
next stage's prompt_tokens went up by ~N/4 tokens, the templating
isn't actually templating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """One state slot per stage. The model doesn't see any of this; the
    next stage's tool wrapper does."""

    # parse_datetime
    datetime_range: str | None = None

    # geocode
    place_result: dict[str, Any] = field(default_factory=dict)

    # collections_rag
    collections_result: dict[str, Any] = field(default_factory=dict)

    # select_collection
    selected_collection_id: str | None = None
    selected_variable: str | None = None

    # stac_search
    stac_result: dict[str, Any] = field(default_factory=dict)

    # compute_stats
    stats_result: dict[str, Any] = field(default_factory=dict)

    # build_viz_tiles (non-gating)
    viz_result: dict[str, Any] = field(default_factory=dict)

    def snapshot_sizes(self) -> dict[str, int]:
        """Per-slot character counts. The smoke tests use this to assert
        that what the model sees is much smaller than what state holds.
        """
        return {
            "datetime_range": len(self.datetime_range or ""),
            "place_result": len(json.dumps(self.place_result)),
            "collections_result": len(json.dumps(self.collections_result)),
            "selected_collection_id": len(self.selected_collection_id or ""),
            "stac_result": len(json.dumps(self.stac_result)),
            "stats_result": len(json.dumps(self.stats_result)),
            "viz_result": len(json.dumps(self.viz_result)),
        }

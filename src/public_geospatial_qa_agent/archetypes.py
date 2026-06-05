"""Five representative user questions plus the fixed tool sequence.

The pipeline is the same six stages for every archetype because the
system prompt locks the order; this just makes the order explicit so
the runner doesn't have to ask the model "what next?" at each step.

The five questions cover the common shapes of geospatial Q&A traffic:
one dataset over one area in one time window; two datasets compared;
a long time window over one area; a spatial intersection; and a pure
catalog-discovery query with no analysis follow-through.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Archetype:
    id: str
    query: str
    pipeline: tuple[str, ...]


# The 6-stage pipeline every cycle traverses. Matches the system
# prompt's documented order.
DEFAULT_PIPELINE = (
    "parse_datetime",
    "geocode",
    "collections_rag",
    "select_collection",
    "stac_search",
    "compute_stats",
)

# build_viz_tiles is non-gating and can fire in the same turn as the
# stats stage. The default workflow stops at stage 6 to keep cycle
# costs comparable across runs; flip this to True if you want viz in
# the trace.
INCLUDE_VIZ = False

ALL_ARCHETYPES: list[Archetype] = [
    Archetype(
        id="single_dataset_viz",
        query=(
            "Show NO2 air quality data for Los Angeles County from "
            "January to June 2020."
        ),
        pipeline=DEFAULT_PIPELINE,
    ),
    Archetype(
        id="multi_dataset_comparison",
        query=(
            "Compare NO2 and CO2 trends over the Houston metro area "
            "for 2019 versus 2021."
        ),
        pipeline=DEFAULT_PIPELINE,
    ),
    Archetype(
        id="time_window_analysis",
        query=(
            "What were the monthly methane concentrations over the "
            "Permian Basin from 2018 through 2023?"
        ),
        pipeline=DEFAULT_PIPELINE,
    ),
    Archetype(
        id="spatial_intersection",
        query=(
            "Show vegetation health indicators where wildfires and "
            "drought overlapped in California in summer 2020."
        ),
        pipeline=DEFAULT_PIPELINE,
    ),
    Archetype(
        id="catalog_discovery",
        query=(
            "What datasets do you have for tracking sea surface "
            "temperature near the Gulf Coast?"
        ),
        pipeline=DEFAULT_PIPELINE,
    ),
]


def archetype_by_id(archetype_id: str) -> Archetype:
    for a in ALL_ARCHETYPES:
        if a.id == archetype_id:
            return a
    raise KeyError(f"no archetype with id={archetype_id!r}")

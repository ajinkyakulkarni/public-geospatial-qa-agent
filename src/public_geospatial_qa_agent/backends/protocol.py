"""The Backend protocol.

Each method returns a plain dict shaped the way the rest of the agent
expects. Whatever the implementation does internally — synthesise the
payload, call Nominatim, hit Planetary Computer — only the return
shape matters here.
"""
from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    """What the tool wrappers call to do real work."""

    def geocode(self, query: str) -> dict[str, Any]:
        """Return {place, bbox, geometry, place_id, admin_level}.

        bbox is [min_lon, min_lat, max_lon, max_lat]. geometry is a
        GeoJSON Polygon (or MultiPolygon) covering the area. place_id
        is whatever stable id the source provides.
        """
        ...

    def collections_rag(
        self, query: str, top_k: int
    ) -> dict[str, Any]:
        """Return {matches: [{id, title, description, score, ...}, ...]}.

        Up to `top_k` collections plausibly relevant to `query`, ranked.
        """
        ...

    def stac_search(
        self,
        collection_id: str,
        bbox: list[float] | None,
        datetime_range: str | None,
        limit: int,
    ) -> dict[str, Any]:
        """Return {items: [STAC Feature, ...], matched: int}.

        Each item is a GeoJSON Feature with the usual STAC properties
        block (datetime, eo:cloud_cover, assets, etc.).
        """
        ...

    def stats(
        self,
        items: list[dict[str, Any]],
        bbox: list[float] | None,
    ) -> dict[str, Any]:
        """Return {per_item: [{item_id, mean, ...}, ...], count}.

        One row per item. The canned backend fabricates statistics;
        the live backend reads what's already in the item properties
        and falls back to per-item summaries from the STAC summary
        block when present.
        """
        ...

    def viz(
        self,
        items: list[dict[str, Any]],
        collection_id: str,
    ) -> dict[str, Any]:
        """Return {layers: [{item_id, tile_url_template, ...}, ...], count}."""
        ...

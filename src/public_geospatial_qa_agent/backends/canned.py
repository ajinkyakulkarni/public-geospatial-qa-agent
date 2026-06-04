"""Deterministic synthetic payloads, sized to match the calibrated
public-geospatial-qa preset. No network, no I/O, seeded RNG.

The numbers in this file determine the templated/freeform contrast.
A freeform stac_search return ends up around 18-20K tokens because
each item carries the full STAC Feature shape; a templated return is
a few status bytes. If you swap this backend out for LiveBackend, the
token sizes shift to match whatever the real catalog returns — the
contrast is still there, just at different absolute numbers.
"""
from __future__ import annotations

import random
import uuid
from typing import Any


class CannedBackend:
    """Synthetic payloads, deterministic per-process."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def geocode(self, query: str) -> dict[str, Any]:
        # LA County as the default placeholder bbox.
        bbox = [-118.7, 33.7, -117.6, 34.5]
        return {
            "place": query,
            "bbox": bbox,
            "geometry": _bbox_polygon(bbox),
            "admin_level": 2,
            "place_id": f"osm-{self._rng.randint(1_000_000, 9_999_999)}",
        }

    def collections_rag(self, query: str, top_k: int) -> dict[str, Any]:
        collection_ids = [
            "no2-monthly", "air-quality-aod", "methane-emi", "co2-anom",
            "lis-global-da-gpp",
        ][:top_k]
        matches = [
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
                "cosine_similarity": round(0.85 - i * 0.1, 3),
                "cosine_distance": round(0.15 + i * 0.1, 3),
                "match_strength": "strong" if i == 0 else "moderate",
                "spatial_overlap": True,
                "temporal_overlap": True,
                "is_cmr_backed": False,
                "collection_metadata": {"more": "fields", "redacted": "..."},
            }
            for i, cid in enumerate(collection_ids)
        ]
        return {"matches": matches}

    def stac_search(
        self,
        collection_id: str,
        bbox: list[float] | None,
        datetime_range: str | None,
        limit: int,
    ) -> dict[str, Any]:
        n = limit
        bbox = bbox or [-118.7, 33.7, -117.6, 34.5]
        geometry = _bbox_polygon(bbox)
        items = [
            {
                "id": f"{collection_id}-{uuid.uuid4().hex[:12]}",
                "type": "Feature",
                "geometry": geometry,
                "bbox": bbox,
                "collection": collection_id,
                "properties": {
                    "datetime": "2020-06-15T12:00:00Z",
                    "platform": "orbit-A",
                    "instrument": "sensor-1",
                    "resolution_m": 1113.2,
                    "eo:cloud_cover": round(self._rng.random() * 100, 3),
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
                    {"rel": "self",
                     "href": f"https://stac.example/v1/{collection_id}/{uuid.uuid4().hex[:12]}"},
                ],
            }
            for _ in range(n)
        ]
        return {"items": items, "matched": n}

    def stats(
        self,
        items: list[dict[str, Any]],
        bbox: list[float] | None,
    ) -> dict[str, Any]:
        per_item = [
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
        return {"per_item": per_item, "count": len(items)}

    def viz(
        self,
        items: list[dict[str, Any]],
        collection_id: str,
    ) -> dict[str, Any]:
        layers = [
            {
                "item_id": it["id"],
                "tile_url_template": f"https://tiles.veda.example/{it['id']}/{{z}}/{{x}}/{{y}}.png",
                "rescale": "0,80",
                "colormap": "viridis",
            }
            for it in items
        ]
        return {"layers": layers, "count": len(items)}


def _bbox_polygon(bbox: list[float]) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]],
            [bbox[0], bbox[1]],
        ]],
    }

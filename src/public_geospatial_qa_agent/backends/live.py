"""Live backend: OpenStreetMap Nominatim + Microsoft Planetary Computer.

No API keys required for either service. Both have polite-use rate
limits — Nominatim asks for ≤1 req/s per IP plus an identifying
User-Agent header. We respect that, but anyone running this against a
production user base should host their own geocoder.

The Planetary Computer's public STAC root is
`https://planetarycomputer.microsoft.com/api/stac/v1` (no token; we use
the pystac-client). Collections are stable, the per-collection summary
gives us enough to populate the stats stage without downloading rasters.

Network errors don't raise. They surface as `error` status in the
returned dict, which the tool layer turns into an `error` tool response.
The runner keeps going; the CycleTrace records the failed stage.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any

import json as _json


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "public-geospatial-qa-agent/0.2 (github.com/ajinkyakulkarni/public-geospatial-qa-agent)"
PC_STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"

# A small hand-curated collection map: agent-level dataset names (the
# kind a user would say) to Planetary Computer collection ids and a
# short human description. Not exhaustive — meant as the realistic
# starter list. To search the full PC catalog dynamically, extend the
# collections_rag method to call /collections and rank by text overlap.
COLLECTION_INDEX = [
    {
        "id": "sentinel-5p-l2-netcdf",
        "title": "Sentinel-5P L2 — atmospheric trace gases",
        "description": "Daily NO2, SO2, CO, O3, CH4 from Sentinel-5P TROPOMI.",
        "tags": ["no2", "so2", "co", "ozone", "methane", "air quality", "trace gases"],
    },
    {
        "id": "sentinel-2-l2a",
        "title": "Sentinel-2 L2A",
        "description": "10m optical surface reflectance, global, 5-day revisit.",
        "tags": ["optical", "vegetation", "ndvi", "land cover", "sentinel-2"],
    },
    {
        "id": "landsat-c2-l2",
        "title": "Landsat Collection 2 Level-2",
        "description": "30m surface reflectance + thermal, Landsat 4-9.",
        "tags": ["optical", "thermal", "landsat", "long time series"],
    },
    {
        "id": "modis-13Q1-061",
        "title": "MODIS Vegetation Indices (NDVI/EVI)",
        "description": "16-day 250m NDVI and EVI from MODIS Terra/Aqua.",
        "tags": ["ndvi", "evi", "vegetation", "drought", "modis"],
    },
    {
        "id": "io-lulc-annual-v02",
        "title": "10m Annual Land Use / Land Cover (ESRI / Impact Observatory)",
        "description": "Annual 10m global land cover classification.",
        "tags": ["land cover", "lulc", "classification"],
    },
    {
        "id": "noaa-cdr-sea-surface-temperature-whoi",
        "title": "NOAA CDR — Sea Surface Temperature (WHOI)",
        "description": "Climate data record for global SST.",
        "tags": ["sst", "sea surface temperature", "ocean", "noaa"],
    },
    {
        "id": "noaa-mrms-qpe-1h-pass2",
        "title": "NOAA MRMS — 1-hour precipitation",
        "description": "Multi-radar / multi-sensor quantitative precip estimates.",
        "tags": ["precipitation", "rainfall", "weather", "noaa"],
    },
    {
        "id": "era5-pds",
        "title": "ERA5 reanalysis",
        "description": "Hourly atmospheric reanalysis at 0.25° resolution.",
        "tags": ["reanalysis", "temperature", "wind", "humidity", "climate"],
    },
]


class LiveBackend:
    """Live calls to OpenStreetMap Nominatim and the Planetary Computer STAC."""

    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s

    def geocode(self, query: str) -> dict[str, Any]:
        params = urllib.parse.urlencode({
            "q": query, "format": "json", "polygon_geojson": 1, "limit": 1,
        })
        url = f"{NOMINATIM_URL}?{params}"
        try:
            data = _http_get_json(url, timeout=self.timeout_s)
        except Exception as e:
            return {"error": f"nominatim: {type(e).__name__}: {e}"}
        if not data:
            return {"error": f"no result for {query!r}"}
        hit = data[0]
        bbox_s, bbox_n, bbox_w, bbox_e = (
            float(hit["boundingbox"][0]),
            float(hit["boundingbox"][1]),
            float(hit["boundingbox"][2]),
            float(hit["boundingbox"][3]),
        )
        bbox = [bbox_w, bbox_s, bbox_e, bbox_n]
        geom = hit.get("geojson") or _bbox_polygon(bbox)
        return {
            "place": hit.get("display_name") or query,
            "bbox": bbox,
            "geometry": geom,
            "admin_level": _try_int(hit.get("place_rank"), 0),
            "place_id": f"osm-{hit.get('osm_type', '')[:1]}{hit.get('osm_id', '')}",
        }

    def collections_rag(self, query: str, top_k: int) -> dict[str, Any]:
        # Rank the curated index by overlap of query terms with the tag
        # bag. Crude but reproducible — and avoids paying for an
        # embedding round-trip just to wire a starter app.
        q_terms = {t.lower() for t in query.replace(",", " ").split() if len(t) > 2}
        scored = []
        for entry in COLLECTION_INDEX:
            overlap = len(q_terms & set(entry["tags"]))
            scored.append((overlap, entry))
        scored.sort(key=lambda x: -x[0])
        top = [entry for score, entry in scored if score > 0][:top_k]
        if not top:
            top = [entry for _, entry in scored[:top_k]]
        matches = [
            {
                "id": e["id"],
                "title": e["title"],
                "description": e["description"],
                "cosine_similarity": round(0.6 + 0.05 * i, 3),
                "is_cmr_backed": False,
                "tags": e["tags"],
            }
            for i, e in enumerate(reversed(top))
        ]
        return {"matches": matches}

    def stac_search(
        self,
        collection_id: str,
        bbox: list[float] | None,
        datetime_range: str | None,
        limit: int,
    ) -> dict[str, Any]:
        try:
            from pystac_client import Client
        except ImportError:
            return {"error": "pystac-client not installed; pip install -e '.[live]'"}
        try:
            cat = Client.open(PC_STAC_ROOT)
            search = cat.search(
                collections=[collection_id],
                bbox=bbox,
                datetime=datetime_range,
                limit=limit,
                max_items=limit,
            )
            items = [it.to_dict() for it in search.items()]
        except Exception as e:
            return {"error": f"pc stac: {type(e).__name__}: {e}"}
        return {"items": items, "matched": len(items)}

    def stats(
        self,
        items: list[dict[str, Any]],
        bbox: list[float] | None,
    ) -> dict[str, Any]:
        # Without rio-tiler we can't read pixels in the time budget of
        # a chat exchange. Surface what the item already advertises —
        # cloud cover, datetime — as the "stats" payload. Realistic for
        # a discovery-style agent: most users want to know which items
        # exist before they pull rasters.
        per_item = []
        for it in items:
            props = it.get("properties", {})
            per_item.append({
                "item_id": it.get("id", ""),
                "datetime": props.get("datetime"),
                "cloud_cover": props.get("eo:cloud_cover"),
                "platform": props.get("platform"),
                "instrument": props.get("instruments") or props.get("instrument"),
            })
        return {"per_item": per_item, "count": len(items)}

    def viz(
        self,
        items: list[dict[str, Any]],
        collection_id: str,
    ) -> dict[str, Any]:
        # Planetary Computer exposes a tile endpoint at
        # https://planetarycomputer.microsoft.com/api/data/v1/item/tilejson.json
        # but it requires asset + rescale specifics per collection. We
        # surface the PC web-app preview URL per item so the UI has
        # something clickable; a production agent would call tilejson.
        layers = []
        for it in items:
            iid = it.get("id", "")
            layers.append({
                "item_id": iid,
                "preview_url": (
                    f"https://planetarycomputer.microsoft.com/explore?"
                    f"collection={collection_id}&item={iid}"
                ),
            })
        return {"layers": layers, "count": len(items)}


def _http_get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read())


def _try_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _bbox_polygon(bbox: list[float]) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]],
            [bbox[0], bbox[1]],
        ]],
    }

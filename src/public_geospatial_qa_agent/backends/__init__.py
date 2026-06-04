"""Backend protocol and registry.

A Backend is what the tool wrappers call to actually do the work:
geocode an address, hit a STAC catalog, compute zonal statistics.
There are two implementations.

CannedBackend returns deterministic synthetic payloads sized to match
the calibrated public-geospatial-qa preset. Offline, no network, every
run is byte-identical. This is what the prompt-cache measurements were
written against and what tests/ exercises.

LiveBackend talks to OpenStreetMap Nominatim and the Microsoft
Planetary Computer STAC. Use it when you want the agent to actually
work — type "Houston" and see the real Houston bounding box on the
map, ask for NO2 over LA and search a real catalog.

Both backends produce the same shape of result. The tool wrappers in
tools.py are backend-agnostic; flipping the backend doesn't change
what the LLM sees in templated mode, only what's behind the status
strings.
"""
from __future__ import annotations

from .protocol import Backend
from .canned import CannedBackend
from .live import LiveBackend


def make_backend(name: str) -> Backend:
    if name == "canned":
        return CannedBackend()
    if name == "live":
        return LiveBackend()
    raise ValueError(f"Unknown backend {name!r} — use 'canned' or 'live'.")


__all__ = ["Backend", "CannedBackend", "LiveBackend", "make_backend"]

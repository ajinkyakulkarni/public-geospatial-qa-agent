"""Six-stage geospatial dataset-discovery workflow against the OpenAI API,
instrumented so prompt-cache behavior is visible per stage.

Two pieces are shipped together: a CLI under cli.py for batch measurement
runs (one cycle, full suite, JSONL aggregate), and a small browser app
under web/ that streams stages over SSE and renders the geometry on a
Leaflet map. Both call into the same run_cycle function in runner.py.

There are two tool-return modes. Templated keeps heavy structured
payloads in agent-side state and sends short status strings to the
model. Freeform passes the full payload through. Same archetype + same
backend + same prompt_cache_key isolates the cache-rate difference
between the two.
"""

__version__ = "0.1.0"

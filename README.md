# public-geospatial-qa-agent

A small Python application that runs a six-stage geospatial dataset-discovery
workflow against the OpenAI API. Two pieces ship together: a CLI for batch
measurement runs, and a browser UI with a map for interactive testing. The
point is to make prompt-cache behavior in a multi-stage tool-calling workflow
observable end to end — token counts, cache hits, per-stage cost, and the
state that stays server-side instead of going to the model.

The workflow models a public earth-observation Q&A interface. A user asks a
question, the agent walks six tool stages (parse a date range, geocode a
place, search a dataset catalog, pick a collection, enumerate items in the
area, compute per-item statistics), and the result lands on the map. Tool
returns are kept to short status strings; heavy payloads stay in agent-side
state and never enter the model's context window.

There are two tool modes you can flip between. `templated` is the production
shape — short messages back to the model, large payloads in state. `freeform`
is the contrast — the full STAC item array, the full RAG match descriptions,
the full statistics dictionaries all go into the model's context. Same
archetype, same backend, same `prompt_cache_key`; the difference in
`cached_tokens` is what the two modes are there to surface.

There are also two backends. `canned` (the default) returns deterministic
synthetic payloads sized to match a calibrated cost preset — repeatable,
no network, no rate-limit risk. `live` calls OpenStreetMap Nominatim and
the Microsoft Planetary Computer STAC so the agent actually does what it
says: type "Houston" and the real Houston bbox lands on the map, ask for
sea surface temperature near the Gulf Coast and a real STAC catalog gets
searched. Templated/freeform contrast works in either backend.

## Setup

```bash
cd public-geospatial-qa-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .              # canned backend, offline
# Optional — install pystac-client for the live Planetary Computer backend:
pip install -e '.[live]'

cp .env.example .env          # then edit, dropping in your OPENAI_API_KEY
```

Tool schemas, the system prompt, and response templates live under `data/`
and are checked in.

## CLI

Four subcommands. They go from cheap to expensive.

```
show-config           # prints the loaded sysprompt + tool schemas, no API calls
run-once              # one cycle, six stages, single archetype (~$0.01)
run-suite             # five archetypes × N samples × two modes, JSONL logged
analyze               # roll up the JSONL into per-stage averages
serve                 # FastAPI app + browser UI on http://127.0.0.1:8000
```

`run-once` is the right place to start once the key is in place.

```bash
# Canned (offline, deterministic — what the measurement runs use):
python3 -m public_geospatial_qa_agent.cli run-once \
    --archetype single_dataset_viz --mode templated

# Live (OpenStreetMap + Planetary Computer):
python3 -m public_geospatial_qa_agent.cli run-once \
    --archetype catalog_discovery --mode templated --backend live
```

`run-suite` writes one JSON object per LLM call into `runs/measurement.jsonl`.
`analyze` reads that file and prints per-stage averages plus a monthly
extrapolation against the gpt-5.2 standard-tier rate card.

```bash
python3 -m public_geospatial_qa_agent.cli run-suite --samples 3 --budget 5.00
python3 -m public_geospatial_qa_agent.cli analyze --log runs/measurement.jsonl
```

`--budget` is a hard cap. The suite stops as soon as cumulative spend
crosses it. The web app uses a separate process-wide budget guard you can
override per launch. Run-suite defaults to the `canned` backend because
reproducible measurements need deterministic tool payloads — flip to
`--backend live` if you want to bench what the live STAC actually
returns instead.

## Browser UI

```bash
python3 -m public_geospatial_qa_agent.cli serve --budget 1.00 --backend live
```

Open `http://127.0.0.1:8000/`. The page is conversation-style: type a
question, the user bubble appears, the agent bubble fills in stage by
stage as the SSE stream arrives, and the map updates after the geocode
and STAC search. Ask a follow-up; the next turn's overlays draw in a
new colour so you can compare areas side by side on the map.

Each turn shows running token counts, cache share, and per-cycle cost
right below the agent's response. The cache warm-up is visible: the
first turn pays cold-cache rates on stage 1; later turns within the
same `prompt_cache_key` window hit warm rates from stage 1 onward. Keep
asking and watch the cached-share climb.

## Why the pipeline is fixed

The runner walks the six stages in a fixed order rather than letting the
model pick which tool to call next. Two reasons. First, the agent's system
prompt itself locks the order, so a model that follows instructions will
pick the same sequence anyway. Second, forcing the order means per-stage
token counts are comparable across samples and across the two tool modes;
if the model picked a different sequence each run, the cache numbers would
not line up. There's nothing stopping you from removing the forced sequence
and reading off whatever shape the model produces — the runner is one
function — but the published numbers in this repo are from the deterministic
walk.

## Project layout

```
public-geospatial-qa-agent/
├── pyproject.toml
├── README.md
├── .env.example
├── data/                            # checked in; written from scratch
│   ├── sysprompt.txt
│   ├── response_templates.json
│   └── tool_schemas.json
├── src/
│   └── public_geospatial_qa_agent/
│       ├── __init__.py
│       ├── state.py                 # agent-side state, kept off the wire
│       ├── tools.py                 # templated + freeform tool wrappers
│       ├── archetypes.py            # five representative user questions
│       ├── runner.py                # the six-stage cycle
│       ├── instrumentation.py       # one JSON object per LLM call
│       ├── cost.py                  # gpt-5.2 rate card
│       ├── cli.py                   # argparse entry point
│       ├── backends/                # tool data source
│       │   ├── protocol.py          # Backend Protocol
│       │   ├── canned.py            # deterministic synthetic payloads
│       │   └── live.py              # Nominatim + Planetary Computer
│       └── web/
│           ├── __init__.py
│           ├── app.py               # FastAPI + SSE
│           ├── budget.py            # thread-safe process budget
│           └── static/
│               ├── index.html       # conversation-style chat UI
│               ├── app.css
│               └── app.js
└── tests/
    └── test_smoke.py
```

## Reading the code

A reviewer who wants to confirm the load the model actually sees should walk
two files. `tools.py` is where each tool wrapper either writes the heavy
payload to `state` (templated) or hands it back to the model (freeform).
`runner.py` is where messages get assembled and `chat.completions.create` is
called once per stage. Everything between those two — `state.py`, `cost.py`,
`instrumentation.py`, the web app — is plumbing.

The smoke tests under `tests/test_smoke.py` pin the properties that have to
hold for the cache numbers to mean anything: templated mode keeps state at
least 10x larger than what reaches the model, freeform mode passes through
the full STAC item array unchanged, and the cost arithmetic agrees with hand
calculation. None of them hit the network; `pytest -v` is a fast first pass.

## Notes on the JSONL log

`runs/*.jsonl` records one line per LLM call: token counts, cache hits,
per-call cost, server-side state size, and the OpenAI `response_id`. It
does not record full prompts or full tool messages.

The `response_id` lines up with what shows on your OpenAI billing dashboard,
so the log is account-scoped — treat it as you would any billing artefact.
If you wire this into a multi-user deployment, drop either `user_query` or
`response_id` from the record; keeping both creates a privacy hazard you
don't need.

## Configuration via .env

The CLI loads `.env` at startup. The minimum is:

```
OPENAI_API_KEY=sk-...
PGQA_BUDGET_USD=1.00     # default budget for `serve`
```

`.env` is gitignored. `serve --budget X` overrides the env value per launch.

## Testing

```bash
pytest -v                                  # no API calls
python3 -m public_geospatial_qa_agent.cli show-config
python3 -m public_geospatial_qa_agent.cli run-once
```

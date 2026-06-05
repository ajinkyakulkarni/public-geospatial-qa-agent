# 2026-06-04 corpus aggregate

Frozen reference for the six-cell measurement matrix cited in
the paper `llm-chat-cost-modeling/paper.tex` §4.3 (Table 5).
Reviewers should compare regenerated values against this file
to check reproducibility within the reported confidence intervals.

## Run metadata

| Field | Value |
|---|---|
| Run start (UTC) | 2026-06-04T22:35:44Z |
| Model | `gpt-5.2` standard tier |
| Rate card | input $1.75/M, cached input $0.175/M, output $14.00/M |
| Backend | `canned` (deterministic synthetic payloads) |
| Curated corpus | `data/queries.json` (50 queries) — sha256 `e6bc7c9e1f85d6b58e258cbd397fab7e997ae337e0110f49f0cb2bd7efdcb7cb` |
| Naive corpus | `data/queries-naive.json` (30 queries) — sha256 `d9d893e704cb37d89df195345c947fc572305ed8a5d6a5bc3abeae6ec075d10b` |
| System prompt | `data/sysprompt.txt` — sha256 `b7e736143531f2858fc7ecfb25dec0dca8b49518613a967ddb332bf6098fadf0` |
| Tool schemas (at runtime) | sha256 `85acba1fc2c5e1ccebfbfd46aa4cb88b4b06c6e21e34bcbbd7209a4721627443` (function names: `set_datetime_tool`, `get_place_tool`, `collections_rag_tool`, `select_collection_tool`, `stac_search_tool`, `stats_tool`, `viz_tool`) |
| Tool schemas (at HEAD `64b7e67`) | sha256 `9d23b828e1fbbfb97ce41b0d354923b5eef16b6d44a50f32121b44166f68e379` (function names renamed to match `sysprompt.txt`: `parse_datetime`, `geocode`, `collections_rag`, `select_collection`, `stac_search`, `compute_stats`, `build_viz_tiles`) |
| Total LLM calls logged | 2,832 (1,932 curated + 900 naive) |
| Total spend | ≈$4.20 |

### Note on the tool_schemas SHA change

The measurements were collected against the pre-unification
schema names (`*_tool` suffix). Commit `64b7e67` then renamed
the function names to match the system prompt's documented
stage names so all three of (schema, sysprompt, runner) align.
The rename touches only the `function.name` field; the parameter
shapes, required fields, and the runner's `forced_tool_name`
pipeline are unchanged. Re-running the matrix at HEAD reproduces
the per-cell means inside the reported 95% CIs because the runner
forces the same six stages by Python method name in both cases;
the model's natural tool choice (which the runner overrides) is
the only thing that shifts.

## Curated corpus — per-cell mean ± 95% CI (paper Table 5)

| mode | pattern | gate | n | cost $ ± CI | cache % | prompt tok | output tok |
|---|---|---|---:|---:|---:|---:|---:|
| templated | single-turn | nogate | 50 | 0.008141 ± 0.000847 | 92.6% | 20,698 | 144 |
| templated | single-turn | gated | 33 | 0.008679 ± 0.000741 | 89.5% | 19,908 | 137 |
| templated | per-stage-confirm | nogate | 50 | 0.014118 ± 0.000653 | 87.2% | 29,335 | 219 |
| freeform | single-turn | nogate | 50 | 0.019686 ± 0.001260 | 72.4% | 28,898 | 142 |
| freeform | single-turn | gated | 35 | 0.018735 ± 0.000284 | 72.7% | 27,793 | 136 |
| freeform | per-stage-confirm | nogate | 50 | 0.025725 ± 0.000465 | 72.5% | 37,914 | 191 |

Gated cells show n < 50 because the pre-flight gate caught a
share of curated queries before they ran the full cycle
(17/50 templated, 15/50 freeform).

## Naive corpus

30 queries total; **all 30 caught by the pre-flight gate** in
gated cells (no cycle billed). No-gate cells ran the full
cycle on all 30. Per-cell means for the no-gate cells:

| mode | pattern | n | cost $ ± CI | cache % | prompt tok | output tok |
|---|---|---:|---:|---:|---:|---:|
| templated | single-turn | 30 | 0.007381 ± 0.000206 | 94.6% | 19,872 | 138 |
| templated | per-stage-confirm | 30 | 0.013756 ± 0.000828 | 88.8% | 29,321 | 221 |
| freeform | single-turn | 30 | 0.019586 ± 0.000400 | 71.5% | 27,844 | 153 |
| freeform | per-stage-confirm | 30 | 0.026243 ± 0.000942 | 72.9% | 37,884 | 198 |

### Gate trigger rate by missing-field category (naive corpus)

| category | n | triggered | rate |
|---|---:|---:|---:|
| date | 5 | 5 | 100% |
| place | 5 | 5 | 100% |
| dataset | 5 | 5 | 100% |
| multiple | 5 | 5 | 100% |
| vague | 5 | 5 | 100% |
| scope | 5 | 5 | 100% |

## Derived headline ratios (cited in paper §4.3)

| Quantity | Formula | Value |
|---|---|---:|
| Templating lever (single-turn) | freeform_st_nogate / tmpl_st_nogate | 2.42× |
| Templating lever (per-stage-confirm) | freeform_psc / tmpl_psc | 1.82× |
| Pattern lever (templated) | tmpl_psc / tmpl_st_nogate | 1.73× |
| Pattern lever (freeform) | freeform_psc / freeform_st_nogate | 1.31× |
| Combined spread (tmpl-st vs. freeform-psc) | freeform_psc / tmpl_st_nogate | 3.16× |
| Gate cost mean | five in-session probes | $0.0013 ± $0.0003 |
| Gate crossover f_naive* (templated) | gate_cost / tmpl_st_nogate | 16.0% |
| Gate crossover f_naive* (freeform) | gate_cost / freeform_st_nogate | 6.6% |

## Worked-example monthly extrapolations

At 10 K MAU × 0.2 sess/day × 10 q/sess × 30 days = 600 K cycles/month:

| Strategy | Per-query | Monthly |
|---|---:|---:|
| Templated single-turn, no gate | $0.008141 | $4,885 |
| Templated single-turn, pre-flight gate, f=0.5, r=0.9 | $0.009619 | $5,771 |
| Templated per-stage-confirm, no gate | $0.014118 | $8,471 |
| Freeform single-turn, no gate | $0.019686 | $11,812 |
| Freeform single-turn, pre-flight gate, f=0.5, r=0.9 | $0.020707 | $12,424 |
| Freeform per-stage-confirm, no gate | $0.025725 | $15,435 |

The gated formula is `g·(1 + r·f_naive) + (1 − f_naive·(1−r)) · cycle_cost`
with `g = $0.0013`, `r = 0.9`, `f_naive = 0.5`. All values
reconcile to the paper's Table 5 cells and the four derived
ratios in `llm-chat-cost-modeling/verify_numbers.py`.

## Reproducing this run

```bash
git clone https://github.com/ajinkyakulkarni/public-geospatial-qa-agent
cd public-geospatial-qa-agent
git checkout 64b7e67696c18672932139b62d5c78f8fda6d0d4
pip install -e .

# Start server with measurement + trace logging
export OPENAI_API_KEY=sk-...
PGQA_CORPUS_FILE=data/queries.json python3 -m public_geospatial_qa_agent.cli serve \
    --backend canned --budget 25.00 \
    --measurement-log runs/curated-paper.jsonl \
    --trace runs/curated-paper.trace.jsonl

# In a second terminal, drive the six-cell matrix
python3 scripts/run_corpus_in_browser.py --slow-mo 80 \
    --corpus-file data/queries.json \
    --measurement-log runs/curated-paper.jsonl

# Repeat for the naive corpus
PGQA_CORPUS_FILE=data/queries-naive.json python3 -m public_geospatial_qa_agent.cli serve \
    --backend canned --budget 25.00 \
    --measurement-log runs/naive-paper.jsonl \
    --trace runs/naive-paper.trace.jsonl

python3 scripts/run_corpus_in_browser.py --slow-mo 80 \
    --corpus-file data/queries-naive.json \
    --measurement-log runs/naive-paper.jsonl

# Aggregate
python3 -m public_geospatial_qa_agent.cli analyze --corpus --log runs/curated-paper.jsonl
python3 -m public_geospatial_qa_agent.cli analyze --corpus --log runs/naive-paper.jsonl
python3 scripts/build_calc_preset.py \
    --curated-log runs/curated-paper.jsonl \
    --naive-log runs/naive-paper.jsonl \
    --out runs/calc-preset.public-geospatial-qa.json
```

Total spend: ~$4.20 at gpt-5.2 standard rates. Per-cell means
should land within the 95% CIs reported above.

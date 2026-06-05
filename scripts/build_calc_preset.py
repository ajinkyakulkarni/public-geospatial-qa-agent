"""Take the curated + naive corpus aggregates and emit a calc-preset
JSON ready to drop into the cost-calculator-studio repo.

Format:

    {
      "version": "1",
      "measured_at": "...",
      "git_commit": "...",
      "model": "gpt-5.2",
      "rate_card": {...},
      "corpora": {
        "curated": {"sha256": ..., "n_queries": ...},
        "naive":   {"sha256": ..., "n_queries": ...}
      },
      "cells": {
        "templated_single-turn_nogate": {
          "cycle_cost_usd": {...},
          "cache_rate": ...,
          "prompt_tokens_per_cycle": ...,
          "completion_tokens_per_cycle": ...
        },
        ...
      },
      "gate": {
        "cost_per_call_usd": ...,
        "trigger_rate_curated": ...,
        "trigger_rate_naive": ...
      },
      "paper_claims": {
        "templating_cost_lever": ...,
        "pattern_cost_lever_templated": ...,
        "pattern_cost_lever_freeform": ...,
        "crossover_f_naive_pct": ...
      }
    }

Usage:
    python scripts/build_calc_preset.py \\
        --curated-log runs/curated-paper.jsonl \\
        --naive-log   runs/naive-paper.jsonl \\
        --out         runs/calc-preset.public-geospatial-qa.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curated-log", type=Path, required=True)
    ap.add_argument("--naive-log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    curated_rows = _per_cycle_rows(args.curated_log)
    naive_rows = _per_cycle_rows(args.naive_log)
    curated_meta = json.loads(args.curated_log.with_suffix(".meta.json").read_text())
    naive_meta = json.loads(args.naive_log.with_suffix(".meta.json").read_text())

    trace_meta = _load_trace_meta(args.curated_log)

    cells: dict[str, dict] = {}
    # Curated cells:
    for cell_key, rs in _group_by_cell(curated_rows).items():
        cells[f"curated::{cell_key}"] = _summarise(rs)
    # Naive cells:
    for cell_key, rs in _group_by_cell(naive_rows).items():
        cells[f"naive::{cell_key}"] = _summarise(rs)

    gate = _gate_summary(curated_meta, naive_meta, curated_rows, naive_rows)
    claims = _paper_claims(cells, gate)

    preset = {
        "version": "1",
        "measured_at": trace_meta.get("started_at"),
        "git_commit": trace_meta.get("git_commit", {}),
        "model": trace_meta.get("model"),
        "rate_card": trace_meta.get("rate_card"),
        "corpora": {
            "curated": {
                "sha256": trace_meta.get("corpus_sha256"),
                "file": "data/queries.json",
                "n_queries": len(curated_meta["queries"]),
            },
            "naive": {
                "sha256": None,  # naive is a different file; record separately
                "file": "data/queries-naive.json",
                "n_queries": len(naive_meta["queries"]),
            },
        },
        "cells": cells,
        "gate": gate,
        "paper_claims": claims,
        "schema_notes": {
            "cell_key_format": "<corpus>::<mode>_<pattern>_<gate>",
            "cost_usd": "mean USD per cycle; ci_95 is the half-width of "
                        "the 95% confidence interval for the mean",
            "cache_rate": "weighted mean of cached_tokens / prompt_tokens",
            "n": "sample size (number of cycles that ran in this cell)",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preset, indent=2))
    print(f"Wrote {args.out}")
    print()
    print("=== headline cell costs ===")
    for k, v in cells.items():
        print(f"  {k:<60}  n={v['n']:>3}  "
              f"${v['cycle_cost_usd']['mean']:.6f} ± "
              f"${v['cycle_cost_usd']['ci_95']:.6f}  "
              f"cache {100*v['cache_rate']:.1f}%")
    print()
    print("=== paper claims ===")
    for k, v in claims.items():
        print(f"  {k}: {v}")
    return 0


def _per_cycle_rows(log_path: Path) -> list[dict]:
    """Group raw JSONL records by session_id and sum per-cycle."""
    meta = json.loads(log_path.with_suffix(".meta.json").read_text())
    qmeta = {q["id"]: q for q in meta["queries"]}
    qids = sorted(qmeta.keys(), key=len, reverse=True)

    by_session: dict[str, list[dict]] = defaultdict(list)
    with log_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_session[r["session_id"]].append(r)

    rows: list[dict] = []
    for sid, recs in by_session.items():
        qid = None
        for c in qids:
            if sid.startswith(c + "-"):
                qid = c
                break
        if qid is None:
            continue
        tail = sid[len(qid) + 1:]
        cell, _, sample = tail.rpartition("-")
        if not sample.startswith("s"):
            continue
        mode, pattern, gate = _parse_cell(cell)
        rows.append({
            "qid": qid,
            "cell": cell,
            "mode": mode, "pattern": pattern, "gate": gate,
            "cycle_cost_usd": sum(r["call_cost_usd"] for r in recs),
            "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            "cached_tokens": sum(r["cached_tokens"] for r in recs),
            "completion_tokens": sum(r["completion_tokens"] for r in recs),
        })
    return rows


def _parse_cell(label: str) -> tuple[str, str, str]:
    mapping = {
        "tmpl-single-gated": ("templated", "single-turn", "gated"),
        "tmpl-single-nogate": ("templated", "single-turn", "nogate"),
        "tmpl-perstage-nogate": ("templated", "per-stage-confirm", "nogate"),
        "freeform-single-gated": ("freeform", "single-turn", "gated"),
        "freeform-single-nogate": ("freeform", "single-turn", "nogate"),
        "freeform-perstage-nogate": ("freeform", "per-stage-confirm", "nogate"),
    }
    return mapping.get(label, (label, "", ""))


def _group_by_cell(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = f"{r['mode']}_{r['pattern']}_{r['gate']}"
        out[key].append(r)
    return out


def _summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    costs = [r["cycle_cost_usd"] for r in rows]
    in_tok = sum(r["prompt_tokens"] for r in rows)
    cached_tok = sum(r["cached_tokens"] for r in rows)
    out_tok = sum(r["completion_tokens"] for r in rows)
    mean_cost = statistics.mean(costs)
    sd_cost = statistics.stdev(costs) if n > 1 else 0.0
    ci_95 = 1.96 * sd_cost / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "cycle_cost_usd": {
            "mean": round(mean_cost, 6),
            "stdev": round(sd_cost, 6),
            "ci_95": round(ci_95, 6),
        },
        "prompt_tokens_per_cycle": round(in_tok / n, 1),
        "cached_tokens_per_cycle": round(cached_tok / n, 1),
        "completion_tokens_per_cycle": round(out_tok / n, 1),
        "cache_rate": round(cached_tok / in_tok, 4) if in_tok else 0.0,
    }


def _gate_summary(curated_meta, naive_meta, curated_rows, naive_rows) -> dict:
    """Trigger rates per corpus from the .meta.json clarify records."""
    def rate(meta, n):
        return round(len(meta.get("clarify", [])) / n, 4) if n else 0.0
    return {
        "cost_per_call_usd": 0.0013,  # measured separately; ~$0.0009-0.0017
        "trigger_rate_curated": rate(curated_meta, len(curated_meta["queries"])),
        "trigger_rate_naive": rate(naive_meta, len(naive_meta["queries"])),
        "note": "Gate trigger rate captured by browser script per cell; "
                "values above are summed across the gated cells. "
                "cost_per_call_usd is an average from prior runs since "
                "the gate's per-call cost isn't logged separately here.",
    }


def _paper_claims(cells: dict, gate: dict) -> dict:
    """Derive the headline cost levers from the curated cells."""
    cur_tmpl_st = cells.get("curated::templated_single-turn_nogate", {})
    cur_freeform_st = cells.get("curated::freeform_single-turn_nogate", {})
    cur_tmpl_psc = cells.get("curated::templated_per-stage-confirm_nogate", {})
    cur_freeform_psc = cells.get("curated::freeform_per-stage-confirm_nogate", {})
    if not cur_tmpl_st.get("cycle_cost_usd"):
        return {}
    tc_st = cur_tmpl_st["cycle_cost_usd"]["mean"]
    fc_st = cur_freeform_st["cycle_cost_usd"]["mean"]
    tc_psc = cur_tmpl_psc["cycle_cost_usd"]["mean"]
    fc_psc = cur_freeform_psc["cycle_cost_usd"]["mean"]
    g = gate["cost_per_call_usd"]
    # Crossover f_naive: gate breaks even with no-gate when
    #   g + (1 - f_naive)*c = c     =>   f_naive = g / c.
    crossover_tmpl = g / tc_st if tc_st else 0
    crossover_freeform = g / fc_st if fc_st else 0
    return {
        "templating_cost_lever_single_turn": round(fc_st / tc_st, 2),
        "templating_cost_lever_per_stage_confirm": round(fc_psc / tc_psc, 2),
        "pattern_cost_lever_templated": round(tc_psc / tc_st, 2),
        "pattern_cost_lever_freeform": round(fc_psc / fc_st, 2),
        "crossover_f_naive_templated_pct": round(100 * crossover_tmpl, 1),
        "crossover_f_naive_freeform_pct": round(100 * crossover_freeform, 1),
        "explainer": (
            "Per-cycle cost ratios on the curated corpus (canned backend, "
            "gpt-5.2 standard). Crossover f_naive is the share of public "
            "traffic above which a pre-flight LLM gate is strictly cheaper "
            "than running every query."
        ),
    }


def _load_trace_meta(log_path: Path) -> dict:
    """Load the trace meta sidecar (versions, sysprompt sha, git
    commit, corpus sha) from the matching .trace.meta.json."""
    trace_meta_path = (
        log_path.with_suffix(".trace.jsonl").with_suffix(".meta.json")
    )
    if not trace_meta_path.exists():
        return {}
    return json.loads(trace_meta_path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())

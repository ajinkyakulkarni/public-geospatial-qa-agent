"""Command-line entry point.

Subcommands:

    show-config        Print the loaded sysprompt + tool schemas + a
                       sample template response. No API calls.
                       Useful for code review — you can see EXACTLY
                       what's sent to OpenAI without spending a cent.

    run-once           Run one cycle in one mode. Prints a stage-by-
                       stage table and the per-cycle cost. Useful for
                       smoke-testing your OpenAI key + budget.

    run-suite          Run all 5 archetypes × N samples × both modes.
                       Default N=3 → 30 cycles → ~$1 spend. Output
                       goes to a JSONL file + a summary table.

    analyze            Aggregate a JSONL log into per-stage cache
                       shares + monthly cost projections. No API
                       calls.

All commands accept --help.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env from the current working directory before reading
# OPENAI_API_KEY. dotenv.load_dotenv silently no-ops if the file
# doesn't exist, which is fine — the env var still wins.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on real shell env

from openai import OpenAI

from . import __version__
from .archetypes import ALL_ARCHETYPES, archetype_by_id
from .backends import make_backend
from .corpus import CorpusQuery, load_corpus
from .cost import GPT_5_2_STANDARD, monthly_extrapolation
from .instrumentation import JsonlLogger
from .runner import (
    llm_clarification_check,
    load_sysprompt,
    load_tool_schemas,
    run_cycle,
)


def cmd_show_config(args: argparse.Namespace) -> int:
    """Inspect the loaded sysprompt + tool schemas + a templated
    sample. Code-review friendly: no API calls."""
    sysprompt = load_sysprompt()
    schemas = load_tool_schemas()

    print("=" * 70)
    print(f"SYSPROMPT ({len(sysprompt)} chars)")
    print("=" * 70)
    print(sysprompt[:2000])
    if len(sysprompt) > 2000:
        print(f"\n... ({len(sysprompt) - 2000} more chars)")

    print()
    print("=" * 70)
    print(f"TOOL SCHEMAS ({len(schemas)} tools)")
    print("=" * 70)
    for s in schemas:
        fn = s["function"]
        print(f"\n  {fn['name']}")
        print(f"    description: {fn['description']}")
        print(f"    parameters:  {list(fn['parameters']['properties'].keys())}")

    print()
    print("=" * 70)
    print("SAMPLE TEMPLATED TOOL RESPONSES")
    print("=" * 70)
    print("(what the LLM sees from each tool — see src/.../tools.py)")
    try:
        import tiktoken
        from .state import AgentState
        from .tools import make_tools
        enc = tiktoken.get_encoding("o200k_base")
        state = AgentState()
        tools = make_tools("templated", state)
        for method, args in [
            ("parse_datetime", {"value": "2020-01-01/2020-06-30"}),
            ("geocode", {"query": "Los Angeles County"}),
            ("collections_rag", {"query": "NO2 air quality", "top_k": 5}),
            ("select_collection", {"collection_id": "no2-monthly"}),
            ("stac_search", {"collection_id": "no2-monthly", "limit": 15}),
            ("stats", {}),
        ]:
            content = getattr(tools, method)(**args)
            n_tok = len(enc.encode(content))
            print(f"\n  {method}() — {n_tok} tokens to LLM, {len(content)} chars")
            print(f"    {content[:250]}" + ("..." if len(content) > 250 else ""))
    except ImportError:
        print("  (install tiktoken to see token counts)")

    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    """Run one cycle in one mode against the OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2
    archetype = archetype_by_id(args.archetype)
    client = OpenAI(api_key=api_key)
    backend = make_backend(args.backend)
    print(f"Running cycle: archetype={archetype.id} mode={args.mode} backend={args.backend}")
    print(f"User query:    {archetype.query}")
    print()
    trace = run_cycle(client, archetype, args.mode, backend=backend)

    print(f"{'Stage':<26} {'prompt':>7} {'cached':>7} {'compl':>6}"
          f" {'cache%':>7} {'cost':>9}")
    print("-" * 67)
    for st in trace.stages:
        ratio = (st.cached_tokens / st.prompt_tokens) if st.prompt_tokens else 0
        print(f"  {st.idx} {st.name:<22} {st.prompt_tokens:>7} {st.cached_tokens:>7}"
              f" {st.completion_tokens:>6} {100 * ratio:>6.1f}%"
              f" ${st.call_cost_usd:>8.6f}")
    print("-" * 67)
    print(f"  TOTAL                      {trace.total_prompt_tokens:>7}"
          f" {trace.total_cached_tokens:>7} {trace.total_completion_tokens:>6}"
          f" {100 * trace.total_cached_tokens / max(1, trace.total_prompt_tokens):>6.1f}%"
          f" ${trace.total_cost_usd:>8.6f}")
    print()
    print(f"Per-cycle cost: ${trace.total_cost_usd:.6f}")
    print(f"Final server-side state size: {trace.final_state_size_chars} chars "
          f"(this NEVER reaches the LLM)")
    print()
    print("Extrapolated monthly at 600K cycles/month:    "
          f"${monthly_extrapolation(trace.total_cost_usd, 600_000):,.0f}")
    print("Extrapolated monthly at 900K cycles/month:    "
          f"${monthly_extrapolation(trace.total_cost_usd, 900_000):,.0f}")
    return 0


def cmd_run_suite(args: argparse.Namespace) -> int:
    """Run the full archetype × samples × modes suite."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key)
    backend = make_backend(args.backend)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    samples = args.samples
    modes = ["templated", "freeform"]
    n_total = len(ALL_ARCHETYPES) * samples * len(modes)
    n_done = 0
    spend = 0.0
    print(f"Running {n_total} cycles ({len(ALL_ARCHETYPES)} archetypes × "
          f"{samples} samples × {len(modes)} modes). "
          f"Budget cap: ${args.budget:.2f}. Log: {log_path}")
    print()

    with JsonlLogger(log_path) as logger:
        for archetype in ALL_ARCHETYPES:
            for sample_idx in range(samples):
                for mode in modes:
                    n_done += 1
                    label = f"  [{n_done}/{n_total}] {archetype.id}/{mode}/s{sample_idx + 1}"
                    print(f"{label} (spend so far ${spend:.4f})…")
                    trace = run_cycle(
                        client, archetype, mode,
                        session_id=f"{archetype.id}-{mode}-s{sample_idx + 1}",
                        logger=logger,
                        backend=backend,
                    )
                    spend += trace.total_cost_usd
                    if spend > args.budget:
                        print(f"  !! Budget cap ${args.budget} hit. Stopping.")
                        print(f"  Final spend: ${spend:.4f}")
                        return 0

    print()
    print(f"Done. Total spend: ${spend:.4f}. Log written to {log_path}.")
    print(f"Run `public-geospatial-qa-agent analyze --log {log_path}` for aggregates.")
    return 0


def cmd_run_corpus(args: argparse.Namespace) -> int:
    """Run the hand-curated query corpus through one or both modes.

    Writes one JSONL line per LLM call into the log path. Each record
    includes the corpus query id so analyze --corpus can slice by
    shape / place_size / dataset_family.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key)
    backend = make_backend(args.backend)
    corpus = load_corpus()
    if args.limit:
        corpus = corpus[: args.limit]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    modes = ["templated", "freeform"] if args.modes == "both" else [args.modes]
    n_total = len(corpus) * args.samples * len(modes)
    n_done = 0
    spend = 0.0

    # Clarify-call cost is logged into the same JSONL but tagged with
    # stage_name="clarify_check" + stage_idx=0 so analyze can split it
    # out of the cycle aggregates.
    clarify_records: list[dict] = []

    print(f"Running {n_total} cycles ({len(corpus)} queries × {args.samples} "
          f"samples × {len(modes)} modes). Budget cap: ${args.budget:.2f}. "
          f"Log: {log_path}. Backend: {args.backend}. "
          f"Clarify gate: {'on' if args.clarify else 'off'}.")
    print()

    with JsonlLogger(log_path) as logger:
        for cq in corpus:
            # One clarify call per query is enough — same query yields
            # the same decision; do not pay it per sample/mode.
            clarify_result = None
            if args.clarify:
                clarify_result = llm_clarification_check(client, cq.query)
                clarify_records.append({
                    "query_id": cq.id, "shape": cq.shape,
                    "place_size": cq.place_size,
                    "dataset_family": cq.dataset_family,
                    "clarify_question": clarify_result.question,
                    "clarify_prompt_tokens": clarify_result.prompt_tokens,
                    "clarify_cached_tokens": clarify_result.cached_tokens,
                    "clarify_completion_tokens": clarify_result.completion_tokens,
                    "clarify_cost_usd": round(clarify_result.cost_usd, 6),
                    "clarify_response_id": clarify_result.openai_response_id,
                })
                spend += clarify_result.cost_usd

            archetype = archetype_by_id(cq.shape)
            for sample_idx in range(args.samples):
                for mode in modes:
                    n_done += 1
                    label = (
                        f"  [{n_done}/{n_total}] {cq.id}/{mode}/s{sample_idx + 1}"
                    )
                    print(f"{label} (spend so far ${spend:.4f})…")
                    session_id = f"{cq.id}-{mode}-s{sample_idx + 1}"
                    trace = run_cycle(
                        client, archetype, mode,
                        session_id=session_id,
                        user_query=cq.query,
                        logger=logger,
                        backend=backend,
                    )
                    spend += trace.total_cost_usd
                    if spend > args.budget:
                        print(f"  !! Budget cap ${args.budget} hit. Stopping.")
                        print(f"  Final spend: ${spend:.4f}")
                        return _write_corpus_meta(log_path, corpus, clarify_records, spend)

    return _write_corpus_meta(log_path, corpus, clarify_records, spend)


def _write_corpus_meta(log_path: Path, corpus: list[CorpusQuery],
                       clarify_records: list[dict], spend: float) -> int:
    """Write the per-query metadata + per-clarify-call records next to
    the JSONL. analyze --corpus reads both."""
    meta = {
        "spend_usd": round(spend, 6),
        "queries": [
            {"id": q.id, "shape": q.shape, "place_size": q.place_size,
             "dataset_family": q.dataset_family, "query": q.query}
            for q in corpus
        ],
        "clarify": clarify_records,
    }
    meta_path = log_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print()
    print(f"Done. Total spend: ${spend:.4f}.")
    print(f"  Log:  {log_path}")
    print(f"  Meta: {meta_path}")
    print(f"Run `public-geospatial-qa-agent analyze --corpus --log {log_path}` for aggregates.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Aggregate a JSONL log into per-stage cache shares + cost
    projections. Read-only, no API calls."""
    path = Path(args.log)
    if not path.exists():
        print(f"ERROR: log file {path} does not exist.", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    if not records:
        print("ERROR: log file is empty.", file=sys.stderr)
        return 1

    if args.corpus:
        return _analyze_corpus(path, records, args)

    from collections import defaultdict
    import statistics

    by_mode_stage: dict[str, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        by_mode_stage[r["mode"]][r["stage_idx"]].append(r)

    for mode in ("templated", "freeform"):
        if not by_mode_stage[mode]:
            continue
        print(f"\n=== {mode} ===")
        print(f"  {'stage':<24}  {'prompt':>8} {'cached':>8} {'cache%':>7} {'cost':>10}  n")
        print(f"  {'-' * 70}")
        cycle_in_avg = 0.0
        cycle_cached_avg = 0.0
        cycle_out_avg = 0.0
        cycle_cost_avg = 0.0
        n_samples_cycle = 0
        for stage_idx in sorted(by_mode_stage[mode]):
            rs = by_mode_stage[mode][stage_idx]
            stage_name = rs[0]["stage_name"]
            avg_in = statistics.mean(r["prompt_tokens"] for r in rs)
            avg_cached = statistics.mean(r["cached_tokens"] for r in rs)
            avg_out = statistics.mean(r["completion_tokens"] for r in rs)
            avg_cost = statistics.mean(r["call_cost_usd"] for r in rs)
            ratio = avg_cached / avg_in if avg_in else 0
            print(f"  {stage_idx} {stage_name:<22}  {avg_in:>8.0f}"
                  f" {avg_cached:>8.0f} {100*ratio:>6.1f}% ${avg_cost:>8.6f}  {len(rs)}")
            cycle_in_avg += avg_in
            cycle_cached_avg += avg_cached
            cycle_out_avg += avg_out
            cycle_cost_avg += avg_cost
            n_samples_cycle = max(n_samples_cycle, len(rs))
        ratio = cycle_cached_avg / cycle_in_avg if cycle_in_avg else 0
        print(f"  {'CYCLE TOTAL':<24}  {cycle_in_avg:>8.0f}"
              f" {cycle_cached_avg:>8.0f} {100*ratio:>6.1f}% ${cycle_cost_avg:>8.6f}  n={n_samples_cycle}")
        for monthly_cycles in (600_000, 900_000):
            monthly = cycle_cost_avg * monthly_cycles
            print(f"  → at {monthly_cycles:>7,} cycles/mo: "
                  f"${monthly:,.0f}/month")

    return 0


def _analyze_corpus(log_path: Path, records: list[dict],
                    args: argparse.Namespace) -> int:
    """Per-axis aggregate report. Mean per-cycle cost + 95% CI broken
    down by archetype shape, by place_size, and by dataset_family.
    Emits a Markdown file next to the JSONL so the paper can include
    it directly."""
    import statistics
    import math
    from collections import defaultdict

    meta_path = log_path.with_suffix(".meta.json")
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Did you run via run-corpus?",
              file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text())
    query_meta = {q["id"]: q for q in meta["queries"]}

    # Each session_id is "<query_id>-<mode>-s<n>". Group records by
    # session, then sum per-session cost / tokens. Per-session
    # aggregates are what we average across the corpus.
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_session[r["session_id"]].append(r)

    rows = []
    for sid, recs in by_session.items():
        if "-" not in sid:
            continue
        # session_id shape: <qid>-<mode>-sN
        parts = sid.rsplit("-", 2)
        if len(parts) != 3:
            continue
        qid, mode, _sample = parts
        if qid not in query_meta:
            continue
        cycle_cost = sum(r["call_cost_usd"] for r in recs)
        cycle_in = sum(r["prompt_tokens"] for r in recs)
        cycle_cached = sum(r["cached_tokens"] for r in recs)
        cycle_out = sum(r["completion_tokens"] for r in recs)
        qm = query_meta[qid]
        rows.append({
            "query_id": qid, "mode": mode,
            "shape": qm["shape"],
            "place_size": qm["place_size"],
            "dataset_family": qm["dataset_family"],
            "cycle_cost_usd": cycle_cost,
            "cycle_prompt_tokens": cycle_in,
            "cycle_cached_tokens": cycle_cached,
            "cycle_completion_tokens": cycle_out,
            "cache_ratio": cycle_cached / cycle_in if cycle_in else 0,
        })

    def mean_ci(values: list[float]) -> tuple[float, float]:
        if len(values) < 2:
            return (statistics.mean(values) if values else 0.0, 0.0)
        m = statistics.mean(values)
        s = statistics.stdev(values)
        # 95% CI for the mean, normal approximation. Adequate for N>=10.
        ci = 1.96 * s / math.sqrt(len(values))
        return m, ci

    md = ["# Corpus aggregate", ""]
    md.append(f"Source: `{log_path.name}` ({len(records)} LLM calls across "
              f"{len(by_session)} cycles).")
    md.append(f"Total spend: ${meta['spend_usd']:.4f}.")
    md.append("")

    # ---- Table 8: per-shape mean + 95% CI ----
    md.append("## Per-archetype-shape per-cycle cost (mean ± 95% CI)")
    md.append("")
    md.append("| shape | mode | n | cost ($) | cache % | prompt tok | output tok |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    by_shape_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_shape_mode[(r["shape"], r["mode"])].append(r)
    for (shape, mode) in sorted(by_shape_mode):
        rs = by_shape_mode[(shape, mode)]
        cost_m, cost_ci = mean_ci([r["cycle_cost_usd"] for r in rs])
        cache_m, _ = mean_ci([r["cache_ratio"] for r in rs])
        in_m, _ = mean_ci([float(r["cycle_prompt_tokens"]) for r in rs])
        out_m, _ = mean_ci([float(r["cycle_completion_tokens"]) for r in rs])
        md.append(
            f"| {shape} | {mode} | {len(rs)} | "
            f"{cost_m:.6f} ± {cost_ci:.6f} | {100*cache_m:.1f}% | "
            f"{in_m:,.0f} | {out_m:,.0f} |"
        )
    md.append("")

    # ---- Table 9: per place_size ----
    md.append("## Per-cycle cost by place size (templated)")
    md.append("")
    md.append("| place_size | n | cost ($) | cache % |")
    md.append("|---|---:|---:|---:|")
    by_psize: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["mode"] != "templated":
            continue
        by_psize[r["place_size"]].append(r)
    for psize in sorted(by_psize, key=lambda k: -len(by_psize[k])):
        rs = by_psize[psize]
        cost_m, cost_ci = mean_ci([r["cycle_cost_usd"] for r in rs])
        cache_m, _ = mean_ci([r["cache_ratio"] for r in rs])
        md.append(
            f"| {psize} | {len(rs)} | "
            f"{cost_m:.6f} ± {cost_ci:.6f} | {100*cache_m:.1f}% |"
        )
    md.append("")

    # ---- Table 10: clarification trigger rate by shape ----
    md.append("## Clarification trigger rate by archetype shape")
    md.append("")
    clarify_records = meta.get("clarify", [])
    if not clarify_records:
        md.append("_No clarification calls were made (gate was off)._")
    else:
        md.append("| shape | queries | triggered | rate | clarify cost ($) |")
        md.append("|---|---:|---:|---:|---:|")
        by_shape_clarify: dict[str, list[dict]] = defaultdict(list)
        for cr in clarify_records:
            by_shape_clarify[cr["shape"]].append(cr)
        for shape in sorted(by_shape_clarify):
            crs = by_shape_clarify[shape]
            triggered = [c for c in crs if c["clarify_question"]]
            total_cost = sum(c["clarify_cost_usd"] for c in crs)
            md.append(
                f"| {shape} | {len(crs)} | {len(triggered)} | "
                f"{100*len(triggered)/len(crs):.1f}% | "
                f"{total_cost:.6f} |"
            )
    md.append("")

    md_path = log_path.with_suffix(".aggregate.md")
    md_path.write_text("\n".join(md))
    print("\n".join(md))
    print()
    print(f"Wrote {md_path}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="public-geospatial-qa-agent",
        description="Six-stage geospatial Q&A agent, instrumented for "
                    "OpenAI prompt-cache measurement.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("show-config", help="Print sysprompt + tool schemas + samples")
    p1.set_defaults(func=cmd_show_config)

    p2 = sub.add_parser("run-once", help="Run one cycle in one mode (one API hit)")
    p2.add_argument("--archetype", default="single_dataset_viz",
                    choices=[a.id for a in ALL_ARCHETYPES])
    p2.add_argument("--mode", default="templated", choices=["templated", "freeform"])
    p2.add_argument("--backend", default="canned", choices=["canned", "live"],
                    help="canned (default) for offline synthetic payloads; "
                         "live for Nominatim + Planetary Computer.")
    p2.set_defaults(func=cmd_run_once)

    p3 = sub.add_parser("run-suite", help="Run 5 archetypes × N samples × 2 modes")
    p3.add_argument("--samples", type=int, default=3)
    p3.add_argument("--budget", type=float, default=5.0,
                    help="Hard cap on USD spend (default $5)")
    p3.add_argument("--log", type=Path, default=Path("runs/measurement.jsonl"))
    p3.add_argument("--backend", default="canned", choices=["canned", "live"],
                    help="Use canned for reproducible measurements.")
    p3.set_defaults(func=cmd_run_suite)

    p4 = sub.add_parser("analyze", help="Aggregate a JSONL log; no API calls")
    p4.add_argument("--log", type=Path, default=Path("runs/measurement.jsonl"))
    p4.add_argument("--corpus", action="store_true",
                    help="Read the .meta.json sibling and emit per-axis "
                         "aggregates (shape, place size, dataset). Use this "
                         "after run-corpus.")
    p4.set_defaults(func=cmd_analyze)

    p6 = sub.add_parser(
        "run-corpus",
        help="Run the hand-curated 50-query corpus from data/queries.json",
    )
    p6.add_argument("--samples", type=int, default=1,
                    help="Samples per query×mode (default 1; raise for CIs)")
    p6.add_argument("--budget", type=float, default=10.0,
                    help="Hard cap on USD spend (default $10)")
    p6.add_argument("--log", type=Path, default=Path("runs/corpus.jsonl"))
    p6.add_argument("--backend", default="canned", choices=["canned", "live"])
    p6.add_argument("--modes", default="both",
                    choices=["templated", "freeform", "both"])
    p6.add_argument("--limit", type=int, default=None,
                    help="Run only the first N corpus queries (debugging)")
    p6.add_argument("--clarify", action="store_true",
                    help="Run an LLM clarification-gate call before each "
                         "cycle and record its cost separately.")
    p6.set_defaults(func=cmd_run_corpus)

    p5 = sub.add_parser("serve", help="Run the web UI + map interface")
    p5.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Default 127.0.0.1 (localhost only).")
    p5.add_argument("--port", type=int, default=8000)
    p5.add_argument("--budget", type=float, default=1.0,
                    help="Process-wide USD budget cap. Default $1.00.")
    p5.add_argument("--backend", default="canned", choices=["canned", "live"],
                    help="canned (default) for synthetic payloads; "
                         "live for OpenStreetMap + Planetary Computer.")
    p5.add_argument("--measurement-log", type=Path, default=None,
                    help="Local-use-only: write one JSONL line per LLM "
                         "call to this path. Off by default because the "
                         "JSONL pairs user_query with response_id; only "
                         "enable on a single-user local server.")
    p5.set_defaults(func=cmd_serve)

    return p


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI web UI server."""
    import uvicorn
    os.environ["PGQA_BUDGET_USD"] = str(args.budget)
    os.environ["PGQA_BACKEND"] = args.backend
    if args.measurement_log:
        os.environ["PGQA_MEASUREMENT_LOG"] = str(args.measurement_log)
        print(f"Measurement logging enabled → {args.measurement_log}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set — /api/ask will 500 until you "
              "set it.", file=sys.stderr)
    print(f"Starting server on http://{args.host}:{args.port}  "
          f"(budget cap ${args.budget:.2f}, backend={args.backend}). "
          f"Ctrl+C to stop.")
    uvicorn.run(
        "public_geospatial_qa_agent.web.app:app",
        host=args.host, port=args.port, reload=False,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

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
from .cost import GPT_5_2_STANDARD, monthly_extrapolation
from .instrumentation import JsonlLogger
from .runner import load_sysprompt, load_tool_schemas, run_cycle


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
    print(f"Running cycle: archetype={archetype.id} mode={args.mode}")
    print(f"User query:    {archetype.query}")
    print()
    trace = run_cycle(client, archetype, args.mode)

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
    p2.set_defaults(func=cmd_run_once)

    p3 = sub.add_parser("run-suite", help="Run 5 archetypes × N samples × 2 modes")
    p3.add_argument("--samples", type=int, default=3)
    p3.add_argument("--budget", type=float, default=5.0,
                    help="Hard cap on USD spend (default $5)")
    p3.add_argument("--log", type=Path, default=Path("runs/measurement.jsonl"))
    p3.set_defaults(func=cmd_run_suite)

    p4 = sub.add_parser("analyze", help="Aggregate a JSONL log; no API calls")
    p4.add_argument("--log", type=Path, default=Path("runs/measurement.jsonl"))
    p4.set_defaults(func=cmd_analyze)

    p5 = sub.add_parser("serve", help="Run the web UI + map interface")
    p5.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Default 127.0.0.1 (localhost only).")
    p5.add_argument("--port", type=int, default=8000)
    p5.add_argument("--budget", type=float, default=1.0,
                    help="Process-wide USD budget cap. Default $1.00.")
    p5.set_defaults(func=cmd_serve)

    return p


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI web UI server."""
    import uvicorn
    os.environ["PGQA_BUDGET_USD"] = str(args.budget)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set — /api/ask will 500 until you "
              "set it.", file=sys.stderr)
    print(f"Starting server on http://{args.host}:{args.port}  "
          f"(budget cap ${args.budget:.2f}). Ctrl+C to stop.")
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

"""Drive the chat UI through the corpus across the full mode × pattern
× gate matrix, headed so you can watch.

For each cell in CELLS, walks every query in the chosen corpus:
  - sets window.PGQA_MODE, PGQA_PATTERN, PGQA_SESSION_ID,
  - sets/clears the clarify toggle,
  - types the query, clicks Send,
  - waits for the agent bubble to show "Done." or a clarification,
  - clicks Reset.

The server writes one JSONL line per LLM call (via --measurement-log).
Each line's session_id encodes the cell label so analyze --corpus can
slice the records by mode/pattern/gate at the end.

Usage:
    # Terminal 1 — start the server with measurement logging on:
    python -m public_geospatial_qa_agent.cli serve \\
        --backend live --budget 25.00 \\
        --measurement-log runs/curated-matrix.jsonl

    # Terminal 2 — drive the UI through every cell:
    python scripts/run_corpus_in_browser.py \\
        --slow-mo 50 \\
        --corpus-file data/queries.json \\
        --measurement-log runs/curated-matrix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES_PATH = REPO_ROOT / "data" / "queries.json"


# The six measurement cells. Order matters: cheap cells first so a
# budget overrun caught early aborts before the expensive ones.
CELLS = [
    {"mode": "templated", "pattern": "single-turn",
     "clarify": True,  "label": "tmpl-single-gated"},
    {"mode": "templated", "pattern": "single-turn",
     "clarify": False, "label": "tmpl-single-nogate"},
    {"mode": "templated", "pattern": "per-stage-confirm",
     "clarify": False, "label": "tmpl-perstage-nogate"},
    {"mode": "freeform",  "pattern": "single-turn",
     "clarify": True,  "label": "freeform-single-gated"},
    {"mode": "freeform",  "pattern": "single-turn",
     "clarify": False, "label": "freeform-single-nogate"},
    {"mode": "freeform",  "pattern": "per-stage-confirm",
     "clarify": False, "label": "freeform-perstage-nogate"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--headed", action="store_true", default=True)
    parser.add_argument("--headless", dest="headed", action="store_false")
    parser.add_argument("--slow-mo", type=int, default=0,
                        help="ms of delay between Playwright actions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N queries (debugging)")
    parser.add_argument("--queries", default=None,
                        help="comma-separated query ids to run")
    parser.add_argument("--cells", default=None,
                        help="comma-separated cell labels to include "
                             "(defaults to all six)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--measurement-log", type=Path,
                        default=REPO_ROOT / "runs" / "matrix.jsonl",
                        help="JSONL the server is writing to; the .meta.json "
                             "sidecar lands next to it.")
    parser.add_argument("--corpus-file", type=Path,
                        default=DEFAULT_QUERIES_PATH)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is not installed. Install with:",
              file=sys.stderr)
        print("    pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2

    queries = _select_queries(args)
    cells = _select_cells(args)
    print(f"Driving {len(queries)} queries × {len(cells)} cells = "
          f"{len(queries) * len(cells)} runs.")
    print(f"Corpus: {args.corpus_file.name}")
    print(f"Cells: {', '.join(c['label'] for c in cells)}")
    print(f"Target: {args.base_url}")
    print()

    all_results: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed, slow_mo=args.slow_mo,
        )
        ctx = browser.new_context(viewport={"width": 1280, "height": 820})
        page = ctx.new_page()
        page.goto(args.base_url)
        page.wait_for_selector("#query")

        for cell_idx, cell in enumerate(cells, 1):
            print(f"=== Cell {cell_idx}/{len(cells)}: {cell['label']} ===")
            _apply_cell(page, cell)
            for i, q in enumerate(queries, 1):
                label = (f"  [{i}/{len(queries)}] {q['id']} "
                         f"({q['shape']}/{q['place_size']})")
                print(label)
                t0 = time.time()
                session_id = f"{q['id']}-{cell['label']}-s1"
                page.evaluate(
                    f"window.PGQA_SESSION_ID = {json.dumps(session_id)}"
                )
                try:
                    outcome = _run_one(page, q["query"],
                                        timeout_s=args.timeout)
                except Exception as e:
                    outcome = {"outcome": "error",
                                "detail": f"{type(e).__name__}: {e}"}
                outcome["query_id"] = q["id"]
                outcome["session_id"] = session_id
                outcome["cell"] = cell["label"]
                outcome["elapsed_s"] = round(time.time() - t0, 2)
                all_results.append(outcome)
                print(f"        → {outcome['outcome']} in {outcome['elapsed_s']}s")
                _reset_session(page)
            print()

        # Write the cross-cell artefacts. browser-corpus-run.json has
        # per-(cell × query) outcomes. The .meta.json sidecar gives
        # analyze --corpus the per-query metadata it needs.
        out_path = args.measurement_log.with_suffix(".browser-runs.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))

        queries_by_id = {q["id"]: q for q in queries}
        clarify_records = [
            {
                "query_id": r["query_id"],
                "cell": r["cell"],
                "shape": queries_by_id.get(r["query_id"], {}).get("shape"),
                "place_size": queries_by_id.get(r["query_id"], {}).get(
                    "place_size"),
                "dataset_family": queries_by_id.get(r["query_id"], {}).get(
                    "dataset_family"),
                "missing": queries_by_id.get(r["query_id"], {}).get("missing"),
                "clarify_question": r.get("question"),
                "clarify_cost_usd": 0.0,
            }
            for r in all_results if r["outcome"] == "clarification"
        ]
        meta_path = args.measurement_log.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "spend_usd": 0.0,
            "queries": queries,
            "cells": cells,
            "clarify": clarify_records,
            "source": "playwright-browser-driver",
            "corpus_file": str(args.corpus_file),
        }, indent=2))

        print()
        print(f"Wrote {out_path}")
        print(f"Wrote {meta_path}")
        print()
        print(f"Aggregate with:")
        print(f"    python -m public_geospatial_qa_agent.cli analyze "
              f"--corpus --log {args.measurement_log}")
        browser.close()
    return 0


def _select_queries(args: argparse.Namespace) -> list[dict]:
    raw = json.loads(args.corpus_file.read_text())
    qs = raw["queries"]
    if args.queries:
        wanted = set(args.queries.split(","))
        qs = [q for q in qs if q["id"] in wanted]
    if args.limit:
        qs = qs[: args.limit]
    return qs


def _select_cells(args: argparse.Namespace) -> list[dict]:
    if not args.cells:
        return list(CELLS)
    wanted = set(args.cells.split(","))
    return [c for c in CELLS if c["label"] in wanted]


def _apply_cell(page, cell: dict) -> None:
    page.evaluate(f"window.PGQA_MODE = {json.dumps(cell['mode'])}")
    page.evaluate(f"window.PGQA_PATTERN = {json.dumps(cell['pattern'])}")
    # The clarify checkbox is the canonical control. Force it to
    # match the cell's intent.
    cb = page.locator("#clarify")
    if cell["clarify"] and not cb.is_checked():
        cb.check()
    elif not cell["clarify"] and cb.is_checked():
        cb.uncheck()


def _run_one(page, query: str, *, timeout_s: int) -> dict:
    page.locator("#query").fill(query)
    page.locator("#ask").click()
    end_locator = page.get_by_text("Done.").last
    clarify_locator = page.locator(".agent-headline.clarification").last
    error_locator = page.locator(".stage-row.error").last
    deadline_ms = timeout_s * 1000
    start = time.time()
    while (time.time() - start) * 1000 < deadline_ms:
        if end_locator.count() > 0 and end_locator.is_visible():
            return {"outcome": "done"}
        if clarify_locator.count() > 0 and clarify_locator.is_visible():
            return {"outcome": "clarification",
                    "question": clarify_locator.inner_text()}
        if error_locator.count() > 0 and error_locator.is_visible():
            return {"outcome": "error",
                    "detail": error_locator.inner_text()}
        page.wait_for_timeout(250)
    return {"outcome": "timeout"}


def _reset_session(page) -> None:
    page.locator("#reset").click()
    page.wait_for_selector("#query")


if __name__ == "__main__":
    sys.exit(main())

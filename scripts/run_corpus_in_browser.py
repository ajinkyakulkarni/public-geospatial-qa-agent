"""Drive the chat UI through the corpus, headed so you can watch.

What this does:
  - Opens http://127.0.0.1:8000 in a real Chromium window.
  - For each query in data/queries.json:
      type → click Send → wait for the agent bubble to show "Done."
      (or a clarification) → click Reset.
  - The server logs every LLM call to a JSONL (passed via --measurement-log
    on `serve`). This script does not need to scrape anything; the server's
    log is the source of truth for cost + token numbers.

Usage:
    # Terminal 1 — start the server with measurement logging on:
    python -m public_geospatial_qa_agent.cli serve \\
        --backend live --measurement-log runs/corpus-browser.jsonl

    # Terminal 2 — drive the UI:
    python scripts/run_corpus_in_browser.py --headed --slow-mo 300

Then:
    python -m public_geospatial_qa_agent.cli analyze \\
        --corpus --log runs/corpus-browser.jsonl

Optional flags:
    --headless        run without a visible window
    --slow-mo MS      delay between actions (default 0)
    --limit N         only run the first N corpus queries
    --queries IDS     comma-separated query ids to run (e.g. sdv-01,mdc-03)
    --modes M         templated|freeform|both (default templated; freeform
                      is admin-only via the API and isn't exposed in the UI)
    --base-url URL    where the server is listening (default
                      http://127.0.0.1:8000)
    --no-clarify      don't tick the Ask-before-running checkbox

Limitations:
  - The UI POSTs /api/ask with clarify enabled by default. If the LLM gate
    asks a follow-up, this script records the clarification text and moves
    on without answering. The corpus is hand-written to be complete so
    this should be rare.
  - Only templated mode is reachable from /api/ask (freeform is admin-only).
    For a templated-vs-freeform comparison use `cli run-corpus` instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = REPO_ROOT / "data" / "queries.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--headed", action="store_true", default=True,
                        help="show the browser window (default on)")
    parser.add_argument("--headless", dest="headed", action="store_false")
    parser.add_argument("--slow-mo", type=int, default=0,
                        help="milliseconds of delay between actions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--queries", default=None,
                        help="comma-separated query ids to run")
    parser.add_argument("--no-clarify", action="store_true",
                        help="uncheck the clarification toggle before running")
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-query timeout in seconds")
    parser.add_argument("--measurement-log", type=Path,
                        default=REPO_ROOT / "runs" / "browser-corpus.jsonl",
                        help="JSONL the server is writing to; we emit "
                             "a .meta.json sidecar next to it so "
                             "analyze --corpus can join the records.")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is not installed. Install with:", file=sys.stderr)
        print("    pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2

    queries = _select_queries(args)
    print(f"Running {len(queries)} queries against {args.base_url}")
    print(f"  Make sure the server is running with --measurement-log "
          f"set if you want to capture aggregate numbers.")
    print()

    results: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed, slow_mo=args.slow_mo)
        ctx = browser.new_context(viewport={"width": 1280, "height": 820})
        page = ctx.new_page()
        page.goto(args.base_url)

        # Wait for the composer to mount.
        page.wait_for_selector("#query")

        if args.no_clarify and page.locator("#clarify").is_checked():
            page.locator("#clarify").uncheck()

        for i, q in enumerate(queries, 1):
            label = f"[{i}/{len(queries)}] {q['id']} ({q['shape']}/{q['place_size']})"
            print(f"  {label}")
            t0 = time.time()
            # Tag this turn with a stable session id so the server's
            # JSONL log records "<query-id>-templated-s1" instead of
            # a random uuid. Format matches what run-corpus emits, so
            # `analyze --corpus` can read both logs the same way.
            session_id = f"{q['id']}-templated-s1"
            page.evaluate(f"window.PGQA_SESSION_ID = {json.dumps(session_id)}")
            try:
                outcome = _run_one(page, q["query"], timeout_s=args.timeout)
            except Exception as e:
                outcome = {"outcome": "error", "detail": f"{type(e).__name__}: {e}"}
            outcome["query_id"] = q["id"]
            outcome["session_id"] = session_id
            outcome["elapsed_s"] = round(time.time() - t0, 2)
            results.append(outcome)
            print(f"      → {outcome['outcome']} in {outcome['elapsed_s']}s")
            _reset_session(page)

        out_path = REPO_ROOT / "runs" / "browser-corpus-run.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

        # Write the .meta.json sidecar next to the server's JSONL so
        # analyze --corpus can resolve session_id -> shape, place_size,
        # dataset_family.
        meta_path = args.measurement_log.with_suffix(".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            "spend_usd": 0.0,    # server already logs per-call cost
            "queries": queries,
            "clarify": [],
            "source": "playwright-browser-driver",
        }, indent=2))

        print()
        print(f"Wrote {out_path}.")
        print(f"Wrote {meta_path}.")
        print()
        print(f"To aggregate:")
        print(f"    python -m public_geospatial_qa_agent.cli analyze "
              f"--corpus --log {args.measurement_log}")
        browser.close()
    return 0


def _select_queries(args: argparse.Namespace) -> list[dict]:
    raw = json.loads(QUERIES_PATH.read_text())
    qs = raw["queries"]
    if args.queries:
        wanted = set(args.queries.split(","))
        qs = [q for q in qs if q["id"] in wanted]
    if args.limit:
        qs = qs[: args.limit]
    return qs


def _run_one(page, query: str, *, timeout_s: int) -> dict:
    page.locator("#query").fill(query)
    page.locator("#ask").click()
    # Either the agent finishes the cycle ("Done.") or the gate asks
    # a follow-up. Race the two locators.
    deadline_ms = timeout_s * 1000
    end_locator = page.get_by_text("Done.").last
    clarify_locator = page.locator(".agent-headline.clarification").last
    error_locator = page.locator(".stage-row.error").last
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
    # Reset should leave the welcome bubble + empty composer behind.
    page.wait_for_selector("#query")


if __name__ == "__main__":
    sys.exit(main())

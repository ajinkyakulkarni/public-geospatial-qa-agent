"""Sanity tests that don't touch the OpenAI API. Run these first,
before any billed run.

    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from public_geospatial_qa_agent.state import AgentState
from public_geospatial_qa_agent.tools import (
    TemplatedTools, FreeformTools, make_tools, load_response_templates,
)
from public_geospatial_qa_agent.cost import (
    GPT_5_2_STANDARD, cost_for_call, monthly_extrapolation,
)
from public_geospatial_qa_agent.archetypes import (
    ALL_ARCHETYPES, archetype_by_id, DEFAULT_PIPELINE,
)


# ---------------------------------------------------------------------
# Tests that don't read data/
# ---------------------------------------------------------------------

def test_archetypes_have_six_stage_pipelines():
    """Every archetype follows the canonical 6-stage pipeline."""
    for a in ALL_ARCHETYPES:
        assert len(a.pipeline) == 6
        assert a.pipeline == DEFAULT_PIPELINE


def test_archetype_lookup_by_id():
    a = archetype_by_id("single_dataset_viz")
    assert a.id == "single_dataset_viz"
    assert "Los Angeles" in a.query

    with pytest.raises(KeyError):
        archetype_by_id("does-not-exist")


def test_cost_card_present():
    """Sanity-check the rate card numbers haven't been mutated."""
    assert GPT_5_2_STANDARD.input_per_million == 1.75
    assert GPT_5_2_STANDARD.cached_input_per_million == 0.175
    assert GPT_5_2_STANDARD.output_per_million == 14.00


def test_cost_for_call_breaks_down_correctly():
    """fresh + cached + output, each at the right rate."""
    # 1,000 fresh input + 9,000 cached input + 100 output
    cost = cost_for_call(GPT_5_2_STANDARD, 10_000, 9_000, 100)
    # fresh: 1,000 * 1.75 / 1M = 0.00175
    # cached: 9,000 * 0.175 / 1M = 0.001575
    # output: 100 * 14 / 1M = 0.0014
    # total: 0.004725
    assert abs(cost - 0.004725) < 1e-9


def test_monthly_extrapolation_is_just_multiplication():
    assert monthly_extrapolation(0.005, 600_000) == 3000.0


# ---------------------------------------------------------------------
# Tests that need data/response_templates.json
# ---------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAVE_DATA = (DATA_DIR / "response_templates.json").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="data/response_templates.json missing")
def test_templated_tool_returns_short_strings():
    """The LLM-visible templated responses must be short status strings,
    NOT the full structured payloads. This is the most important
    invariant in the whole project."""
    state = AgentState()
    tools = make_tools("templated", state)

    # stac_search internally generates a 15-item STAC array stored in
    # state.stac_result. The LLM must NOT see those 15 items.
    msg = tools.stac_search(collection_id="no2-monthly", limit=15)
    assert len(msg) < 200, (
        f"templated stac_search returned {len(msg)} chars to the LLM; "
        f"expected < 200. The full STAC array must stay server-side. "
        f"Message: {msg!r}"
    )
    parsed = json.loads(msg)
    assert parsed["status"] == "complete"
    # The full items array MUST NOT be inside the message
    assert "items" not in parsed

    # But the items MUST be in state for downstream tools
    assert len(state.stac_result["items"]) == 15


@pytest.mark.skipif(not HAVE_DATA, reason="data/response_templates.json missing")
def test_freeform_tool_returns_full_payload():
    """Counterfactual: freeform mode WILL send the full payload to the LLM.

    Also pins the end-to-end invariant that whatever lives in
    state.stac_result['items'] is the EXACT same set the LLM sees in
    freeform mode — no truncation, no sampling. The measurement's
    contrast between templated and freeform is only meaningful if this
    holds; if FreeformTools later sub-samples to reduce its own cost,
    the lever number stops meaning what we say it means.
    """
    state = AgentState()
    tools = make_tools("freeform", state)

    msg = tools.stac_search(collection_id="no2-monthly", limit=15)
    assert len(msg) > 2000, (
        f"freeform stac_search returned {len(msg)} chars to the LLM; "
        f"expected > 2,000. The full STAC array must reach the LLM in "
        f"this mode."
    )
    parsed = json.loads(msg)
    assert "items" in parsed
    assert len(parsed["items"]) == 15
    # End-to-end invariant: LLM sees the same number of items that
    # state holds. If FreeformTools ever silently truncates the payload
    # this test catches it.
    assert len(parsed["items"]) == len(state.stac_result["items"]), (
        "FreeformTools.stac_search dropped items relative to state — "
        "the templated/freeform contrast is no longer apples-to-apples."
    )


@pytest.mark.skipif(not HAVE_DATA, reason="data/response_templates.json missing")
def test_state_isolation_under_templated_mode():
    """Walk a full cycle in templated mode and verify the server-side
    state grows substantially while the LLM-visible tool messages stay
    small."""
    state = AgentState()
    tools = make_tools("templated", state)

    llm_chars_total = 0
    llm_chars_total += len(tools.parse_datetime("2020-01-01/2020-06-30"))
    llm_chars_total += len(tools.geocode("Los Angeles County"))
    llm_chars_total += len(tools.collections_rag("NO2 air quality"))
    llm_chars_total += len(tools.select_collection("no2-monthly"))
    llm_chars_total += len(tools.stac_search("no2-monthly", limit=15))
    llm_chars_total += len(tools.stats())

    state_sizes = state.snapshot_sizes()
    state_total = sum(state_sizes.values())

    print(f"\n  LLM saw: {llm_chars_total} chars across 6 tool messages")
    print(f"  State (server-side, never sent to LLM): {state_total} chars")
    print(f"    breakdown: {state_sizes}")

    # The state should be at least 10x bigger than the LLM-visible
    # messages in templated mode (because the full STAC array, full
    # geometry, full stats are stored server-side).
    assert state_total > 10 * llm_chars_total, (
        f"templated mode is leaking too much to the LLM: state={state_total} "
        f"vs llm={llm_chars_total}; ratio {state_total/llm_chars_total:.1f}x "
        f"(expected >10x)"
    )


@pytest.mark.skipif(not HAVE_DATA, reason="data/response_templates.json missing")
def test_response_templates_load():
    """The response_templates JSON should have all the expected keys."""
    t = load_response_templates()
    assert "AGENT_RESPONSES" in t
    assert "DATETIME_RESPONSES" in t
    assert "PLACE_RESPONSES" in t
    assert "COLLECTIONS_RESPONSES" in t
    assert "SELECT_COLLECTION_RESPONSES" in t
    assert "STAC_RESPONSES" in t
    assert "STATS_RESPONSES" in t
    assert "VIZ_RESPONSES" in t

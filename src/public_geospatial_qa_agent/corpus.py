"""Corpus loader for the run-corpus / analyze --corpus pipeline.

A corpus is a list of CorpusQuery records pulled from data/queries.json.
Each query carries the metadata needed to slice the aggregate report:

    shape          one of the five archetype shapes
    place_size     city, county, state, region, country, continent, global
    dataset_family air_quality, vegetation, sst, methane, precipitation, etc.

run-corpus iterates the list, runs the cycle for each (query × mode ×
sample), and writes one JSONL record per LLM call. analyze --corpus
re-reads the JSONL, joins it back to the query metadata by id, and
emits per-axis means with 95% confidence intervals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class CorpusQuery:
    id: str
    shape: str
    place_size: str
    dataset_family: str
    query: str


def load_corpus(path: Path | None = None) -> list[CorpusQuery]:
    p = path or (DATA_DIR / "queries.json")
    raw = json.loads(p.read_text())
    return [CorpusQuery(**q) for q in raw["queries"]]


def query_by_id(qid: str, corpus: list[CorpusQuery] | None = None) -> CorpusQuery:
    corpus = corpus or load_corpus()
    for q in corpus:
        if q.id == qid:
            return q
    raise KeyError(f"no corpus query with id={qid!r}")

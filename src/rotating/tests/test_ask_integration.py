"""ask 门面在真实 Neo4j 与已入库语料上的契约测试。

需要 docker compose 起的 Neo4j，且不要把 .env 里的图谱清空；默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration

这里验证的是「线上那张图谱能被问出来」，因此断言的是图谱必须满足的契约：
事实带证据、并行边按概念对聚合、引用能回到具体页码与原文。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from idr_core.graph.neo4j import Neo4jGraphStore
from idr_rotating import AnswerPath, AskDeps, ask

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

QUESTION = "轴承外圈疲劳剥落有什么征兆"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not NEO4J_PASSWORD,
        reason="需要 NEO4J_PASSWORD（先 source .env）",
    ),
]


@pytest.fixture
def deps() -> Iterator[AskDeps]:
    graph_store = Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        yield AskDeps(graph_store=graph_store)
    finally:
        graph_store.close()


def test_ask_answers_from_the_live_graph_with_page_level_citations(
    deps: AskDeps,
) -> None:
    answer = ask(QUESTION, deps=deps)

    assert answer.path is AnswerPath.GRAPH, answer.detail
    assert answer.focus is not None
    assert answer.focus.concept.id == "concept:bearing_outer_race_fatigue_spalling"
    assert answer.facts

    relations = {fact.relation for fact in answer.facts}
    assert "indicated_by" in relations
    symptom_tails = {
        fact.tail.id for fact in answer.facts if fact.relation == "indicated_by"
    }
    assert "concept:bpfo_harmonics" in symptom_tails

    # 每条结论都必须带证据，每条证据都要回到具体页码与原文（ADR-0004）。
    cited_ids = {citation.evidence_id for citation in answer.citations}
    for fact in answer.facts:
        assert fact.evidence_ids
        assert set(fact.evidence_ids) <= cited_ids
        assert 0.0 < fact.confidence <= 1.0
    for citation in answer.citations:
        assert citation.page >= 1
        assert citation.text.strip()
        assert citation.doc_id


def test_ask_merges_parallel_edges_from_multiple_documents(deps: AskDeps) -> None:
    """同一概念对在不同文献里的并行边在答案里只出现一次，证据取并集。"""
    answer = ask(QUESTION, deps=deps)

    spectrum = [
        fact
        for fact in answer.facts
        if fact.tail.id == "concept:fft_spectrum_analysis"
    ]
    assert len(spectrum) == 1
    fact = spectrum[0]
    assert len(fact.evidence_ids) > 1

    documents = {
        citation.doc_id
        for citation in answer.citations
        if citation.evidence_id in fact.evidence_ids
    }
    assert len(documents) > 1


def test_ask_refuses_explicitly_for_a_question_outside_the_ontology(
    deps: AskDeps,
) -> None:
    answer = ask("量子退火有什么注意事项", deps=deps)

    assert answer.path is AnswerPath.REFUSAL
    assert answer.matches == ()
    assert answer.facts == ()
    assert answer.citations == ()
    assert answer.detail is not None

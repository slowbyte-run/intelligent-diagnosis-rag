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
from idr_rotating.traversal import EDGE_WHITELIST, MAX_PATHS

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


def test_ask_traverses_two_hops_from_an_equipment_class(deps: AskDeps) -> None:
    """设备→部件→故障模式：离心泵在真实图谱上够得到轴承的故障模式。"""
    answer = ask("离心泵有哪些故障模式", deps=deps)

    assert answer.path is AnswerPath.GRAPH, answer.detail
    assert answer.focus is not None
    assert answer.focus.concept.id == "concept:centrifugal_pump"

    two_hop = [fact for fact in answer.facts if len(fact.path) == 2]
    assert two_hop
    fault_facts = [
        fact for fact in answer.facts if fact.relation == "has_fault_mode"
    ]
    assert any(
        [step.relation for step in fact.path] == ["has_component", "has_fault_mode"]
        and fact.trail[1].id == "concept:rolling_bearing"
        for fact in fault_facts
    )
    assert any(
        fact.tail.id == "concept:bearing_outer_race_fatigue_spalling"
        for fact in fault_facts
    )

    # 每一跳都要能在引用层核对，且没有任何一跳越界。
    cited = {citation.evidence_id for citation in answer.citations}
    for fact in answer.facts:
        assert 1 <= len(fact.path) <= 2
        for step in fact.path:
            assert step.relation in EDGE_WHITELIST
            assert step.evidence_ids
            assert set(step.evidence_ids) <= cited


def test_ask_reaches_symptoms_two_hops_from_a_component(deps: AskDeps) -> None:
    """部件→故障模式→征兆：问句不点名故障模式时，征兆在第二跳上。"""
    answer = ask("轴有什么征兆", deps=deps)

    assert answer.path is AnswerPath.GRAPH, answer.detail
    assert answer.focus is not None
    assert answer.focus.concept.id == "concept:shaft"

    symptom_facts = [
        fact for fact in answer.facts if fact.relation == "indicated_by"
    ]
    assert symptom_facts
    for fact in symptom_facts:
        assert [step.relation for step in fact.path] == [
            "has_fault_mode",
            "indicated_by",
        ]
        assert fact.trail[0].id == "concept:shaft"
        assert fact.tail.type == "symptom"
        for step in fact.path:
            assert set(step.evidence_ids) <= {
                citation.evidence_id for citation in answer.citations
            }


def test_ask_truncates_two_hop_results_at_the_limit(deps: AskDeps) -> None:
    """真实图谱上两跳扇出会撞到上界：截断并在答案里说明。"""
    answer = ask("离心泵的滚动轴承有什么征兆", deps=deps)

    assert answer.path is AnswerPath.GRAPH, answer.detail
    assert answer.truncated is True
    assert len(answer.facts) == MAX_PATHS
    assert answer.detail is not None and "截断" in answer.detail


def test_ask_never_traverses_three_hops_on_the_live_graph(deps: AskDeps) -> None:
    """从设备出发两跳只能到故障模式；征兆需要第三跳，不得出现。"""
    answer = ask("离心泵有什么征兆", deps=deps)

    assert answer.path is AnswerPath.GRAPH, answer.detail
    assert answer.facts
    assert all(len(fact.path) <= 2 for fact in answer.facts)
    assert all(fact.tail.type != "symptom" for fact in answer.facts)


def test_live_graph_citations_point_at_the_cited_span_of_the_evidence(
    deps: AskDeps,
) -> None:
    """引用永远是它所属证据段的一段：有锚点取区间，无锚点（旧数据）取整段。"""
    answer = ask(QUESTION, deps=deps)

    assert answer.path is AnswerPath.GRAPH, answer.detail
    segments = {
        segment.id: segment
        for segment in deps.graph_store.evidence_by_ids(
            [citation.evidence_id for citation in answer.citations]
        )
    }
    assert len(segments) == len({citation.evidence_id for citation in answer.citations})

    for citation in answer.citations:
        segment = segments[citation.evidence_id]
        assert (citation.doc_id, citation.page) == (segment.doc_id, segment.page)
        if citation.anchor is None:
            # 锚点上线前入库的数据：整段引用，与旧行为逐字一致。
            assert citation.text == segment.text
            continue
        assert citation.anchor.evidence_id == citation.evidence_id
        assert 0 <= citation.anchor.start < citation.anchor.end <= len(segment.text)
        assert (
            citation.text
            == segment.text[citation.anchor.start : citation.anchor.end]
        )

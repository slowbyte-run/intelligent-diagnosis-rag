"""ask 门面：两跳遍历、边白名单、路径证据与结果上界。

与 ``test_ask.py`` 一样只在 ``ask`` 门面上断言，图谱用内存假存储；不连 Neo4j、
Postgres 或网络。两跳的语义是「从检索起点连走两条边」，每一步都必须能在答案
里指出用到了哪条关系、哪段证据。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fakes import FakeGraphStore
from idr_core.graph import (
    ConceptNode,
    ConceptRelation,
    GraphBatch,
    Provenance,
)
from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_rotating import AnswerPath, AskDeps, ask
from idr_rotating.traversal import EDGE_WHITELIST, MAX_PATHS

PUMP = "concept:centrifugal_pump"
BEARING = "concept:rolling_bearing"
OUTER_RACE = "concept:bearing_outer_race_fatigue_spalling"
BPFO = "concept:bpfo_harmonics"
ENVELOPE = "concept:envelope_demodulation"


class FakeDocumentNames:
    def __init__(self, names: dict[str, str]) -> None:
        self._names = dict(names)

    def name_for(self, doc_id: str) -> str | None:
        return self._names.get(doc_id)


def concept(instance_id: str, name: str, type_: str) -> ConceptNode:
    """本体规范实例对应的共享概念节点。"""
    return ConceptNode(
        id=f"concept:{instance_id}", name=name, type=type_, doc_id=None
    )


def evidence(doc_id: str, *, page: int, offset: int, text: str) -> EvidenceSegment:
    return EvidenceSegment(
        id=f"ev:{doc_id}:{page}:{offset}-{offset + len(text)}",
        doc_id=doc_id,
        page=page,
        bbox=BBox(0.0, 0.0, 595.0, 842.0),
        charspan=CharSpan(offset, offset + len(text)),
        text=text,
    )


def relation(
    head: ConceptNode,
    type_: str,
    tail: ConceptNode,
    *,
    doc_id: str,
    evidence_ids: tuple[str, ...],
    confidence: float = 0.8,
) -> ConceptRelation:
    return ConceptRelation(
        head_id=head.id,
        type=type_,
        tail_id=tail.id,
        provenance=Provenance(
            doc_id=doc_id,
            evidence_ids=evidence_ids,
            confidence=confidence,
            source="extracted",
        ),
    )


def write_document(
    graph: FakeGraphStore,
    doc_id: str,
    *,
    concepts: Sequence[ConceptNode],
    segments: Sequence[EvidenceSegment],
    relations: Sequence[ConceptRelation],
) -> None:
    graph.write_document(
        GraphBatch(
            doc_id=doc_id,
            concepts=tuple(concepts),
            evidence=tuple(segments),
            relations=tuple(relations),
            evidence_links=(),
        )
    )


@dataclass(frozen=True, slots=True)
class Chain:
    """一条准备好的两跳语料：依赖与各段证据的标识。"""

    deps: AskDeps
    component_evidence: str
    fault_evidence: str
    symptom_evidence: str
    detection_evidence: str


def two_hop_chain() -> Chain:
    """离心泵 → 滚动轴承 → 轴承外圈疲劳剥落 → 征兆。

    问句点名了设备与部件，但没点名故障模式：征兆只能沿「部件→故障模式→征兆」
    两条边走两次才拿得到。
    """
    graph = FakeGraphStore()
    doc_id = "doc-a"
    pump = concept(PUMP.removeprefix("concept:"), "离心泵", "equipment_class")
    bearing = concept(BEARING.removeprefix("concept:"), "滚动轴承", "component")
    outer = concept(
        OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode"
    )
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    envelope = concept(ENVELOPE.removeprefix("concept:"), "包络解调", "detection_method")

    component_evidence = evidence(
        doc_id, page=2, offset=0, text="离心泵包含滚动轴承与联轴器。"
    )
    fault_evidence = evidence(
        doc_id, page=3, offset=0, text="滚动轴承常见外圈疲劳剥落。"
    )
    symptom_evidence = evidence(
        doc_id, page=4, offset=0, text="外圈剥落会在频谱上出现 BPFO 及其倍频。"
    )
    detection_evidence = evidence(
        doc_id, page=5, offset=0, text="包络解调可以确认外圈剥落的冲击。"
    )

    write_document(
        graph,
        doc_id,
        concepts=(pump, bearing, outer, bpfo, envelope),
        segments=(
            component_evidence,
            fault_evidence,
            symptom_evidence,
            detection_evidence,
        ),
        relations=(
            relation(
                pump,
                "has_component",
                bearing,
                doc_id=doc_id,
                evidence_ids=(component_evidence.id,),
                confidence=0.9,
            ),
            relation(
                bearing,
                "has_fault_mode",
                outer,
                doc_id=doc_id,
                evidence_ids=(fault_evidence.id,),
                confidence=0.7,
            ),
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=(symptom_evidence.id,),
                confidence=0.6,
            ),
            relation(
                outer,
                "detected_by",
                envelope,
                doc_id=doc_id,
                evidence_ids=(detection_evidence.id,),
                confidence=0.8,
            ),
        ),
    )
    return Chain(
        deps=AskDeps(
            graph_store=graph,
            document_names=FakeDocumentNames({doc_id: "论文A.pdf"}),
        ),
        component_evidence=component_evidence.id,
        fault_evidence=fault_evidence.id,
        symptom_evidence=symptom_evidence.id,
        detection_evidence=detection_evidence.id,
    )


def test_two_hop_path_from_a_component_explains_the_route_to_a_symptom() -> None:
    chain = two_hop_chain()

    answer = ask("离心泵的滚动轴承有什么征兆", deps=chain.deps)

    assert answer.path is AnswerPath.GRAPH
    assert answer.focus is not None
    assert answer.focus.concept.id == BEARING
    # 关系类型决定结论顺序：征兆在前，检测手段其次，结构事实最后。
    assert [fact.relation for fact in answer.facts] == [
        "indicated_by",
        "detected_by",
        "has_fault_mode",
    ]

    symptom_facts = [fact for fact in answer.facts if fact.tail.id == BPFO]
    assert len(symptom_facts) == 1
    fact = symptom_facts[0]

    # 答案要说清「怎么走到的」，不只是终点。
    assert [node.name for node in fact.trail] == [
        "滚动轴承",
        "轴承外圈疲劳剥落",
        "外圈故障频率及其倍频",
    ]
    assert [
        (step.head.name, step.relation, step.tail.name) for step in fact.path
    ] == [
        ("滚动轴承", "has_fault_mode", "轴承外圈疲劳剥落"),
        ("轴承外圈疲劳剥落", "indicated_by", "外圈故障频率及其倍频"),
    ]

    # 多跳结论的可信度不会高于最弱的一步。
    assert fact.confidence == 0.6
    assert fact.evidence_ids == (chain.fault_evidence, chain.symptom_evidence)

    # 每一步都带证据，且证据都能回溯到引用层。
    cited = {citation.evidence_id for citation in answer.citations}
    for step in fact.path:
        assert step.evidence_ids
        assert set(step.evidence_ids) <= cited

    # 同一部件的检测手段是另一条一跳结论，证据同样要能回溯。
    detection_facts = [
        fact for fact in answer.facts if fact.relation == "detected_by"
    ]
    assert [fact.tail.id for fact in detection_facts] == [ENVELOPE]
    assert detection_facts[0].evidence_ids == (
        chain.fault_evidence,
        chain.detection_evidence,
    )


def test_equipment_question_reaches_fault_modes_through_a_component() -> None:
    """设备→部件→故障模式：起点是设备时，故障模式在第二跳上。"""
    chain = two_hop_chain()

    answer = ask("离心泵有哪些故障模式", deps=chain.deps)

    assert answer.path is AnswerPath.GRAPH
    assert answer.focus is not None
    assert answer.focus.concept.id == PUMP

    fault_facts = [
        fact for fact in answer.facts if fact.relation == "has_fault_mode"
    ]
    assert len(fault_facts) == 1
    fault_fact = fault_facts[0]
    assert fault_fact.head.id == PUMP
    assert fault_fact.tail.id == OUTER_RACE
    assert [node.name for node in fault_fact.trail] == [
        "离心泵",
        "滚动轴承",
        "轴承外圈疲劳剥落",
    ]
    assert [step.relation for step in fault_fact.path] == [
        "has_component",
        "has_fault_mode",
    ]

    # 一跳的部件结论同样有依据，不能因为能继续走就丢掉。
    component_facts = [
        fact for fact in answer.facts if fact.relation == "has_component"
    ]
    assert [fact.tail.id for fact in component_facts] == [BEARING]


def test_three_hop_chain_is_never_traversed() -> None:
    """征兆在第三跳上：两跳边界之外，答案里不能出现它。"""
    chain = two_hop_chain()

    answer = ask("离心泵有什么征兆", deps=chain.deps)

    assert answer.path is AnswerPath.GRAPH
    assert answer.focus is not None
    assert answer.focus.concept.id == PUMP
    # 离心泵→滚动轴承→轴承外圈疲劳剥落可达；再走到征兆需要第三跳。
    assert {fact.tail.id for fact in answer.facts} == {BEARING, OUTER_RACE}
    assert all(len(fact.path) <= 2 for fact in answer.facts)


def test_edge_whitelist_is_exactly_the_six_ticket_relations() -> None:
    assert EDGE_WHITELIST == (
        "has_component",
        "has_fault_mode",
        "has_cause",
        "indicated_by",
        "detected_by",
        "confused_with",
    )


def test_edges_outside_the_whitelist_are_never_traversed() -> None:
    """白名单外的边即使证据齐全也不进路径。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    outer = concept(
        OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode"
    )
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    envelope = concept(ENVELOPE.removeprefix("concept:"), "包络解调", "detection_method")
    symptom_evidence = evidence(
        doc_id, page=4, offset=0, text="外圈剥落会出现 BPFO。"
    )
    detection_evidence = evidence(
        doc_id, page=5, offset=0, text="包络解调可以确认冲击。"
    )
    write_document(
        graph,
        doc_id,
        concepts=(outer, bpfo, envelope),
        segments=(symptom_evidence, detection_evidence),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=(symptom_evidence.id,),
            ),
            relation(
                outer,
                "evidenced_by",
                envelope,
                doc_id=doc_id,
                evidence_ids=(detection_evidence.id,),
            ),
        ),
    )

    answer = ask(
        "轴承外圈疲劳剥落有什么征兆", deps=AskDeps(graph_store=graph)
    )

    assert answer.path is AnswerPath.GRAPH
    assert {fact.relation for fact in answer.facts} == {"indicated_by"}
    assert {fact.tail.id for fact in answer.facts} == {BPFO}


def broken_evidence_deps(
    *,
    symptom_evidence_ids: tuple[str, ...],
    store_symptom_evidence: bool,
) -> AskDeps:
    """滚动轴承→外圈疲劳剥落→征兆，第二跳的证据可按需缺失或查不到。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    bearing = concept(BEARING.removeprefix("concept:"), "滚动轴承", "component")
    outer = concept(
        OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode"
    )
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    fault_evidence = evidence(
        doc_id, page=3, offset=0, text="滚动轴承常见外圈疲劳剥落。"
    )
    symptom_evidence = evidence(
        doc_id, page=4, offset=0, text="外圈剥落会出现 BPFO。"
    )
    segments = (
        (fault_evidence, symptom_evidence)
        if store_symptom_evidence
        else (fault_evidence,)
    )
    write_document(
        graph,
        doc_id,
        concepts=(bearing, outer, bpfo),
        segments=segments,
        relations=(
            relation(
                bearing,
                "has_fault_mode",
                outer,
                doc_id=doc_id,
                evidence_ids=(fault_evidence.id,),
            ),
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=symptom_evidence_ids,
            ),
        ),
    )
    return AskDeps(graph_store=graph)


def test_conclusion_with_an_unevidenced_second_step_is_dropped() -> None:
    deps = broken_evidence_deps(
        symptom_evidence_ids=(), store_symptom_evidence=False
    )

    answer = ask("滚动轴承有什么征兆", deps=deps)

    assert answer.path is AnswerPath.GRAPH
    # 第二跳没有证据：整条征兆结论不存在，而不是降级成「部件有故障模式」。
    assert {fact.tail.id for fact in answer.facts} == {OUTER_RACE}
    assert all(len(fact.path) == 1 for fact in answer.facts)


def test_conclusion_whose_second_step_evidence_cannot_be_retrieved_is_dropped() -> None:
    deps = broken_evidence_deps(
        symptom_evidence_ids=("ev:ghost:9:0-3",),
        store_symptom_evidence=False,
    )

    answer = ask("滚动轴承有什么征兆", deps=deps)

    assert answer.path is AnswerPath.GRAPH
    assert {fact.tail.id for fact in answer.facts} == {OUTER_RACE}
    assert all(len(fact.path) == 1 for fact in answer.facts)


def test_conclusion_with_an_unevidenced_first_step_is_dropped() -> None:
    """第一跳缺证据时，第二跳的句柄不该凭空接在第一跳上。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    bearing = concept(BEARING.removeprefix("concept:"), "滚动轴承", "component")
    outer = concept(
        OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode"
    )
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    symptom_evidence = evidence(
        doc_id, page=4, offset=0, text="外圈剥落会出现 BPFO。"
    )
    write_document(
        graph,
        doc_id,
        concepts=(bearing, outer, bpfo),
        segments=(symptom_evidence,),
        relations=(
            relation(
                bearing,
                "has_fault_mode",
                outer,
                doc_id=doc_id,
                evidence_ids=(),
            ),
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=(symptom_evidence.id,),
            ),
        ),
    )

    answer = ask(
        "滚动轴承有什么征兆", deps=AskDeps(graph_store=graph)
    )

    assert answer.path is AnswerPath.REFUSAL
    assert answer.facts == ()
    assert answer.citations == ()
    assert answer.detail is not None and "证据" in answer.detail


def graph_with_fault_modes(count: int) -> AskDeps:
    """离心泵→滚动轴承，再挂 ``count`` 个故障模式；用来观察结果集上界。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    pump = concept(PUMP.removeprefix("concept:"), "离心泵", "equipment_class")
    bearing = concept(BEARING.removeprefix("concept:"), "滚动轴承", "component")
    shared_evidence = evidence(
        doc_id, page=2, offset=0, text="离心泵包含滚动轴承，轴承有若干故障模式。"
    )
    concepts = [pump, bearing]
    relations = [
        relation(
            pump,
            "has_component",
            bearing,
            doc_id=doc_id,
            evidence_ids=(shared_evidence.id,),
        )
    ]
    for index in range(count):
        fault = concept(
            f"test_fault_{index:02d}", f"测试故障模式{index:02d}", "fault_mode"
        )
        concepts.append(fault)
        relations.append(
            relation(
                bearing,
                "has_fault_mode",
                fault,
                doc_id=doc_id,
                evidence_ids=(shared_evidence.id,),
            )
        )
    write_document(
        graph,
        doc_id,
        concepts=concepts,
        segments=(shared_evidence,),
        relations=relations,
    )
    return AskDeps(graph_store=graph)


def test_result_set_is_bounded_and_the_answer_says_it_was_truncated() -> None:
    deps = graph_with_fault_modes(MAX_PATHS + 5)

    answer = ask("离心泵有哪些故障模式", deps=deps)

    assert answer.path is AnswerPath.GRAPH
    assert answer.truncated is True
    assert len(answer.facts) == MAX_PATHS
    assert answer.detail is not None
    assert "截断" in answer.detail
    assert str(MAX_PATHS) in answer.detail
    assert all(len(fact.path) == 2 for fact in answer.facts)


def test_result_set_at_the_limit_is_not_flagged_as_truncated() -> None:
    """恰好等于上界不算截断：没有丢东西就不该报警。"""
    deps = graph_with_fault_modes(MAX_PATHS - 1)

    answer = ask("离心泵有哪些故障模式", deps=deps)

    assert answer.path is AnswerPath.GRAPH
    assert answer.truncated is False
    assert answer.detail is None
    assert len(answer.facts) == MAX_PATHS

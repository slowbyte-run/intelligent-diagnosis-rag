"""ask 门面：别名链接、一跳事实、引用与检索路径。

单元测试只依赖内存假存储与本体 YAML，不连 Neo4j、Postgres 或网络。
"""

from __future__ import annotations

from collections.abc import Sequence

from idr_core.graph import (
    ConceptNode,
    ConceptRelation,
    GraphBatch,
    Provenance,
)
from fakes import FakeGraphStore
from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_rotating import AnswerPath, AskDeps, LinkKind, ask

OUTER_RACE = "concept:bearing_outer_race_fatigue_spalling"
BPFO = "concept:bpfo_harmonics"
ENVELOPE = "concept:envelope_demodulation"


class FakeDocumentNames:
    """按 doc_id 查文献名，模拟命令行注入的语料目录。"""

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


def deps_with_one_hop_facts() -> AskDeps:
    """一篇文献：外圈剥落有征兆与检测手段各一条，写入顺序与答案顺序相反。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    symptom_evidence = evidence(
        doc_id,
        page=4,
        offset=0,
        text="外圈剥落会在频谱上出现 BPFO 及其倍频。",
    )
    detection_evidence = evidence(
        doc_id,
        page=5,
        offset=100,
        text="包络解调可以确认外圈剥落的冲击。",
    )
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    envelope = concept(ENVELOPE.removeprefix("concept:"), "包络解调", "detection_method")
    write_document(
        graph,
        doc_id,
        concepts=(outer, bpfo, envelope),
        segments=(symptom_evidence, detection_evidence),
        relations=(
            relation(
                outer,
                "detected_by",
                envelope,
                doc_id=doc_id,
                evidence_ids=(detection_evidence.id,),
                confidence=0.7,
            ),
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=(symptom_evidence.id,),
                confidence=0.8,
            ),
        ),
    )
    return AskDeps(
        graph_store=graph,
        document_names=FakeDocumentNames({doc_id: "论文A.pdf"}),
    )


def test_ask_returns_one_hop_facts_with_citations() -> None:
    answer = ask("轴承外圈疲劳剥落有什么征兆", deps=deps_with_one_hop_facts())

    assert answer.question == "轴承外圈疲劳剥落有什么征兆"
    assert answer.path is AnswerPath.GRAPH
    assert answer.detail is None
    assert answer.focus is not None
    assert answer.focus.concept.id == OUTER_RACE
    assert answer.focus.alias == "轴承外圈疲劳剥落"

    # 关系白名单的顺序就是结论顺序：征兆在前、检测手段在后，与写入顺序无关。
    assert [(fact.relation, fact.tail.name) for fact in answer.facts] == [
        ("indicated_by", "外圈故障频率及其倍频"),
        ("detected_by", "包络解调"),
    ]
    symptom_fact = answer.facts[0]
    assert symptom_fact.head.id == OUTER_RACE
    assert symptom_fact.confidence == 0.8
    detection_fact = answer.facts[1]
    assert detection_fact.confidence == 0.7

    citations = {citation.evidence_id: citation for citation in answer.citations}
    assert set(citations) == {
        symptom_fact.evidence_ids[0],
        detection_fact.evidence_ids[0],
    }
    symptom_citation = citations[symptom_fact.evidence_ids[0]]
    assert symptom_citation.document == "论文A.pdf"
    assert symptom_citation.doc_id == "doc-a"
    assert symptom_citation.page == 4
    assert symptom_citation.text == "外圈剥落会在频谱上出现 BPFO 及其倍频。"


def empty_deps() -> AskDeps:
    """空图谱：只观察链接与检索路径，不掺入事实。"""
    return AskDeps(graph_store=FakeGraphStore())


def test_ask_explains_which_alias_linked_each_concept() -> None:
    answer = ask("离心泵的轴承外圈疲劳剥落有什么征兆", deps=empty_deps())

    by_id = {match.concept.id: match for match in answer.matches}
    assert by_id[OUTER_RACE].alias == "轴承外圈疲劳剥落"
    assert by_id[OUTER_RACE].kind is LinkKind.SUBSTRING
    assert by_id["concept:centrifugal_pump"].alias == "离心泵"
    # 焦点取最具体的命中：故障模式，而不是同一句里的设备类别。
    assert answer.focus is not None
    assert answer.focus.concept.id == OUTER_RACE
    # 图谱里没有这个节点，但链接结果仍如实带回，解释「问题被怎么理解」。
    assert answer.path is AnswerPath.REFUSAL
    assert answer.facts == ()
    assert answer.detail is not None and "图谱" in answer.detail


def test_ask_reports_exact_alias_matches_for_a_bare_term() -> None:
    answer = ask("轴承外圈疲劳剥落", deps=empty_deps())

    match = next(
        item for item in answer.matches if item.concept.id == OUTER_RACE
    )
    assert match.alias == "轴承外圈疲劳剥落"
    assert match.kind is LinkKind.EXACT


def test_ask_refuses_without_guessing_when_concepts_are_equally_specific() -> None:
    answer = ask("磨粒磨损有什么征兆", deps=empty_deps())

    assert answer.path is AnswerPath.REFUSAL
    assert answer.focus is None
    assert answer.facts == ()
    assert answer.citations == ()
    assert {
        "concept:bearing_abrasive_wear",
        "concept:gear_tooth_abrasive_wear",
    } <= {match.concept.id for match in answer.matches}
    assert answer.detail is not None and "多个" in answer.detail


def test_ask_refuses_explicitly_when_no_concept_matches() -> None:
    answer = ask("今天天气怎么样", deps=empty_deps())

    assert answer.path is AnswerPath.REFUSAL
    assert answer.matches == ()
    assert answer.focus is None
    assert answer.facts == ()
    assert answer.citations == ()
    assert answer.detail is not None and "匹配" in answer.detail


def test_ask_merges_parallel_edges_into_one_fact_with_all_citations() -> None:
    """同一概念对跨文献的并行边聚合成一条结论，证据取并集。"""
    graph = FakeGraphStore()
    first, second = "doc-a", "doc-b"
    first_evidence = evidence(first, page=3, offset=0, text="第一篇文献的证据。")
    second_evidence = evidence(second, page=7, offset=0, text="第二篇文献的证据。")
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        first,
        concepts=(outer, bpfo),
        segments=(first_evidence,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=first,
                evidence_ids=(first_evidence.id,),
                confidence=0.6,
            ),
        ),
    )
    write_document(
        graph,
        second,
        concepts=(outer, bpfo),
        segments=(second_evidence,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=second,
                evidence_ids=(second_evidence.id,),
                confidence=0.9,
            ),
        ),
    )

    answer = ask(
        "轴承外圈疲劳剥落有什么征兆",
        deps=AskDeps(
            graph_store=graph,
            document_names=FakeDocumentNames(
                {first: "论文A.pdf", second: "论文B.pdf"}
            ),
        ),
    )

    assert answer.path is AnswerPath.GRAPH
    assert len(answer.facts) == 1
    fact = answer.facts[0]
    assert fact.evidence_ids == (first_evidence.id, second_evidence.id)
    assert fact.confidence == 0.9
    assert [
        (citation.document, citation.page) for citation in answer.citations
    ] == [("论文A.pdf", 3), ("论文B.pdf", 7)]


def test_ask_leaves_out_facts_without_evidence() -> None:
    graph = FakeGraphStore()
    doc_id = "doc-a"
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        doc_id,
        concepts=(outer, bpfo),
        segments=(),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=(),
            ),
        ),
    )

    answer = ask(
        "轴承外圈疲劳剥落有什么征兆", deps=AskDeps(graph_store=graph)
    )

    assert answer.path is AnswerPath.REFUSAL
    assert answer.facts == ()
    assert answer.citations == ()
    assert answer.detail is not None and "证据" in answer.detail


def test_ask_leaves_out_facts_whose_evidence_cannot_be_retrieved() -> None:
    """引用回溯不到原段落的事实不进答案，哪怕边本身带 evidence_ids。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        doc_id,
        concepts=(outer, bpfo),
        segments=(),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=("ev:ghost:1:0-3",),
            ),
        ),
    )

    answer = ask(
        "轴承外圈疲劳剥落有什么征兆", deps=AskDeps(graph_store=graph)
    )

    assert answer.path is AnswerPath.REFUSAL
    assert answer.facts == ()
    assert answer.detail is not None and "证据" in answer.detail


def test_ask_refuses_when_matched_concepts_are_outside_the_relation_scope() -> None:
    """本版只走故障模式的一跳关系；设备类问题留给两跳检索（#15）。"""
    answer = ask("离心泵包含哪些部件", deps=empty_deps())

    assert answer.path is AnswerPath.REFUSAL
    assert answer.focus is None
    assert answer.detail is not None and "关系范围" in answer.detail


def test_ask_keeps_the_doc_id_when_no_document_catalog_is_injected() -> None:
    answer = ask("轴承外圈疲劳剥落有什么征兆", deps=deps_with_one_hop_facts())
    assert answer.citations

    without_names = ask(
        "轴承外圈疲劳剥落有什么征兆",
        deps=AskDeps(graph_store=deps_with_one_hop_facts().graph_store),
    )

    assert [citation.document for citation in without_names.citations] == [
        None,
        None,
    ]


def test_ask_does_not_report_aliases_nested_inside_a_longer_match() -> None:
    """问句里「轴承」「轴」「疲劳」「剥落」都只是「轴承外圈疲劳剥落」的碎片。"""
    answer = ask("轴承外圈疲劳剥落有什么征兆", deps=empty_deps())

    assert [match.concept.id for match in answer.matches] == [OUTER_RACE]

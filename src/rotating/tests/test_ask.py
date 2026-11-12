"""ask 门面：别名链接、一跳事实、引用与检索路径。

单元测试只依赖内存假存储与本体 YAML，不连 Neo4j、Postgres 或网络。
"""

from __future__ import annotations

from collections.abc import Sequence

from idr_core.graph import (
    ConceptNode,
    ConceptRelation,
    EvidenceAnchor,
    GraphBatch,
    Provenance,
)
from fakes import FakeGraphStore
from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_rotating import AnswerPath, AskDeps, LinkKind, RefusalReason, ask

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
    anchors: tuple[EvidenceAnchor, ...] = (),
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
            anchors=anchors,
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
    assert answer.refusal_reason is RefusalReason.NO_EVIDENCE
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
    assert answer.refusal_reason is RefusalReason.NO_EVIDENCE
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
    assert answer.refusal_reason is RefusalReason.NO_CONCEPT
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
    assert answer.refusal_reason is RefusalReason.NO_EVIDENCE
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
    """症状只是关系的值域，不能作为检索起点；设备类问题已由两跳检索接住。"""
    answer = ask("温升异常", deps=empty_deps())

    assert answer.path is AnswerPath.REFUSAL
    assert answer.refusal_reason is RefusalReason.NO_EVIDENCE
    assert answer.focus is None
    assert answer.facts == ()
    assert answer.citations == ()
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


INNER_RACE = "concept:bearing_inner_race_fatigue_spalling"
ROLLING_ELEMENT = "concept:bearing_rolling_element_fatigue_spalling"
BPFI = "concept:bpfi_harmonics"
SPECTRUM = "concept:fft_spectrum_analysis"


def deps_with_two_fault_modes(*, confused: bool = False) -> AskDeps:
    """一篇文献：内外圈剥落各有自己的征兆，但共用同一种检测手段。"""
    graph = FakeGraphStore()
    doc_id = "doc-a"
    outer_symptom = evidence(
        doc_id, page=2, offset=0, text="外圈剥落会出现 BPFO 及其倍频。"
    )
    inner_symptom = evidence(
        doc_id, page=2, offset=100, text="内圈剥落会出现 BPFI 及其倍频。"
    )
    outer_detection = evidence(
        doc_id, page=3, offset=0, text="外圈剥落常用频谱分析确认。"
    )
    inner_detection = evidence(
        doc_id, page=3, offset=100, text="内圈剥落也常用频谱分析确认。"
    )
    confused_statement = evidence(
        doc_id, page=4, offset=0, text="外圈剥落与内圈剥落容易混淆，需看特征频率。"
    )
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    inner = concept(INNER_RACE.removeprefix("concept:"), "轴承内圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    bpfi = concept(BPFI.removeprefix("concept:"), "内圈故障频率及其倍频", "symptom")
    spectrum = concept(SPECTRUM.removeprefix("concept:"), "频谱分析", "detection_method")
    write_document(
        graph,
        doc_id,
        concepts=(outer, inner, bpfo, bpfi, spectrum),
        segments=(
            outer_symptom,
            inner_symptom,
            outer_detection,
            inner_detection,
            confused_statement,
        ),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=doc_id,
                evidence_ids=(outer_symptom.id,),
                confidence=0.8,
            ),
            relation(
                inner,
                "indicated_by",
                bpfi,
                doc_id=doc_id,
                evidence_ids=(inner_symptom.id,),
                confidence=0.8,
            ),
            relation(
                outer,
                "detected_by",
                spectrum,
                doc_id=doc_id,
                evidence_ids=(outer_detection.id,),
                confidence=0.7,
            ),
            relation(
                inner,
                "detected_by",
                spectrum,
                doc_id=doc_id,
                evidence_ids=(inner_detection.id,),
                confidence=0.7,
            ),
            *(
                (
                    relation(
                        outer,
                        "confused_with",
                        inner,
                        doc_id=doc_id,
                        evidence_ids=(confused_statement.id,),
                        confidence=0.9,
                    ),
                )
                if confused
                else ()
            ),
        ),
    )
    return AskDeps(
        graph_store=graph,
        document_names=FakeDocumentNames({doc_id: "论文A.pdf"}),
    )


def test_ask_compares_two_equally_specific_fault_modes() -> None:
    """两个同级锚点不再判歧义拒答，而是按 (关系, 尾概念) 分成共同项与各自独有项。"""
    answer = ask("轴承外圈剥落和轴承内圈剥落怎么区分？", deps=deps_with_two_fault_modes())

    assert answer.path is AnswerPath.GRAPH
    assert answer.focus is None
    assert answer.facts == ()
    comparison = answer.comparison
    assert comparison is not None
    assert comparison.left.concept.id == OUTER_RACE
    assert comparison.right.concept.id == INNER_RACE

    # 两侧都检测「频谱分析」：这是共同点，两侧证据都要保留。
    assert [(shared.relation, shared.tail.id) for shared in comparison.shared] == [
        ("detected_by", SPECTRUM)
    ]
    shared = comparison.shared[0]
    assert len(shared.left) == 1 and len(shared.right) == 1
    assert [fact.tail.name for fact in shared.left] == ["频谱分析"]
    assert [fact.tail.name for fact in shared.right] == ["频谱分析"]

    # 各自独有的征兆：外圈 BPFO、内圈 BPFI。
    assert [(fact.relation, fact.tail.id) for fact in comparison.only_left] == [
        ("indicated_by", BPFO)
    ]
    assert [(fact.relation, fact.tail.id) for fact in comparison.only_right] == [
        ("indicated_by", BPFI)
    ]
    assert comparison.discriminating == ()

    # 引用覆盖两侧全部结论，按「共同项、仅左、仅右」的顺序展开。
    assert [citation.evidence_id for citation in answer.citations] == [
        shared.left[0].evidence_ids[0],
        shared.right[0].evidence_ids[0],
        comparison.only_left[0].evidence_ids[0],
        comparison.only_right[0].evidence_ids[0],
    ]


def test_ask_reports_confused_with_as_a_separate_discriminating_statement() -> None:
    """两侧之间的 confused_with 是文献自己说的「易混淆」，不混进「仅左侧」。"""
    answer = ask(
        "轴承外圈剥落和轴承内圈剥落怎么区分？",
        deps=deps_with_two_fault_modes(confused=True),
    )

    comparison = answer.comparison
    assert comparison is not None
    assert [(fact.relation, fact.tail.id) for fact in comparison.discriminating] == [
        ("confused_with", INNER_RACE)
    ]
    assert all(
        fact.relation != "confused_with" for fact in comparison.only_left
    )
    discriminating = comparison.discriminating[0]
    assert any(
        citation.evidence_id == discriminating.evidence_ids[0]
        and "容易混淆" in citation.text
        for citation in answer.citations
    )


def test_ask_links_the_rolling_element_spalling_alias() -> None:
    """「滚动体剥落」是文献里的常用说法，别名表缺了它就只能答一侧。"""
    answer = ask("轴承外圈剥落和滚动体剥落怎么区分？", deps=empty_deps())

    matched = {match.concept.id for match in answer.matches}
    assert ROLLING_ELEMENT in matched
    assert OUTER_RACE in matched


def test_ask_keeps_single_focus_when_specificity_is_far_apart() -> None:
    """泛称与专称相差太多时不强行配对：仍是单焦点，不是对比。"""
    answer = ask("轴承和鼠笼转子绕组有什么不同", deps=empty_deps())

    assert answer.comparison is None
    assert answer.focus is not None
    assert answer.focus.concept.id == "concept:squirrel_cage"


def test_ask_refuses_more_than_two_anchors_as_out_of_scope() -> None:
    """一次只比较两个：三个同级锚点不静默丢弃，也不假装答全。"""
    answer = ask(
        "轴承外圈剥落、内圈剥落和滚动体疲劳剥落怎么区分？", deps=empty_deps()
    )

    assert answer.path is AnswerPath.REFUSAL
    assert answer.comparison is None
    assert answer.detail is not None and "两两" in answer.detail


# 引用锚点：答案引用指向证据段内的支撑片段，而不是整段；缺锚点时退回整段。
FRAGMENT_DOC = "doc-a"
FRAGMENT_TEXT = "外圈剥落会在频谱上出现 BPFO 及其倍频，能量集中在高频段。"
FRAGMENT_QUOTE = "外圈剥落会在频谱上出现 BPFO 及其倍频"


def deps_with_anchored_one_hop_fact() -> AskDeps:
    """一篇文献：征兆边带锚点，片段只是证据段里的一句话。"""
    graph = FakeGraphStore()
    segment = evidence(FRAGMENT_DOC, page=4, offset=120, text=FRAGMENT_TEXT)
    start = FRAGMENT_TEXT.index(FRAGMENT_QUOTE)
    anchor = EvidenceAnchor(
        evidence_id=segment.id,
        start=start,
        end=start + len(FRAGMENT_QUOTE),
    )
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        FRAGMENT_DOC,
        concepts=(outer, bpfo),
        segments=(segment,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.8,
                anchors=(anchor,),
            ),
        ),
    )
    return AskDeps(
        graph_store=graph,
        document_names=FakeDocumentNames({FRAGMENT_DOC: "论文A.pdf"}),
    )


def test_ask_cites_the_anchor_span_instead_of_the_whole_evidence_segment() -> None:
    """有锚点时引用展示锚点区间对应的原文，页码与证据段标识不变。"""
    answer = ask("轴承外圈疲劳剥落有什么征兆", deps=deps_with_anchored_one_hop_fact())

    assert answer.path is AnswerPath.GRAPH
    (citation,) = answer.citations
    assert citation.text == FRAGMENT_QUOTE
    assert citation.anchor is not None
    assert citation.anchor.evidence_id == citation.evidence_id
    assert citation.anchor.end - citation.anchor.start == len(FRAGMENT_QUOTE)
    assert citation.evidence_id == f"ev:{FRAGMENT_DOC}:4:120-{120 + len(FRAGMENT_TEXT)}"
    assert (citation.doc_id, citation.document, citation.page) == (
        FRAGMENT_DOC,
        "论文A.pdf",
        4,
    )
    # 结论仍指向整段证据：锚点只改变引用展示，不改变检索命中的判定单位。
    assert answer.facts[0].evidence_ids == (citation.evidence_id,)


def test_ask_falls_back_to_the_whole_segment_when_no_anchor_is_recorded() -> None:
    """证据段只有一句话时无锚点仍整段引用：选句只取整句，不会切出半句。"""
    graph = FakeGraphStore()
    segment = evidence(FRAGMENT_DOC, page=4, offset=0, text=FRAGMENT_TEXT)
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        FRAGMENT_DOC,
        concepts=(outer, bpfo),
        segments=(segment,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.8,
            ),
        ),
    )

    answer = ask(
        "轴承外圈疲劳剥落有什么征兆",
        deps=AskDeps(
            graph_store=graph,
            document_names=FakeDocumentNames({FRAGMENT_DOC: "论文A.pdf"}),
        ),
    )

    (citation,) = answer.citations
    assert citation.anchor is None
    assert citation.text == FRAGMENT_TEXT
    assert citation.page == 4


def test_ask_gives_each_assertion_its_own_fragment_from_a_shared_segment() -> None:
    """同一证据段支撑两条断言时，两条引用各自指向各自的片段。"""
    text = "外圈剥落会产生 BPFO 特征频率。内圈剥落会产生 BPFI 特征频率。"
    outer_quote = "外圈剥落会产生 BPFO 特征频率。"
    inner_quote = "内圈剥落会产生 BPFI 特征频率。"
    graph = FakeGraphStore()
    segment = evidence(FRAGMENT_DOC, page=2, offset=0, text=text)

    def anchor_for(quote: str) -> EvidenceAnchor:
        start = text.index(quote)
        return EvidenceAnchor(
            evidence_id=segment.id, start=start, end=start + len(quote)
        )

    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    inner = concept(INNER_RACE.removeprefix("concept:"), "轴承内圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    bpfi = concept(BPFI.removeprefix("concept:"), "内圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        FRAGMENT_DOC,
        concepts=(outer, inner, bpfo, bpfi),
        segments=(segment,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.8,
                anchors=(anchor_for(outer_quote),),
            ),
            relation(
                inner,
                "indicated_by",
                bpfi,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.8,
                anchors=(anchor_for(inner_quote),),
            ),
        ),
    )

    answer = ask(
        "轴承外圈剥落和轴承内圈剥落怎么区分？",
        deps=AskDeps(
            graph_store=graph,
            document_names=FakeDocumentNames({FRAGMENT_DOC: "论文A.pdf"}),
        ),
    )

    assert answer.path is AnswerPath.GRAPH
    assert [citation.text for citation in answer.citations] == [outer_quote, inner_quote]
    assert {citation.evidence_id for citation in answer.citations} == {segment.id}
    assert {citation.page for citation in answer.citations} == {2}
    outer_anchor, inner_anchor = (citation.anchor for citation in answer.citations)
    assert isinstance(outer_anchor, EvidenceAnchor)
    assert isinstance(inner_anchor, EvidenceAnchor)
    assert outer_anchor != inner_anchor
    assert text[outer_anchor.start : outer_anchor.end] == outer_quote
    assert text[inner_anchor.start : inner_anchor.end] == inner_quote


# 无锚点选句兜底：旧数据没有锚点时，按断言的头概念名、关系标签与尾概念名
# 在证据段里挑出最相关的一句，而不是把整段都当成引用。
SELECTION_TEXT = (
    "外圈剥落会在频谱上出现 BPFO 及其倍频。"
    "内圈剥落会产生 BPFI 特征频率，与转速同步。"
)
SELECTION_QUOTE = "外圈剥落会在频谱上出现 BPFO 及其倍频。"


def deps_with_unanchored_segment(text: str) -> tuple[AskDeps, EvidenceSegment]:
    """一篇文献：一条无锚点断言，证据段文本由调用方给定。"""
    graph = FakeGraphStore()
    segment = evidence(FRAGMENT_DOC, page=4, offset=0, text=text)
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    write_document(
        graph,
        FRAGMENT_DOC,
        concepts=(outer, bpfo),
        segments=(segment,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.8,
            ),
        ),
    )
    return (
        AskDeps(
            graph_store=graph,
            document_names=FakeDocumentNames({FRAGMENT_DOC: "论文A.pdf"}),
        ),
        segment,
    )


def test_ask_selects_the_sentence_matching_the_assertion_when_no_anchor() -> None:
    """没有锚点时引用指向与断言词法重合最多的那一句，且是逐字切片。"""
    deps, segment = deps_with_unanchored_segment(SELECTION_TEXT)

    answer = ask("轴承外圈疲劳剥落有什么征兆", deps=deps)

    (citation,) = answer.citations
    assert citation.text == SELECTION_QUOTE
    assert citation.anchor is not None
    assert citation.text == segment.text[citation.anchor.start : citation.anchor.end]
    assert citation.evidence_id == segment.id
    assert citation.page == 4
    # 检索命中的判定单位仍是整条证据段。
    assert answer.facts[0].evidence_ids == (segment.id,)
    # 同一断言重复运行必须得到同一条引用。
    assert ask("轴承外圈疲劳剥落有什么征兆", deps=deps).citations == answer.citations


def test_ask_falls_back_to_the_whole_segment_when_no_sentence_matches() -> None:
    """没有任何句子命中断言关键词时整段引用，不凭空选一句。"""
    text = "该文献给出巡检周期建议。振动幅值需要按周记录。"
    deps, segment = deps_with_unanchored_segment(text)

    answer = ask("轴承外圈疲劳剥落有什么征兆", deps=deps)

    (citation,) = answer.citations
    assert citation.anchor is None
    assert citation.text == segment.text


def test_ask_gives_each_assertion_its_own_selected_window_on_a_shared_segment() -> None:
    """同一段证据支撑两条断言时，选出的两个句子窗口各自成一条引用。"""
    text = (
        "外圈剥落会在频谱上出现 BPFO 及其倍频。"
        "包络解调可以确认外圈剥落的冲击。"
    )
    graph = FakeGraphStore()
    segment = evidence(FRAGMENT_DOC, page=5, offset=0, text=text)
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    envelope = concept(ENVELOPE.removeprefix("concept:"), "包络解调", "detection_method")
    write_document(
        graph,
        FRAGMENT_DOC,
        concepts=(outer, bpfo, envelope),
        segments=(segment,),
        relations=(
            relation(
                outer,
                "indicated_by",
                bpfo,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.8,
            ),
            relation(
                outer,
                "detected_by",
                envelope,
                doc_id=FRAGMENT_DOC,
                evidence_ids=(segment.id,),
                confidence=0.7,
            ),
        ),
    )

    answer = ask(
        "轴承外圈疲劳剥落有什么征兆",
        deps=AskDeps(
            graph_store=graph,
            document_names=FakeDocumentNames({FRAGMENT_DOC: "论文A.pdf"}),
        ),
    )

    assert [citation.text for citation in answer.citations] == [
        "外圈剥落会在频谱上出现 BPFO 及其倍频。",
        "包络解调可以确认外圈剥落的冲击。",
    ]
    assert {citation.evidence_id for citation in answer.citations} == {segment.id}
    anchors = [citation.anchor for citation in answer.citations]
    assert all(anchor is not None for anchor in anchors)
    assert anchors[0] != anchors[1]

"""图检索读模型：跨文献并行边聚合，以及无证据边的排除。"""

from __future__ import annotations

from idr_core.anchors import EvidenceAnchor
from idr_core.graph.read import FactEdge, aggregate_facts


def make_edge(
    tail_id: str,
    evidence_ids: tuple[str, ...],
    *,
    doc_id: str = "doc-a",
    confidence: float = 0.5,
    relation: str = "linked_to",
    head_id: str = "head",
    anchors: tuple[EvidenceAnchor, ...] = (),
) -> FactEdge:
    return FactEdge(
        head_id=head_id,
        relation=relation,
        tail_id=tail_id,
        evidence_ids=evidence_ids,
        confidence=confidence,
        doc_id=doc_id,
        anchors=anchors,
    )


def test_parallel_edges_are_merged_by_source_relation_target() -> None:
    """同一概念对每篇文献一条并行边：聚合成一条事实，证据取并集。"""
    edges = (
        make_edge("tail-a", ("e1",), doc_id="doc-1", confidence=0.6),
        make_edge("tail-b", ("e2",), doc_id="doc-1", confidence=0.9),
        make_edge("tail-a", ("e3", "e4"), doc_id="doc-2", confidence=0.8),
    )

    facts = aggregate_facts(edges)

    assert [(fact.tail_id, fact.evidence_ids) for fact in facts] == [
        ("tail-a", ("e1", "e3", "e4")),
        ("tail-b", ("e2",)),
    ]
    # 聚合后的置信度取各并行边的最大值：一条事实的可信度由最强支持决定。
    assert facts[0].confidence == 0.8


def test_edges_without_evidence_are_dropped() -> None:
    """没有 evidence_ids 的事实边不进答案（ADR-0004）。"""
    edges = (
        make_edge("tail-a", ()),
        make_edge("tail-b", ("e1",)),
    )

    assert [fact.tail_id for fact in aggregate_facts(edges)] == ["tail-b"]


def test_same_evidence_on_parallel_edges_is_listed_once() -> None:
    """不同文献引用了同一证据段时，证据并集里只出现一次。"""
    edges = (
        make_edge("tail-a", ("e1",), doc_id="doc-1"),
        make_edge("tail-a", ("e1",), doc_id="doc-2"),
    )

    assert aggregate_facts(edges)[0].evidence_ids == ("e1",)


def test_facts_keep_the_order_edges_arrive_in() -> None:
    """聚合不改变遍历顺序：关系类型的先后由检索编排决定，不在这里重排。"""
    edges = (
        make_edge("tail-b", ("e1",), relation="indicated_by"),
        make_edge("tail-a", ("e2",), relation="detected_by"),
    )

    assert [(fact.relation, fact.tail_id) for fact in aggregate_facts(edges)] == [
        ("indicated_by", "tail-b"),
        ("detected_by", "tail-a"),
    ]


def test_anchors_are_merged_with_the_evidence_they_belong_to() -> None:
    """并行边聚合后锚点仍在：按证据段对齐，没有锚点的证据段保持没有。"""
    first = EvidenceAnchor(evidence_id="e1", start=2, end=5)
    third = EvidenceAnchor(evidence_id="e3", start=0, end=6)
    edges = (
        make_edge("tail-a", ("e1", "e2"), doc_id="doc-1", anchors=(first,)),
        make_edge("tail-a", ("e3",), doc_id="doc-2", anchors=(third,)),
    )

    (fact,) = aggregate_facts(edges)

    assert fact.evidence_ids == ("e1", "e2", "e3")
    # 锚点顺序跟证据并集一致；e2 没有锚点，不在其中（引用层退回整段）。
    assert fact.anchors == (first, third)


def test_edges_without_anchors_still_aggregate() -> None:
    """锚点上线前的旧边没有锚点：整条事实照常聚合，只是引用整段。"""
    edges = (make_edge("tail-a", ("e1",)),)

    (fact,) = aggregate_facts(edges)

    assert fact.evidence_ids == ("e1",)
    assert fact.anchors == ()

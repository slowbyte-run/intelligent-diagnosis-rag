"""图谱快照导出：闭包规则、证据正文完整性，以及同一份图的确定性。

导出只走 ``GraphReader`` 读端口，所以这里用内存假图验证闭包边界：种子的
白名单邻接边、两跳内的结构关系中转、被引证据段正文，以及金标准直接绑定的
证据段；闭包之外的节点、边与证据一条都不进快照。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from idr_core.anchors import EvidenceAnchor
from idr_core.graph.read import Concept, EvidenceRecord, FactEdge, GraphReader
from idr_core.graph.snapshot import serialize_snapshot
from idr_core.models import CharSpan
from idr_rotating.evaluation import GoldenQuestion
from idr_rotating.ontology import (
    EntityTypeDef,
    InstanceDef,
    Ontology,
    RelationTypeDef,
)
from idr_rotating.snapshot_export import SnapshotExportError, export_snapshot

DOC = "0123456789abcdef"
OTHER_DOC = "fedcba9876543210"


def concept(name: str, *, type_: str = "kind") -> Concept:
    return Concept(id=f"concept:{name}", name=f"节点{name}", type=type_)


def evidence(name: str) -> EvidenceRecord:
    """每条证据段有唯一字符区间，标识因此不会撞。"""
    index = int(name[1:])
    text = f"证据段{name}的原文"
    start = index * 100
    return EvidenceRecord(
        id=f"ev:{DOC}:1:{start}-{start + len(text)}",
        doc_id=DOC,
        page=1,
        text=text,
        charspan=CharSpan(start, start + len(text)),
    )


def question(
    *,
    question_id: str = "T-01",
    text: str = "任意的问题",
    concepts: tuple[str, ...] = ("concept:alpha",),
    evidence_ids: tuple[str, ...] = (),
) -> GoldenQuestion:
    return GoldenQuestion(
        id=question_id,
        question_class="factual",
        question=text,
        expected_concepts=concepts,
        expected_evidence_ids=evidence_ids,
    )


def minimal_ontology() -> Ontology:
    """只用于链接测试的最小本体：别名「甲概念」指向 alpha。"""
    return Ontology(
        version="test",
        domain="test",
        entity_types={"kind": EntityTypeDef(label="类别")},
        relation_types={
            "linked_to": RelationTypeDef(label="关联", domain=["kind"], range=["kind"])
        },
        instances={"kind": [InstanceDef(id="alpha", label="甲概念", aliases=["甲概念"])]},
    )


@dataclass(frozen=True)
class MemoryGraph:
    """内存图读取端口：按标识返回记录，``shuffled`` 用来破坏返回顺序。"""

    concepts: tuple[Concept, ...]
    edges: tuple[FactEdge, ...]
    evidence: tuple[EvidenceRecord, ...]
    shuffled: bool = False

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        selected = tuple(
            edge
            for edge in self.edges
            if edge.head_id == concept_id and edge.relation in relation_types
        )
        return tuple(reversed(selected)) if self.shuffled else selected

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        wanted = set(concept_ids)
        selected = tuple(item for item in self.concepts if item.id in wanted)
        return tuple(reversed(selected)) if self.shuffled else selected

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        wanted = set(evidence_ids)
        selected = tuple(item for item in self.evidence if item.id in wanted)
        return tuple(reversed(selected)) if self.shuffled else selected

    def evidence_by_page(self, doc_id: str, page: int) -> tuple[EvidenceRecord, ...]:
        return tuple(
            item for item in self.evidence if (item.doc_id, item.page) == (doc_id, page)
        )


def edge(
    head: str,
    relation: str,
    tail: str,
    evidence_ids: tuple[str, ...],
    *,
    doc_id: str = DOC,
    confidence: float = 0.5,
    anchors: tuple[EvidenceAnchor, ...] = (),
) -> FactEdge:
    return FactEdge(
        head_id=f"concept:{head}",
        relation=relation,
        tail_id=f"concept:{tail}",
        evidence_ids=evidence_ids,
        confidence=confidence,
        doc_id=doc_id,
        anchors=anchors,
    )


def graph_with_closure() -> tuple[MemoryGraph, dict[str, EvidenceRecord]]:
    records = {f"e{index}": evidence(f"e{index}") for index in range(1, 9)}
    concepts = tuple(
        concept(name)
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "far", "zeta")
    )
    edges = (
        # 种子的一跳邻接：两类关系都导出，锚点随之保留。
        edge(
            "alpha",
            "has_component",
            "beta",
            (records["e1"].id,),
            confidence=0.9,
            anchors=(
                EvidenceAnchor(evidence_id=records["e1"].id, start=0, end=3),
            ),
        ),
        edge("alpha", "indicated_by", "gamma", (records["e2"].id,), doc_id=OTHER_DOC),
        # 两跳：beta 是结构关系中转点，它的边要导出。
        edge("beta", "has_fault_mode", "delta", (records["e3"].id,)),
        # 三跳：delta 在第二跳终点，不再继续展开。
        edge("delta", "detected_by", "epsilon", (records["e4"].id,)),
        # 闭包之外的节点：它的边不进快照。
        edge("far", "has_component", "beta", (records["e5"].id,)),
        # 同一对概念的另一类关系：并行保留。
        edge("alpha", "has_cause", "beta", (records["e6"].id,), doc_id=OTHER_DOC),
        # 非白名单关系：检索走不到，不导出。
        edge("alpha", "mentions", "far", (records["e7"].id,)),
        # 没有证据的边：聚合层本来就会丢弃。
        edge("beta", "has_component", "zeta", ()),
    )
    return (
        MemoryGraph(concepts=concepts, edges=edges, evidence=tuple(records.values())),
        records,
    )


def test_closure_is_the_seed_adjacency_plus_two_hop_structural_expansion() -> None:
    graph, records = graph_with_closure()

    snapshot = export_snapshot(
        graph,
        (question(evidence_ids=(records["e8"].id,)),),
        ontology=minimal_ontology(),
    )
    document = json.loads(serialize_snapshot(snapshot))

    assert [item["id"] for item in document["concepts"]] == [
        "concept:alpha",
        "concept:beta",
        "concept:delta",
        "concept:gamma",
    ]
    assert [
        (item["head_id"], item["relation"], item["tail_id"], item["doc_id"])
        for item in document["edges"]
    ] == [
        ("concept:alpha", "has_cause", "concept:beta", OTHER_DOC),
        ("concept:alpha", "has_component", "concept:beta", DOC),
        ("concept:alpha", "indicated_by", "concept:gamma", OTHER_DOC),
        ("concept:beta", "has_fault_mode", "concept:delta", DOC),
    ]


def test_snapshot_carries_text_for_every_cited_segment_and_every_golden_binding() -> None:
    graph, records = graph_with_closure()

    snapshot = export_snapshot(
        graph,
        (question(evidence_ids=(records["e8"].id,)),),
        ontology=minimal_ontology(),
    )
    document = json.loads(serialize_snapshot(snapshot))

    exported = {item["id"]: item["text"] for item in document["evidence"]}
    cited = {
        evidence_id
        for item in document["edges"]
        for evidence_id in item["evidence_ids"]
    }
    assert set(exported) == cited | {records["e8"].id}
    for evidence_id in cited:
        assert exported[evidence_id] == next(
            item.text for item in graph.evidence if item.id == evidence_id
        )
    # 金标准绑定的证据段即使不在任何关系边上，也带正文进快照，加载期才能
    # 核对「题目引用的东西确实存在」，而不是等到评分时才莫名少一条。
    assert exported[records["e8"].id] == records["e8"].text
    assert records["e4"].id not in exported
    assert records["e5"].id not in exported
    assert records["e7"].id not in exported


def test_anchors_are_exported_with_their_offsets() -> None:
    graph, records = graph_with_closure()

    snapshot = export_snapshot(graph, (question(),), ontology=minimal_ontology())
    document = json.loads(serialize_snapshot(snapshot))
    anchored = next(
        item
        for item in document["edges"]
        if item["relation"] == "has_component" and item["head_id"] == "concept:alpha"
    )

    assert anchored["anchors"] == [
        {
            "evidence_id": records["e1"].id,
            "start": 0,
            "end": 3,
            "ambiguous": False,
        }
    ]


def test_parallel_edges_from_different_documents_stay_separate() -> None:
    """同一对概念在每篇文献一条并行边；快照保留原貌，聚合留给读侧。"""
    records = {name: evidence(name) for name in ("e1", "e6")}
    graph = MemoryGraph(
        concepts=(concept("alpha"), concept("beta")),
        edges=(
            edge("alpha", "has_cause", "beta", (records["e1"].id,)),
            edge(
                "alpha", "has_cause", "beta", (records["e6"].id,), doc_id=OTHER_DOC
            ),
        ),
        evidence=tuple(records.values()),
    )

    snapshot = export_snapshot(graph, (question(),), ontology=minimal_ontology())

    assert len(snapshot.edges) == 2
    assert {item.doc_id for item in snapshot.edges} == {DOC, OTHER_DOC}


def test_question_linked_concepts_are_seeds_even_when_not_in_the_golden_bindings() -> None:
    """评分调用的是 ask(question)：问句链接到的概念也必须是闭包起点。

    否则同一句话在快照上会因为「概念不在图谱里」而拒答，与线上行为不一致。
    """
    graph, _ = graph_with_closure()

    snapshot = export_snapshot(
        graph,
        (question(text="甲概念怎么用", concepts=("concept:far",)),),
        ontology=minimal_ontology(),
    )

    assert {item.id for item in snapshot.concepts} >= {"concept:alpha", "concept:far"}
    assert any(edge.head_id == "concept:far" for edge in snapshot.edges)


def test_export_is_byte_identical_whatever_order_the_reader_returns() -> None:
    graph, _ = graph_with_closure()
    ordered = MemoryGraph(
        concepts=graph.concepts, edges=graph.edges, evidence=graph.evidence
    )
    shuffled = MemoryGraph(
        concepts=graph.concepts,
        edges=graph.edges,
        evidence=graph.evidence,
        shuffled=True,
    )

    first = export_snapshot(ordered, (question(),), ontology=minimal_ontology())
    second = export_snapshot(shuffled, (question(),), ontology=minimal_ontology())

    assert serialize_snapshot(first) == serialize_snapshot(second)
    assert first.checksum == second.checksum


def test_a_golden_binding_missing_from_the_graph_fails_loudly() -> None:
    graph, _ = graph_with_closure()

    with pytest.raises(SnapshotExportError, match="concept:ghost"):
        export_snapshot(
            graph,
            (question(concepts=("concept:ghost",)),),
            ontology=minimal_ontology(),
        )


def test_a_cited_evidence_segment_missing_from_the_graph_fails_loudly() -> None:
    graph, records = graph_with_closure()
    broken = MemoryGraph(
        concepts=graph.concepts,
        edges=graph.edges,
        evidence=tuple(item for item in graph.evidence if item.id != records["e1"].id),
    )

    with pytest.raises(SnapshotExportError, match="证据段"):
        export_snapshot(broken, (question(),), ontology=minimal_ontology())


def test_an_edge_tail_missing_from_the_graph_fails_loudly() -> None:
    graph, _ = graph_with_closure()
    broken = MemoryGraph(
        concepts=tuple(item for item in graph.concepts if item.id != "concept:beta"),
        edges=graph.edges,
        evidence=graph.evidence,
    )

    with pytest.raises(SnapshotExportError, match="concept:beta"):
        export_snapshot(broken, (question(),), ontology=minimal_ontology())


def test_export_declares_the_closure_scope_in_the_file() -> None:
    graph, _ = graph_with_closure()

    snapshot = export_snapshot(graph, (question(),), ontology=minimal_ontology())

    assert snapshot.closure.seed_concept_ids == ("concept:alpha",)
    assert set(snapshot.closure.relation_types) == {
        "has_component",
        "has_fault_mode",
        "has_cause",
        "indicated_by",
        "detected_by",
        "confused_with",
    }
    assert snapshot.closure.structural_relations == (
        "has_component",
        "has_fault_mode",
    )
    assert snapshot.closure.max_hops == 2


def test_graph_reader_protocol_is_all_the_export_needs() -> None:
    """导出只依赖读端口，不依赖 Neo4j 适配器。"""
    graph, _ = graph_with_closure()
    reader: GraphReader = graph

    snapshot = export_snapshot(reader, (question(),), ontology=minimal_ontology())

    assert snapshot.edges

"""从线上图谱导出检索评估用的快照：只读、只导闭包、结果确定。

闭包边界与 ``traverse`` 的读取范围一致，否则快照上的评估会与线上行为分叉：

1. **起点**：金标准每题绑定的 ``expected_concepts``，加上问句经本体别名表
   链接到的概念。评分调用的是 ``ask(question)``，问句链接到的概念会成为检索
   起点，所以它们也必须在快照里；只按题面绑定的概念导出，会让「问句命中、
   但期望概念里没写」的题在快照上因概念缺失而改变行为。
2. **一跳**：从每个起点取白名单关系（``EDGE_WHITELIST``）的出边，这是检索
   实际能读到的候选结论。
3. **两跳**：一跳里沿结构关系（``STRUCTURAL_RELATIONS``）到达的节点作为
   中转，再取它们的白名单出边。三跳及以上不取，与 ``MAX_HOPS`` 一致。
4. **边**：同一对概念、同一关系、同一文献的一条边原样保留（并行边不聚合，
   聚合是读侧的事）；没有证据的边不导出——聚合层本来就会丢弃它。
5. **证据段**：已导出边引用的全部证据段，加上金标准直接绑定的证据段。后者
   即使不在任何关系边上也带正文进快照：加载期才能核对「题目引用的东西确实
   存在」，而不是等到评分时才莫名少一条。节点→证据的内部边不导出，检索
   端口不读它。

导出失败是数据错误，不静默跳过：金标准绑定的概念或证据段在图谱里不存在，
或边的任一端、任一段证据在端口里取不到，都直接报错，不生成半份快照。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from idr_core.graph.read import Concept, EvidenceRecord, FactEdge, GraphReader
from idr_core.graph.snapshot import (
    GraphSnapshot,
    SnapshotClosure,
    SnapshotError,
    write_snapshot,
)
from idr_rotating.evaluation import GoldenQuestion
from idr_rotating.linking import link_question
from idr_rotating.ontology import Ontology, load_ontology
from idr_rotating.traversal import (
    EDGE_WHITELIST,
    MAX_HOPS,
    STRUCTURAL_RELATIONS,
)

# 快照与金标准集同级：两者一起决定「这次评分用的是哪版图谱、哪些题」。
DEFAULT_SNAPSHOT_PATH = Path(__file__).with_name("graph_snapshot.json")


class SnapshotExportError(SnapshotError):
    """闭包不完整或数据对不上：导出必须失败，不能交出一份看起来正常的快照。"""


def export_snapshot(
    graph: GraphReader,
    questions: Sequence[GoldenQuestion],
    *,
    ontology: Ontology | None = None,
) -> GraphSnapshot:
    """按闭包规则从图读取端口导出快照；不写文件、不写图谱。"""
    if not questions:
        raise SnapshotExportError("没有金标准题，无法确定要导出哪部分图谱")

    expected_concepts = {
        concept_id
        for question in questions
        for concept_id in question.expected_concepts
    }
    expected_evidence = {
        evidence_id
        for question in questions
        for evidence_id in question.expected_evidence_ids
    }
    linked_concepts = {
        match.concept.id
        for question in questions
        for match in link_question(
            question.question, load_ontology() if ontology is None else ontology
        )
    }

    seeds = _resolve_seeds(
        graph, expected_concepts=expected_concepts, linked_concepts=linked_concepts
    )
    if not seeds:
        raise SnapshotExportError(
            "金标准集没有绑定任何概念，问句也没有链接到任何概念，没有可导出的闭包"
        )

    edges = _collect_edges(graph, seeds)
    concepts = _collect_concepts(graph, seeds, edges)
    evidence = _collect_evidence(graph, edges, expected_evidence)

    return GraphSnapshot.build(
        closure=SnapshotClosure(
            seed_concept_ids=seeds,
            relation_types=EDGE_WHITELIST,
            structural_relations=STRUCTURAL_RELATIONS,
            max_hops=MAX_HOPS,
        ),
        concepts=concepts,
        edges=edges,
        evidence=evidence,
    )


def write_export(
    graph: GraphReader,
    questions: Sequence[GoldenQuestion],
    *,
    path: Path | None = None,
    ontology: Ontology | None = None,
) -> GraphSnapshot:
    """导出并写文件；默认写到领域包内与金标准同级的位置。"""
    snapshot = export_snapshot(graph, questions, ontology=ontology)
    write_snapshot(DEFAULT_SNAPSHOT_PATH if path is None else Path(path), snapshot)
    return snapshot


def _resolve_seeds(
    graph: GraphReader,
    *,
    expected_concepts: set[str],
    linked_concepts: set[str],
) -> tuple[str, ...]:
    """起点概念：金标准绑定的必须在图里；问句链接到的若图里没有就跳过。

    跳过链接不存在的概念与线上一致：``ask`` 查不到节点时本来就会拒答，
    快照不替它补一个不存在的节点。
    """
    requested = sorted(expected_concepts | linked_concepts)
    found = {concept.id for concept in graph.concepts_by_ids(requested)}
    missing = sorted(expected_concepts - found)
    if missing:
        raise SnapshotExportError(
            "金标准集绑定的概念不在图谱里：" + "、".join(missing)
        )
    return tuple(sorted(found))


def _collect_edges(
    graph: GraphReader, seeds: Sequence[str]
) -> tuple[FactEdge, ...]:
    """取起点的一跳邻接与两跳结构扩展，按并行边原貌保留。"""
    collected: dict[tuple[str, str, str, str], FactEdge] = {}
    frontier = tuple(seeds)
    fetched: set[str] = set()

    for hop in range(1, MAX_HOPS + 1):
        current = tuple(node for node in frontier if node not in fetched)
        fetched.update(current)
        reachable: list[str] = []
        for node in current:
            for edge in graph.facts_from(node, EDGE_WHITELIST):
                if edge.relation not in EDGE_WHITELIST:
                    continue
                if not edge.evidence_ids:
                    # 聚合层会丢弃无证据的边：导出一份永远不会被读到的边没有意义。
                    continue
                key = (edge.head_id, edge.relation, edge.tail_id, edge.doc_id)
                previous = collected.get(key)
                if previous is not None:
                    if previous != edge:
                        raise SnapshotExportError(
                            "同一条关系边在读取端口里返回了两种内容："
                            f"{edge.head_id} -{edge.relation}-> {edge.tail_id}"
                            f"（{edge.doc_id}）"
                        )
                    continue
                collected[key] = edge
                if hop < MAX_HOPS and edge.relation in STRUCTURAL_RELATIONS:
                    reachable.append(edge.tail_id)
        frontier = tuple(sorted(set(reachable)))

    return tuple(collected[key] for key in sorted(collected))


def _collect_concepts(
    graph: GraphReader, seeds: Sequence[str], edges: Sequence[FactEdge]
) -> tuple[Concept, ...]:
    """起点与已导出边两端的全部概念；缺任何一个都说明闭包不完整。"""
    needed = sorted(
        set(seeds) | {edge.head_id for edge in edges} | {edge.tail_id for edge in edges}
    )
    found = {concept.id: concept for concept in graph.concepts_by_ids(needed)}
    missing = [concept_id for concept_id in needed if concept_id not in found]
    if missing:
        raise SnapshotExportError(
            "快照闭包里缺少概念节点：" + "、".join(missing)
        )
    return tuple(found[concept_id] for concept_id in needed)


def _collect_evidence(
    graph: GraphReader,
    edges: Sequence[FactEdge],
    expected_evidence: set[str],
) -> tuple[EvidenceRecord, ...]:
    """被导出边引用的证据段，加上金标准直接绑定的证据段。"""
    requested = sorted(
        {evidence_id for edge in edges for evidence_id in edge.evidence_ids}
        | expected_evidence
    )
    found = {record.id: record for record in graph.evidence_by_ids(requested)}
    missing = [evidence_id for evidence_id in requested if evidence_id not in found]
    if missing:
        raise SnapshotExportError(
            "快照闭包里缺少证据段正文：" + "、".join(missing)
        )
    return tuple(found[evidence_id] for evidence_id in requested)

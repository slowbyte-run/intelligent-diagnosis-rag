"""把一篇文献的校验结果转成写图批次。

概念节点 ID 分两种作用域：解析到本体规范实例的概念用共享 ID
``concept:{实例 ID}``，不带 ``doc_id``，因此可以跨文献合并；自由词在
本文献内用 ``{doc_id}:concept:{key}``，带 ``doc_id``，重跑本文献时随文档
作用域节点一起被替换。所有边都带 ``doc_id``，按文献归属删除。
"""

from __future__ import annotations

from idr_core.graph import (
    ConceptEvidenceLink,
    ConceptNode,
    ConceptRelation,
    GraphBatch,
    Provenance,
)
from idr_core.models import EvidenceSegment
from idr_rotating.errors import IngestError
from idr_rotating.extraction import (
    ExtractedRelation,
    ValidatedEntity,
)


def build_graph_batch(
    doc_id: str,
    segments: tuple[EvidenceSegment, ...],
    entities: tuple[ValidatedEntity, ...],
    relations: tuple[ExtractedRelation, ...],
) -> GraphBatch:
    """把一篇文献的校验结果转成写图批次。"""
    concepts: list[ConceptNode] = []
    links: list[ConceptEvidenceLink] = []
    concept_by_name: dict[str, ConceptNode] = {}

    for entity in entities:
        concept = ConceptNode(
            id=(
                f"concept:{entity.key}"
                if entity.canonical
                else f"{doc_id}:concept:{entity.key}"
            ),
            name=entity.name,
            type=entity.type,
            doc_id=None if entity.canonical else doc_id,
            attributes=dict(entity.attributes),
        )
        concepts.append(concept)
        concept_by_name.setdefault(entity.name, concept)
        # 锚点按证据段挂到对应的概念—证据边上；没有锚点的证据段照旧整段引用。
        anchors_by_evidence = {
            anchor.evidence_id: anchor for anchor in entity.anchors
        }
        for evidence_id in entity.evidence_ids:
            anchor = anchors_by_evidence.get(evidence_id)
            links.append(
                ConceptEvidenceLink(
                    concept_id=concept.id,
                    provenance=Provenance(
                        doc_id=doc_id,
                        evidence_ids=(evidence_id,),
                        confidence=entity.confidence,
                        source="extracted",
                        anchors=() if anchor is None else (anchor,),
                    ),
                )
            )

    graph_relations: list[ConceptRelation] = []
    for relation in relations:
        head = concept_by_name.get(relation.head)
        tail = concept_by_name.get(relation.tail)
        if head is None or tail is None:
            # 端点缺失是数据错误（AGENTS.md 第 5 条），不得静默丢边；
            # 抛出让本篇记为失败，报告里能看见是哪条关系出的问题。
            raise IngestError(
                f"关系 {relation.type} 的端点未写入概念节点："
                f"{relation.head} → {relation.tail}"
            )
        graph_relations.append(
            ConceptRelation(
                head_id=head.id,
                type=relation.type,
                tail_id=tail.id,
                provenance=Provenance(
                    doc_id=doc_id,
                    evidence_ids=tuple(relation.evidence_ids),
                    confidence=relation.confidence,
                    source="extracted",
                    anchors=tuple(relation.anchors),
                ),
            )
        )

    return GraphBatch(
        doc_id=doc_id,
        concepts=tuple(concepts),
        evidence=segments,
        relations=tuple(graph_relations),
        evidence_links=tuple(links),
    )

"""Neo4j 图存储适配器：概念 + 证据双层，边属性携带证据与置信度。"""

from __future__ import annotations

import re
from typing import Any, LiteralString, cast

from neo4j import Driver, GraphDatabase, ManagedTransaction

from idr_core.graph.models import GraphBatch
from idr_core.models import EvidenceSegment

_RELATION_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Neo4jGraphStore:
    """把 ``GraphBatch`` 写进 Neo4j。

    节点 ID 由调用方稳定推导，写入一律用 MERGE；重复处理同一篇文献不会产生
    重复节点。关系类型来自本体校验后的清单，写入前再校验字符集，不能直接
    拼进 Cypher。
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def write_document(self, batch: GraphBatch) -> None:
        with self._driver.session() as session:
            session.execute_write(self._write_batch, batch)

    @staticmethod
    def _write_batch(tx: ManagedTransaction, batch: GraphBatch) -> None:
        # 文档替换语义：先删这篇文献上一次贡献的节点与边，再写本次结果。
        # 共享概念节点不带 doc_id，不会被删；其他文献的节点与边也不受影响。
        tx.run(
            """
            MATCH (node)
            WHERE node.doc_id = $doc_id
            DETACH DELETE node
            """,
            doc_id=batch.doc_id,
        ).consume()
        # 指向共享概念节点的边不在上面的 DETACH 范围内（两端都存活），
        # 必须按边自身的 doc_id 单独删除，否则重跑会叠加重复边。
        tx.run(
            """
            MATCH ()-[edge]->()
            WHERE edge.doc_id = $doc_id
            DELETE edge
            """,
            doc_id=batch.doc_id,
        ).consume()

        tx.run(
            """
            UNWIND $rows AS row
            MERGE (concept:Concept {id: row.id})
            SET concept.name = row.name,
                concept.type = row.type,
                concept.doc_id = row.doc_id
            SET concept += row.attributes
            """,
            rows=[
                {
                    "id": concept.id,
                    "name": concept.name,
                    "type": concept.type,
                    # 共享概念的 doc_id 为 null，SET 会移除该属性，
                    # 这正是「共享节点不属于任何单篇文献」在图里的表达。
                    "doc_id": concept.doc_id,
                    "attributes": dict(concept.attributes),
                }
                for concept in batch.concepts
            ],
        ).consume()

        tx.run(
            """
            UNWIND $rows AS row
            MERGE (evidence:Evidence {id: row.id})
            SET evidence.doc_id = row.doc_id,
                evidence.page = row.page,
                evidence.bbox = row.bbox,
                evidence.charspan = row.charspan,
                evidence.text = row.text
            """,
            rows=[_evidence_row(segment) for segment in batch.evidence],
        ).consume()

        tx.run(
            """
            UNWIND $rows AS row
            MATCH (concept:Concept {id: row.concept_id})
            MATCH (evidence:Evidence {id: row.evidence_id})
            MERGE (concept)-[edge:EVIDENCED_BY {doc_id: row.doc_id}]->(evidence)
            SET edge.evidence_ids = row.evidence_ids,
                edge.confidence = row.confidence,
                edge.source = row.source,
                edge.doc_id = row.doc_id
            """,
            rows=[
                {
                    "concept_id": link.concept_id,
                    "evidence_id": link.evidence_id,
                    "doc_id": link.provenance.doc_id,
                    "evidence_ids": list(link.provenance.evidence_ids),
                    "confidence": link.provenance.confidence,
                    "source": link.provenance.source,
                }
                for link in batch.evidence_links
            ],
        ).consume()

        rows_by_type: dict[str, list[dict[str, Any]]] = {}
        for relation in batch.relations:
            if not _RELATION_TYPE_PATTERN.fullmatch(relation.type):
                raise ValueError(f"非法关系类型：{relation.type}")
            rows_by_type.setdefault(relation.type.upper(), []).append(
                {
                    "head_id": relation.head_id,
                    "tail_id": relation.tail_id,
                    "doc_id": relation.provenance.doc_id,
                    "evidence_ids": list(relation.provenance.evidence_ids),
                    "confidence": relation.provenance.confidence,
                    "source": relation.provenance.source,
                }
            )

        for relation_type, rows in rows_by_type.items():
            # 关系类型已过白名单正则，才能安全插值；Neo4j 不支持参数化关系类型。
            # MERGE 带上 doc_id：同一对概念在不同文献里的断言保留为并行边。
            query = cast(
                LiteralString,
                f"""
                UNWIND $rows AS row
                MATCH (head:Concept {{id: row.head_id}})
                MATCH (tail:Concept {{id: row.tail_id}})
                MERGE (head)-[edge:{relation_type} {{doc_id: row.doc_id}}]->(tail)
                SET edge.evidence_ids = row.evidence_ids,
                    edge.confidence = row.confidence,
                    edge.source = row.source,
                    edge.doc_id = row.doc_id
                """,
            )
            tx.run(query, rows=rows).consume()


def _evidence_row(segment: EvidenceSegment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "doc_id": segment.doc_id,
        "page": segment.page,
        "bbox": [
            segment.bbox.x0,
            segment.bbox.y0,
            segment.bbox.x1,
            segment.bbox.y1,
        ],
        "charspan": [segment.charspan.start, segment.charspan.end],
        "text": segment.text,
    }

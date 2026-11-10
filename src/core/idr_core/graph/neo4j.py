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
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (concept:Concept {id: row.id})
            SET concept.name = row.name,
                concept.type = row.type,
                concept.doc_id = row.doc_id
            """,
            rows=[
                {
                    "id": concept.id,
                    "name": concept.name,
                    "type": concept.type,
                    "doc_id": concept.doc_id,
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
            MERGE (concept)-[edge:EVIDENCED_BY]->(evidence)
            SET edge.evidence_ids = row.evidence_ids,
                edge.confidence = row.confidence,
                edge.source = row.source
            """,
            rows=[
                {
                    "concept_id": link.concept_id,
                    "evidence_id": link.evidence_id,
                    "evidence_ids": list(link.evidence_ids),
                    "confidence": link.confidence,
                    "source": link.source,
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
                    "evidence_ids": list(relation.evidence_ids),
                    "confidence": relation.confidence,
                    "source": relation.source,
                }
            )

        for relation_type, rows in rows_by_type.items():
            # 关系类型已过白名单正则，才能安全插值；Neo4j 不支持参数化关系类型。
            query = cast(
                LiteralString,
                f"""
                UNWIND $rows AS row
                MATCH (head:Concept {{id: row.head_id}})
                MATCH (tail:Concept {{id: row.tail_id}})
                MERGE (head)-[edge:{relation_type}]->(tail)
                SET edge.evidence_ids = row.evidence_ids,
                    edge.confidence = row.confidence,
                    edge.source = row.source
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

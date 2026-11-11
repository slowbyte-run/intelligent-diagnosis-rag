"""Neo4j 图存储适配器的集成契约测试。

需要 docker compose 起的真实 Neo4j，且已 source .env；默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from neo4j import Driver, GraphDatabase

from idr_core.graph import (
    ConceptEvidenceLink,
    ConceptNode,
    ConceptRelation,
    GraphBatch,
    Provenance,
)
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.models import BBox, CharSpan, EvidenceSegment

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not NEO4J_PASSWORD,
        reason="需要 NEO4J_PASSWORD（先 source .env）",
    ),
]


@pytest.fixture
def driver() -> Iterator[Driver]:
    with GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    ) as driver:
        yield driver


@pytest.fixture
def graph_keys(driver: Driver) -> Iterator[tuple[str, str, str]]:
    """两篇文献的标识与共享节点的唯一前缀；结束后清掉本测试写入的图数据。"""
    unique = uuid4().hex
    first = f"it-{unique}-a"
    second = f"it-{unique}-b"
    prefix = f"concept:{unique}"
    yield first, second, prefix
    with driver.session() as session:
        session.run(
            """
            MATCH (node)
            WHERE node.doc_id IN $doc_ids OR node.id STARTS WITH $prefix
            DETACH DELETE node
            """,
            doc_ids=[first, second],
            prefix=prefix,
        ).consume()
        session.run(
            """
            MATCH ()-[edge]->()
            WHERE edge.doc_id IN $doc_ids
            DELETE edge
            """,
            doc_ids=[first, second],
        ).consume()


def make_batch(
    doc_id: str,
    *,
    prefix: str,
    evidence_text: str,
    scoped_name: str,
    confidence: float = 0.8,
) -> GraphBatch:
    """一篇文献的写入批次：两个共享概念、一个文档作用域概念和一条关系。"""
    evidence = EvidenceSegment(
        id=f"ev:{doc_id}:1:0-{len(evidence_text)}",
        doc_id=doc_id,
        page=1,
        bbox=BBox(72.0, 72.0, 540.0, 720.0),
        charspan=CharSpan(0, len(evidence_text)),
        text=evidence_text,
    )
    head = ConceptNode(
        id=f"{prefix}-head",
        name="entity-a",
        type="kind_a",
        doc_id=None,
        attributes={"position": "part_a"},
    )
    tail = ConceptNode(
        id=f"{prefix}-tail",
        name="entity-b",
        type="kind_b",
        doc_id=None,
        attributes={"position": "part_b"},
    )
    scoped = ConceptNode(
        id=f"{doc_id}:concept:{scoped_name}",
        name=scoped_name,
        type="kind_c",
        doc_id=doc_id,
    )
    return GraphBatch(
        doc_id=doc_id,
        concepts=(head, tail, scoped),
        evidence=(evidence,),
        relations=(
            ConceptRelation(
                head_id=head.id,
                type="linked_to",
                tail_id=tail.id,
                provenance=Provenance(
                    doc_id=doc_id,
                    evidence_ids=(evidence.id,),
                    confidence=confidence,
                    source="extracted",
                ),
            ),
        ),
        evidence_links=(
            ConceptEvidenceLink(
                concept_id=head.id,
                provenance=Provenance(
                    doc_id=doc_id,
                    evidence_ids=(evidence.id,),
                    confidence=0.9,
                    source="extracted",
                ),
            ),
            ConceptEvidenceLink(
                concept_id=scoped.id,
                provenance=Provenance(
                    doc_id=doc_id,
                    evidence_ids=(evidence.id,),
                    confidence=0.85,
                    source="extracted",
                ),
            ),
        ),
    )


def store() -> Neo4jGraphStore:
    return Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


def test_written_concepts_and_evidence_are_queryable(
    driver: Driver, graph_keys: tuple[str, str, str]
) -> None:
    doc_id, _, prefix = graph_keys
    batch = make_batch(
        doc_id,
        prefix=prefix,
        evidence_text="entity-a relates to entity-b.",
        scoped_name="entity-c",
    )
    graph_store = store()
    try:
        graph_store.write_document(batch)
    finally:
        graph_store.close()

    concept_ids = [concept.id for concept in batch.concepts]
    with driver.session() as session:
        records = list(
            session.run(
                """
                MATCH (concept:Concept)-[link:EVIDENCED_BY]->
                      (evidence:Evidence {doc_id: $doc_id})
                WHERE concept.id IN $concept_ids
                RETURN concept.id AS concept_id,
                       concept.name AS name,
                       concept.type AS type,
                       concept.doc_id AS concept_doc_id,
                       concept.position AS position,
                       evidence.id AS evidence_id,
                       evidence.page AS page,
                       evidence.bbox AS bbox,
                       evidence.charspan AS charspan,
                       link.doc_id AS link_doc_id,
                       link.evidence_ids AS link_evidence_ids,
                       link.confidence AS link_confidence,
                       link.source AS link_source
                ORDER BY concept_id
                """,
                doc_id=doc_id,
                concept_ids=concept_ids,
            )
        )

    assert len(records) == 2
    by_name = {record["name"]: record for record in records}
    assert set(by_name) == {"entity-a", "entity-c"}
    # 共享概念不带单篇文献归属；文档作用域概念带 doc_id。
    assert by_name["entity-a"]["concept_doc_id"] is None
    assert by_name["entity-a"]["position"] == "part_a"
    assert by_name["entity-c"]["concept_doc_id"] == doc_id
    assert all(record["evidence_id"] == batch.evidence[0].id for record in records)
    assert all(record["page"] == 1 for record in records)
    assert all(record["bbox"] == [72.0, 72.0, 540.0, 720.0] for record in records)
    assert all(
        record["charspan"] == [0, len(batch.evidence[0].text)]
        for record in records
    )
    # 每条边都带文档归属，删除与并行边才能按文献隔离。
    assert all(record["link_doc_id"] == doc_id for record in records)
    assert all(
        record["link_evidence_ids"] == [batch.evidence[0].id] for record in records
    )
    assert [record["link_confidence"] for record in records] == [0.9, 0.85]
    assert all(record["link_source"] == "extracted" for record in records)

    with driver.session() as session:
        relation = session.run(
            """
            MATCH (head:Concept {id: $head_id})
                  -[edge:LINKED_TO]->
                  (tail:Concept {id: $tail_id})
            RETURN edge.doc_id AS doc_id,
                   edge.evidence_ids AS evidence_ids,
                   edge.confidence AS confidence,
                   edge.source AS source
            """,
            head_id=batch.relations[0].head_id,
            tail_id=batch.relations[0].tail_id,
        ).single()

    assert relation is not None
    assert relation["doc_id"] == doc_id
    assert relation["evidence_ids"] == [batch.evidence[0].id]
    assert relation["confidence"] == 0.8
    assert relation["source"] == "extracted"


def test_rewriting_same_batch_does_not_duplicate_graph(
    driver: Driver, graph_keys: tuple[str, str, str]
) -> None:
    doc_id, _, prefix = graph_keys
    batch = make_batch(
        doc_id,
        prefix=prefix,
        evidence_text="entity-a relates to entity-b.",
        scoped_name="entity-c",
    )
    graph_store = store()
    try:
        graph_store.write_document(batch)
        graph_store.write_document(batch)
    finally:
        graph_store.close()

    concept_ids = [concept.id for concept in batch.concepts]
    with driver.session() as session:
        counts = session.run(
            """
            MATCH (concept:Concept)
            WHERE concept.id IN $concept_ids
            WITH count(concept) AS concepts
            MATCH (evidence:Evidence {doc_id: $doc_id})
            WITH concepts, count(evidence) AS evidence_nodes
            MATCH (concept:Concept)-[link:EVIDENCED_BY]->
                  (:Evidence {doc_id: $doc_id})
            WHERE concept.id IN $concept_ids
            WITH concepts, evidence_nodes, count(link) AS links
            MATCH (head:Concept)-[relation:LINKED_TO]->(tail:Concept)
            WHERE head.id IN $concept_ids AND tail.id IN $concept_ids
            RETURN concepts, evidence_nodes, links, count(relation) AS relations
            """,
            doc_id=doc_id,
            concept_ids=concept_ids,
        ).single()

    assert counts is not None
    assert dict(counts) == {
        "concepts": 3,
        "evidence_nodes": 1,
        "links": 2,
        "relations": 1,
    }


def test_reingesting_one_document_replaces_only_its_contribution(
    driver: Driver, graph_keys: tuple[str, str, str]
) -> None:
    first, second, prefix = graph_keys
    first_before = make_batch(
        first,
        prefix=prefix,
        evidence_text="first paper evidence.",
        scoped_name="only-first-before",
        confidence=0.8,
    )
    other = make_batch(
        second,
        prefix=prefix,
        evidence_text="second paper evidence.",
        scoped_name="only-second",
        confidence=0.7,
    )
    graph_store = store()
    try:
        graph_store.write_document(first_before)
        graph_store.write_document(other)

        # 重跑第一篇：模型这次抽到另一个自由词，证据段落也随之改变。
        first_after = make_batch(
            first,
            prefix=prefix,
            evidence_text="first paper evidence, revised.",
            scoped_name="only-first-after",
            confidence=0.95,
        )
        graph_store.write_document(first_after)
    finally:
        graph_store.close()

    concept_ids = [
        concept.id
        for batch in (first_before, other, first_after)
        for concept in batch.concepts
    ]
    with driver.session() as session:
        concepts = {
            record["id"]: record
            for record in session.run(
                """
                MATCH (concept:Concept)
                WHERE concept.id IN $concept_ids
                RETURN concept.id AS id, concept.doc_id AS doc_id
                """,
                concept_ids=concept_ids,
            )
        }
        relations = list(
            session.run(
                """
                MATCH (head:Concept)-[edge:LINKED_TO]->(tail:Concept)
                WHERE head.id = $head_id AND tail.id = $tail_id
                RETURN edge.doc_id AS doc_id,
                       edge.evidence_ids AS evidence_ids,
                       edge.confidence AS confidence
                ORDER BY doc_id
                """,
                head_id=f"{prefix}-head",
                tail_id=f"{prefix}-tail",
            )
        )
        evidence_ids = {
            record["id"]
            for record in session.run(
                """
                MATCH (evidence:Evidence)
                WHERE evidence.id IN $evidence_ids
                RETURN evidence.id AS id
                """,
                evidence_ids=[
                    segment.id
                    for batch in (first_before, other, first_after)
                    for segment in batch.evidence
                ],
            )
        }

    # 共享概念节点不被任何一篇的重跑删除，也不带单篇归属。
    assert concepts[f"{prefix}-head"]["doc_id"] is None
    assert concepts[f"{prefix}-tail"]["doc_id"] is None

    # 第一篇旧贡献被替换，第二篇原样保留。
    assert f"{first}:concept:only-first-before" not in concepts
    assert concepts[f"{first}:concept:only-first-after"]["doc_id"] == first
    assert concepts[f"{second}:concept:only-second"]["doc_id"] == second

    # 同一对共享概念在每个文献各留一条并行边；第一篇那条被更新。
    assert [(record["doc_id"], record["confidence"]) for record in relations] == [
        (first, 0.95),
        (second, 0.7),
    ]
    assert [record["evidence_ids"] for record in relations] == [
        [first_after.evidence[0].id],
        [other.evidence[0].id],
    ]

    # 第一篇被替换掉的证据节点已删除，第二篇的证据节点仍在。
    assert first_before.evidence[0].id not in evidence_ids
    assert first_after.evidence[0].id in evidence_ids
    assert other.evidence[0].id in evidence_ids

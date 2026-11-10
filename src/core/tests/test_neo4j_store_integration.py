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
def doc_id(driver: Driver) -> Iterator[str]:
    identifier = f"it-{uuid4().hex}"
    yield identifier
    with driver.session() as session:
        session.run(
            "MATCH (node) WHERE node.doc_id = $doc_id DETACH DELETE node",
            doc_id=identifier,
        ).consume()


def make_batch(doc_id: str) -> GraphBatch:
    text = "entity-a relates to entity-b."
    evidence = EvidenceSegment(
        id=f"ev:{doc_id}:1:0-{len(text)}",
        doc_id=doc_id,
        page=1,
        bbox=BBox(72.0, 72.0, 540.0, 720.0),
        charspan=CharSpan(0, len(text)),
        text=text,
    )
    first = ConceptNode(
        id=f"{doc_id}:concept:entity-a",
        name="entity-a",
        type="kind_a",
        doc_id=doc_id,
    )
    second = ConceptNode(
        id=f"{doc_id}:concept:entity-b",
        name="entity-b",
        type="kind_b",
        doc_id=doc_id,
    )
    return GraphBatch(
        doc_id=doc_id,
        concepts=(first, second),
        evidence=(evidence,),
        relations=(
            ConceptRelation(
                head_id=first.id,
                type="linked_to",
                tail_id=second.id,
                evidence_ids=(evidence.id,),
                confidence=0.8,
                source="extracted",
            ),
        ),
        evidence_links=(
            ConceptEvidenceLink(
                concept_id=first.id,
                evidence_id=evidence.id,
                evidence_ids=(evidence.id,),
                confidence=0.9,
                source="extracted",
            ),
            ConceptEvidenceLink(
                concept_id=second.id,
                evidence_id=evidence.id,
                evidence_ids=(evidence.id,),
                confidence=0.85,
                source="extracted",
            ),
        ),
    )


def store() -> Neo4jGraphStore:
    return Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


def test_written_concepts_and_evidence_are_queryable(driver: Driver, doc_id: str) -> None:
    batch = make_batch(doc_id)
    graph_store = store()
    try:
        graph_store.write_document(batch)
    finally:
        graph_store.close()

    with driver.session() as session:
        records = list(
            session.run(
                """
                MATCH (concept:Concept {doc_id: $doc_id})
                      -[link:EVIDENCED_BY]->
                      (evidence:Evidence {doc_id: $doc_id})
                RETURN concept.id AS concept_id,
                       concept.name AS name,
                       concept.type AS type,
                       evidence.id AS evidence_id,
                       evidence.page AS page,
                       evidence.bbox AS bbox,
                       evidence.charspan AS charspan,
                       link.evidence_ids AS link_evidence_ids,
                       link.confidence AS link_confidence,
                       link.source AS link_source
                ORDER BY concept_id
                """,
                doc_id=doc_id,
            )
        )

    assert len(records) == 2
    assert [record["name"] for record in records] == ["entity-a", "entity-b"]
    assert all(record["evidence_id"] == batch.evidence[0].id for record in records)
    assert all(record["page"] == 1 for record in records)
    assert all(record["bbox"] == [72.0, 72.0, 540.0, 720.0] for record in records)
    assert all(record["charspan"] == [0, len(batch.evidence[0].text)] for record in records)
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
            RETURN edge.evidence_ids AS evidence_ids,
                   edge.confidence AS confidence,
                   edge.source AS source
            """,
            head_id=batch.relations[0].head_id,
            tail_id=batch.relations[0].tail_id,
        ).single()

    assert relation is not None
    assert relation["evidence_ids"] == [batch.evidence[0].id]
    assert relation["confidence"] == 0.8
    assert relation["source"] == "extracted"


def test_rewriting_same_batch_does_not_duplicate_graph(driver: Driver, doc_id: str) -> None:
    batch = make_batch(doc_id)
    graph_store = store()
    try:
        graph_store.write_document(batch)
        graph_store.write_document(batch)
    finally:
        graph_store.close()

    with driver.session() as session:
        counts = session.run(
            """
            MATCH (concept:Concept {doc_id: $doc_id})
            WITH count(concept) AS concepts
            MATCH (evidence:Evidence {doc_id: $doc_id})
            WITH concepts, count(evidence) AS evidence_nodes
            MATCH (:Concept {doc_id: $doc_id})-[link:EVIDENCED_BY]->
                  (:Evidence {doc_id: $doc_id})
            WITH concepts, evidence_nodes, count(link) AS links
            MATCH (:Concept {doc_id: $doc_id})-[relation:LINKED_TO]->
                  (:Concept {doc_id: $doc_id})
            RETURN concepts, evidence_nodes, links, count(relation) AS relations
            """,
            doc_id=doc_id,
        ).single()

    assert counts is not None
    assert dict(counts) == {
        "concepts": 2,
        "evidence_nodes": 1,
        "links": 2,
        "relations": 1,
    }

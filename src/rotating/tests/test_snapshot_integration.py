"""图谱快照在真实图谱上的契约：可重复、闭包完整、提交的资产没有过期。

需要 docker compose 起的 Neo4j；默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration

这里验证的是「仓库里那份快照确实是这张图的导出」：同一张图连导两次字节
一致；重新导出的内容与提交进版本库的文件完全相同；金标准绑定的概念与证据
段都能在快照里找到。图谱一旦变化，这份测试会先红，提醒重新导出，而不是让
评估在无声中换了一张图。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.graph.snapshot import load_snapshot, serialize_snapshot
from idr_rotating.evaluation import load_golden_set
from idr_rotating.snapshot_export import DEFAULT_SNAPSHOT_PATH, export_snapshot

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
def graph() -> Iterator[Neo4jGraphStore]:
    store = Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        yield store
    finally:
        store.close()


def test_exporting_the_same_graph_twice_is_byte_identical(
    graph: Neo4jGraphStore,
) -> None:
    questions = load_golden_set().questions

    first = export_snapshot(graph, questions)
    second = export_snapshot(graph, questions)

    assert serialize_snapshot(first) == serialize_snapshot(second)
    assert first.checksum == second.checksum


def test_committed_snapshot_is_the_current_export_of_the_live_graph(
    graph: Neo4jGraphStore,
) -> None:
    """仓库里的快照必须与线上图谱当次导出一致：资产过期时先红，而不是悄悄错分。"""
    questions = load_golden_set().questions

    exported = export_snapshot(graph, questions)
    committed = load_snapshot(DEFAULT_SNAPSHOT_PATH)

    assert committed.checksum == exported.checksum
    assert committed == exported


def test_every_golden_binding_resolves_inside_the_committed_snapshot() -> None:
    questions = load_golden_set().questions
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    concept_ids = {concept.id for concept in snapshot.concepts}
    evidence_ids = {record.id for record in snapshot.evidence}

    missing_concepts = sorted(
        {
            concept_id
            for question in questions
            for concept_id in question.expected_concepts
            if concept_id not in concept_ids
        }
    )
    missing_evidence = sorted(
        {
            evidence_id
            for question in questions
            for evidence_id in question.expected_evidence_ids
            if evidence_id not in evidence_ids
        }
    )

    assert missing_concepts == []
    assert missing_evidence == []


def test_snapshot_is_a_closure_not_a_dump_of_the_whole_graph() -> None:
    """整库有数千段证据；快照只含闭包内的一百多段，才可能进版本库审阅。"""
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    with GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    ) as driver:
        with driver.session() as session:
            total = session.run(
                "MATCH (evidence:Evidence) RETURN count(evidence) AS total"
            ).single()
    assert total is not None
    total_evidence = int(total["total"])

    assert 0 < len(snapshot.evidence) < total_evidence // 10
    assert DEFAULT_SNAPSHOT_PATH.stat().st_size < 512 * 1024

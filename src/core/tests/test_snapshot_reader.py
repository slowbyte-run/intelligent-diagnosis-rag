"""快照读端口：快照 JSON 可以直接当 ``GraphReader`` 用，行为与在线端口一致。

这些测试只用内存里构造的小快照，不读提交进版本库的资产、也不连 Neo4j：
读端口是检索评估与图数据库之间唯一的缝，口径必须在这里钉死——按起点与关系
类型过滤、并行边原样返回、缺失标识不编造、同页证据按页面位置排序。
"""

from __future__ import annotations

from pathlib import Path

from idr_core.graph.read import Concept, EvidenceRecord, FactEdge, aggregate_facts
from idr_core.graph.snapshot import GraphSnapshot, SnapshotClosure, write_snapshot
from idr_core.graph.snapshot_reader import SnapshotGraphReader
from idr_core.models import CharSpan

DOC_A = "0123456789abcdef"
DOC_B = "fedcba9876543210"
ALPHA = "concept:alpha"
BETA = "concept:beta"
GAMMA = "concept:gamma"

EV_A1 = f"ev:{DOC_A}:1:0-10"
EV_A2 = f"ev:{DOC_A}:1:10-20"
EV_A3 = f"ev:{DOC_A}:2:0-5"
EV_B1 = f"ev:{DOC_B}:1:0-5"


def build_snapshot() -> GraphSnapshot:
    """两篇文献、三个概念、四条边；其中一条是跨文献的并行边。

    同页两段证据故意按标识逆序构造（``...:1:10-20`` 排在 ``...:1:0-10`` 前），
    用来验证 ``evidence_by_page`` 是页面位置排序而不是文件顺序。
    """
    return GraphSnapshot.build(
        closure=SnapshotClosure(
            seed_concept_ids=(ALPHA,),
            relation_types=("indicated_by", "has_component"),
            structural_relations=("has_component",),
            max_hops=2,
        ),
        concepts=(
            Concept(id=ALPHA, name="甲", type="type_a"),
            Concept(id=BETA, name="乙", type="type_b"),
            Concept(id=GAMMA, name="丙", type="type_c"),
        ),
        edges=(
            FactEdge(
                head_id=ALPHA,
                relation="indicated_by",
                tail_id=BETA,
                evidence_ids=(EV_B1,),
                confidence=0.6,
                doc_id=DOC_B,
            ),
            FactEdge(
                head_id=ALPHA,
                relation="indicated_by",
                tail_id=BETA,
                evidence_ids=(EV_A1,),
                confidence=0.9,
                doc_id=DOC_A,
            ),
            FactEdge(
                head_id=ALPHA,
                relation="has_component",
                tail_id=GAMMA,
                evidence_ids=(EV_A2,),
                confidence=0.8,
                doc_id=DOC_A,
            ),
            FactEdge(
                head_id=BETA,
                relation="indicated_by",
                tail_id=GAMMA,
                evidence_ids=(EV_A3,),
                confidence=0.7,
                doc_id=DOC_A,
            ),
        ),
        evidence=(
            EvidenceRecord(
                id=EV_A2,
                doc_id=DOC_A,
                page=1,
                text="第二段证据",
                charspan=CharSpan(10, 20),
            ),
            EvidenceRecord(
                id=EV_A1,
                doc_id=DOC_A,
                page=1,
                text="第一段证据",
                charspan=CharSpan(0, 10),
            ),
            EvidenceRecord(
                id=EV_A3,
                doc_id=DOC_A,
                page=2,
                text="第二页",
                charspan=CharSpan(0, 5),
            ),
            EvidenceRecord(
                id=EV_B1,
                doc_id=DOC_B,
                page=1,
                text="另一篇",
                charspan=CharSpan(0, 5),
            ),
        ),
    )


def test_facts_from_returns_unaggregated_parallel_edges_for_one_head() -> None:
    reader = SnapshotGraphReader(build_snapshot())

    edges = reader.facts_from(ALPHA, ("indicated_by",))

    assert [(edge.tail_id, edge.doc_id) for edge in edges] == [
        (BETA, DOC_A),
        (BETA, DOC_B),
    ]
    assert {edge.confidence for edge in edges} == {0.9, 0.6}
    # 聚合是读侧的事：端口必须把两篇文献的并行边原样交出去。
    aggregated = aggregate_facts(edges)
    assert len(aggregated) == 1
    assert aggregated[0].evidence_ids == (EV_A1, EV_B1)


def test_facts_from_filters_by_relation_type_and_head() -> None:
    reader = SnapshotGraphReader(build_snapshot())

    assert [
        edge.relation for edge in reader.facts_from(ALPHA, ("has_component",))
    ] == ["has_component"]
    assert reader.facts_from(ALPHA, ()) == ()
    assert reader.facts_from(BETA, ("has_component",)) == ()
    assert reader.facts_from("concept:missing", ("indicated_by",)) == ()


def test_concepts_by_ids_omits_missing_ids_and_keeps_one_record_per_id() -> None:
    reader = SnapshotGraphReader(build_snapshot())

    found = reader.concepts_by_ids([BETA, "concept:missing", ALPHA, BETA])

    assert [concept.id for concept in found] == [ALPHA, BETA]
    assert found[0].name == "甲"
    assert found[0].type == "type_a"


def test_evidence_by_ids_omits_missing_ids_and_keeps_one_record_per_id() -> None:
    reader = SnapshotGraphReader(build_snapshot())

    found = reader.evidence_by_ids([EV_A2, EV_A2, "ev:missing:1:0-1"])

    assert [record.id for record in found] == [EV_A2]
    assert found[0].text == "第二段证据"


def test_evidence_by_page_is_scoped_to_the_page_and_sorted_by_position() -> None:
    reader = SnapshotGraphReader(build_snapshot())

    first_page = reader.evidence_by_page(DOC_A, 1)

    assert [record.id for record in first_page] == [EV_A1, EV_A2]
    assert reader.evidence_by_page(DOC_A, 2)[0].id == EV_A3
    assert reader.evidence_by_page(DOC_A, 3) == ()
    assert reader.evidence_by_page(DOC_B, 1)[0].id == EV_B1


def test_reader_caches_nothing_from_outside_the_snapshot(tmp_path: Path) -> None:
    """快照里没有的标识就是查不到；端口不替调用方补数据。"""
    path = tmp_path / "snapshot.json"
    write_snapshot(path, build_snapshot())

    reader = SnapshotGraphReader.load(path)

    assert reader.snapshot.checksum == build_snapshot().checksum
    assert reader.facts_from(GAMMA, ("indicated_by",)) == ()
    assert reader.concepts_by_ids(["concept:ghost"]) == ()

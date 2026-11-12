"""图谱快照格式：自描述版本、稳定排序、内容校验和与加载校验。

快照是评估资产：同一份图谱导出两次必须给出同样的字节与校验和，任何人拿到
文件都能核对它没被改动过。这里测的是格式本身，不涉及任何图存储实现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idr_core.anchors import EvidenceAnchor
from idr_core.graph.read import Concept, EvidenceRecord, FactEdge
from idr_core.graph.snapshot import (
    SNAPSHOT_FORMAT,
    SNAPSHOT_SCHEMA_VERSION,
    GraphSnapshot,
    SnapshotClosure,
    SnapshotError,
    load_snapshot,
    serialize_snapshot,
    snapshot_checksum,
    write_snapshot,
)
from idr_core.models import CharSpan

FIRST_DOC = "a" * 16
SECOND_DOC = "b" * 16
FIRST_EVIDENCE = f"ev:{FIRST_DOC}:1:0-10"
SECOND_EVIDENCE = f"ev:{SECOND_DOC}:2:20-40"


def make_closure(
    *,
    seeds: tuple[str, ...] = ("concept:beta", "concept:alpha"),
    relations: tuple[str, ...] = ("linked_to", "part_of"),
    structural: tuple[str, ...] = ("part_of",),
) -> SnapshotClosure:
    return SnapshotClosure(
        seed_concept_ids=seeds,
        relation_types=relations,
        structural_relations=structural,
        max_hops=2,
    )


def make_concepts() -> tuple[Concept, ...]:
    return (
        Concept(id="concept:beta", name="节点乙", type="kind_b"),
        Concept(id="concept:alpha", name="节点甲", type="kind_a"),
        Concept(id="concept:gamma", name="节点丙", type="kind_c"),
    )


def make_edges() -> tuple[FactEdge, ...]:
    return (
        FactEdge(
            head_id="concept:beta",
            relation="part_of",
            tail_id="concept:gamma",
            evidence_ids=(SECOND_EVIDENCE, FIRST_EVIDENCE),
            confidence=0.5,
            doc_id=SECOND_DOC,
            anchors=(
                EvidenceAnchor(evidence_id=SECOND_EVIDENCE, start=3, end=7),
                EvidenceAnchor(
                    evidence_id=FIRST_EVIDENCE, start=1, end=4, ambiguous=True
                ),
            ),
        ),
        FactEdge(
            head_id="concept:alpha",
            relation="linked_to",
            tail_id="concept:beta",
            evidence_ids=(FIRST_EVIDENCE,),
            confidence=0.75,
            doc_id=FIRST_DOC,
        ),
    )


def make_evidence() -> tuple[EvidenceRecord, ...]:
    return (
        EvidenceRecord(
            id=SECOND_EVIDENCE,
            doc_id=SECOND_DOC,
            page=2,
            text="第二段原文",
            charspan=CharSpan(20, 40),
        ),
        EvidenceRecord(
            id=FIRST_EVIDENCE,
            doc_id=FIRST_DOC,
            page=1,
            text="第一段原文\n换行也原样保留",
            charspan=CharSpan(0, 10),
        ),
    )


def make_snapshot() -> GraphSnapshot:
    return GraphSnapshot.build(
        closure=make_closure(),
        concepts=make_concepts(),
        edges=make_edges(),
        evidence=make_evidence(),
    )


def test_snapshot_is_self_describing_and_round_trips(tmp_path: Path) -> None:
    snapshot = make_snapshot()
    target = tmp_path / "snapshot.json"

    checksum = write_snapshot(target, snapshot)
    document = json.loads(target.read_text("utf-8"))

    assert document["format"] == SNAPSHOT_FORMAT
    assert document["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert document["checksum"] == checksum
    assert checksum.startswith("sha256:")
    assert load_snapshot(target) == snapshot


def test_sections_are_sorted_so_diffs_do_not_depend_on_reader_order() -> None:
    document = json.loads(serialize_snapshot(make_snapshot()))

    assert [item["id"] for item in document["concepts"]] == [
        "concept:alpha",
        "concept:beta",
        "concept:gamma",
    ]
    assert [
        (item["head_id"], item["relation"], item["tail_id"], item["doc_id"])
        for item in document["edges"]
    ] == [
        ("concept:alpha", "linked_to", "concept:beta", FIRST_DOC),
        ("concept:beta", "part_of", "concept:gamma", SECOND_DOC),
    ]
    assert [item["id"] for item in document["evidence"]] == [
        FIRST_EVIDENCE,
        SECOND_EVIDENCE,
    ]
    # 边内部的证据清单与锚点同样定序：同一份图不能因为写入顺序不同产生两份快照。
    assert document["edges"][1]["evidence_ids"] == [FIRST_EVIDENCE, SECOND_EVIDENCE]
    assert [
        (anchor["evidence_id"], anchor["start"], anchor["end"], anchor["ambiguous"])
        for anchor in document["edges"][1]["anchors"]
    ] == [
        (FIRST_EVIDENCE, 1, 4, True),
        (SECOND_EVIDENCE, 3, 7, False),
    ]
    assert document["closure"]["seed_concept_ids"] == [
        "concept:alpha",
        "concept:beta",
    ]
    assert document["closure"]["relation_types"] == ["linked_to", "part_of"]


def test_same_content_in_any_input_order_gives_identical_bytes_and_checksum() -> None:
    snapshot = make_snapshot()
    reversed_snapshot = GraphSnapshot.build(
        closure=make_closure(seeds=("concept:alpha", "concept:beta")),
        concepts=tuple(reversed(make_concepts())),
        edges=tuple(reversed(make_edges())),
        evidence=tuple(reversed(make_evidence())),
    )

    assert serialize_snapshot(reversed_snapshot) == serialize_snapshot(snapshot)
    assert snapshot_checksum(reversed_snapshot) == snapshot_checksum(snapshot)


def test_checksum_changes_when_content_changes() -> None:
    snapshot = make_snapshot()
    edited_evidence = EvidenceRecord(
        id=FIRST_EVIDENCE,
        doc_id=FIRST_DOC,
        page=1,
        text="改了一个字",
        charspan=CharSpan(0, 10),
    )
    changed = GraphSnapshot.build(
        closure=make_closure(),
        concepts=make_concepts(),
        edges=make_edges(),
        evidence=(edited_evidence, make_evidence()[0]),
    )

    assert snapshot_checksum(changed) != snapshot_checksum(snapshot)


def test_checksum_is_over_content_not_over_whitespace(tmp_path: Path) -> None:
    snapshot = make_snapshot()
    path = tmp_path / "snapshot.json"
    checksum = write_snapshot(path, snapshot)
    document = json.loads(path.read_text("utf-8"))

    path.write_text(json.dumps(document, ensure_ascii=False, indent=4), encoding="utf-8")

    reloaded = load_snapshot(path)
    assert reloaded == snapshot
    assert snapshot_checksum(reloaded) == checksum


def test_load_rejects_a_file_whose_content_was_edited(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(path, make_snapshot())
    document = json.loads(path.read_text("utf-8"))
    document["evidence"][0]["text"] = "被改过的原文"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SnapshotError, match="校验和"):
        load_snapshot(path)


def test_build_rejects_an_edge_whose_evidence_text_is_missing() -> None:
    with pytest.raises(SnapshotError, match="证据段"):
        GraphSnapshot.build(
            closure=make_closure(),
            concepts=make_concepts(),
            edges=make_edges(),
            evidence=(make_evidence()[0],),
        )


def test_build_rejects_an_edge_endpoint_that_is_not_in_the_snapshot() -> None:
    with pytest.raises(SnapshotError, match="概念"):
        GraphSnapshot.build(
            closure=make_closure(),
            concepts=(
                Concept(id="concept:alpha", name="节点甲", type="kind_a"),
                Concept(id="concept:beta", name="节点乙", type="kind_b"),
            ),
            edges=make_edges(),
            evidence=make_evidence(),
        )


def test_build_rejects_duplicate_edges() -> None:
    edge = make_edges()[0]
    with pytest.raises(SnapshotError, match="重复"):
        GraphSnapshot.build(
            closure=make_closure(),
            concepts=make_concepts(),
            edges=(edge, edge),
            evidence=make_evidence(),
        )


def test_load_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(path, make_snapshot())
    document = json.loads(path.read_text("utf-8"))
    document["schema_version"] = SNAPSHOT_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SnapshotError, match="schema_version"):
        load_snapshot(path)


def test_load_rejects_an_unknown_field_in_a_record(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(path, make_snapshot())
    document = json.loads(path.read_text("utf-8"))
    document["concepts"][0]["unexpected"] = "字段拼错必须报错"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SnapshotError, match="unexpected"):
        load_snapshot(path)


def test_closure_metadata_is_required_to_describe_the_export_scope() -> None:
    with pytest.raises(SnapshotError, match="max_hops"):
        SnapshotClosure(
            seed_concept_ids=("concept:alpha",),
            relation_types=("linked_to",),
            structural_relations=("linked_to",),
            max_hops=0,
        )

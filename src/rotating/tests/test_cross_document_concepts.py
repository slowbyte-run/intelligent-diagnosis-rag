"""跨文献概念层：规范实例共享一个节点，自由词留在各自文献内。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest
from idr_rotating.pending import PendingKind, RejectionReason

from fakes import (
    FakeEmbedder,
    FakeGraphStore,
    FakeParser,
    FakeVectorStore,
    ScriptedLLM,
)

PAGE_BBOX = BBox(0.0, 0.0, 595.0, 842.0)
TEXT = "外圈剥落是常见故障，自由词概念也会在文献里反复出现。"
SHARED_FAULT_MODE_ID = "concept:bearing_outer_race_fatigue_spalling"


def doc_id_for(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def evidence_id(doc_id: str) -> str:
    return f"ev:{doc_id}:1:0-{len(TEXT)}"


def entity(name: str, type_: str, evidence: str) -> dict[str, object]:
    return {
        "name": name,
        "type": type_,
        "evidence_ids": [evidence],
        "quote": TEXT,
        "confidence": 0.9,
    }


def relation(head: str, type_: str, tail: str, evidence: str) -> dict[str, object]:
    return {
        "head": head,
        "type": type_,
        "tail": tail,
        "evidence_ids": [evidence],
        "quote": TEXT,
        "confidence": 0.8,
    }


def extraction(
    entities: list[dict[str, object]], relations: list[dict[str, object]]
) -> str:
    return json.dumps(
        {"entities": entities, "relations": relations}, ensure_ascii=False
    )


def deps(llm: ScriptedLLM, graph_store: FakeGraphStore) -> IngestDeps:
    return IngestDeps(
        llm=llm,
        parser=FakeParser([ParsedPage(number=1, text=TEXT, bbox=PAGE_BBOX)]),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )


def write_paper(directory: Path, name: str, content: bytes) -> None:
    (directory / name).write_bytes(content)


def test_same_canonical_concept_in_two_documents_is_one_node_with_both_evidence(
    tmp_path: Path,
) -> None:
    first = b"%PDF-1.4 first paper"
    second = b"%PDF-1.4 second paper"
    write_paper(tmp_path, "a.pdf", first)
    write_paper(tmp_path, "b.pdf", second)
    first_doc = doc_id_for(first)
    second_doc = doc_id_for(second)
    first_evidence = evidence_id(first_doc)
    second_evidence = evidence_id(second_doc)
    llm = ScriptedLLM(
        [
            extraction(
                [
                    entity("外圈剥落", "fault_mode", first_evidence),
                    entity("BPFO 倍频", "symptom", first_evidence),
                ],
                [relation("外圈剥落", "indicated_by", "BPFO 倍频", first_evidence)],
            ),
            extraction(
                [
                    entity("外套圈剥落", "fault_mode", second_evidence),
                    entity("润滑不良", "fault_cause", second_evidence),
                ],
                [relation("外套圈剥落", "has_cause", "润滑不良", second_evidence)],
            ),
        ]
    )
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps(llm, graph_store))

    assert [document.error for document in report.documents] == [None, None]

    # 两篇文献的「外圈剥落」落到同一个规范概念节点，且不带单篇文献归属。
    shared = graph_store.concepts[SHARED_FAULT_MODE_ID]
    assert shared.name == "轴承外圈疲劳剥落"
    assert shared.doc_id is None
    same_name = [
        node for node in graph_store.concepts.values() if node.name == shared.name
    ]
    assert len(same_name) == 1

    # 证据来自两篇文献：节点在两个文档各有一条带归属的证据边。
    assert graph_store.evidence_ids_for(shared.id) == {
        first_evidence,
        second_evidence,
    }
    shared_links = [
        link
        for link in graph_store.evidence_links
        if link.concept_id == shared.id
    ]
    assert {link.provenance.doc_id for link in shared_links} == {
        first_doc,
        second_doc,
    }

    # 规范实例之间的关系边也跨文献挂在同一个节点上，且各自带 doc_id。
    assert {relation.provenance.doc_id for relation in graph_store.relations} == {
        first_doc,
        second_doc,
    }
    assert {relation.head_id for relation in graph_store.relations} == {shared.id}
    assert {
        (
            relation.head_id,
            relation.type,
            relation.tail_id,
            relation.provenance.doc_id,
        )
        for relation in graph_store.relations
    } == {
        (SHARED_FAULT_MODE_ID, "indicated_by", "concept:bpfo_harmonics", first_doc),
        (
            SHARED_FAULT_MODE_ID,
            "has_cause",
            "concept:inadequate_lubrication",
            second_doc,
        ),
    }


def test_free_form_entity_in_two_documents_stays_document_scoped(
    tmp_path: Path,
) -> None:
    first = b"%PDF-1.4 first paper"
    second = b"%PDF-1.4 second paper"
    write_paper(tmp_path, "a.pdf", first)
    write_paper(tmp_path, "b.pdf", second)
    first_doc = doc_id_for(first)
    second_doc = doc_id_for(second)
    first_evidence = evidence_id(first_doc)
    second_evidence = evidence_id(second_doc)
    llm = ScriptedLLM(
        [
            extraction([entity("异常啸叫", "symptom", first_evidence)], []),
            extraction([entity("异常啸叫", "symptom", second_evidence)], []),
        ]
    )
    graph_store = FakeGraphStore()

    ingest(tmp_path, deps=deps(llm, graph_store))

    # 未命中本体规范实例的自由词无法跨文献判定为同一概念，各自成节点。
    first_id = f"{first_doc}:concept:symptom:异常啸叫"
    second_id = f"{second_doc}:concept:symptom:异常啸叫"
    assert set(graph_store.concepts) == {first_id, second_id}
    assert graph_store.concepts[first_id].doc_id == first_doc
    assert graph_store.concepts[second_id].doc_id == second_doc
    assert graph_store.evidence_ids_for(first_id) == {first_evidence}
    assert graph_store.evidence_ids_for(second_id) == {second_evidence}


def test_reingesting_one_document_replaces_only_its_own_contribution(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = b"%PDF-1.4 first paper"
    second = b"%PDF-1.4 second paper"
    write_paper(first_dir, "paper.pdf", first)
    write_paper(second_dir, "paper.pdf", second)
    first_doc = doc_id_for(first)
    second_doc = doc_id_for(second)
    first_evidence = evidence_id(first_doc)
    second_evidence = evidence_id(second_doc)
    graph_store = FakeGraphStore()

    ingest(
        first_dir,
        deps=deps(
            ScriptedLLM(
                [
                    extraction(
                        [
                            entity("外圈剥落", "fault_mode", first_evidence),
                            entity("自由概念甲", "symptom", first_evidence),
                        ],
                        [
                            relation(
                                "外圈剥落", "indicated_by", "自由概念甲", first_evidence
                            )
                        ],
                    )
                ]
            ),
            graph_store,
        ),
    )
    ingest(
        second_dir,
        deps=deps(
            ScriptedLLM(
                [
                    extraction(
                        [
                            entity("外圈剥落", "fault_mode", second_evidence),
                            entity("自由概念乙", "symptom", second_evidence),
                        ],
                        [
                            relation(
                                "外圈剥落", "indicated_by", "自由概念乙", second_evidence
                            )
                        ],
                    )
                ]
            ),
            graph_store,
        ),
    )

    # 重跑第一篇：模型这次抽到的是另一个自由词，旧的关系与节点都应被替换。
    ingest(
        first_dir,
        deps=deps(
            ScriptedLLM(
                [
                    extraction(
                        [
                            entity("外圈剥落", "fault_mode", first_evidence),
                            entity("自由概念丙", "symptom", first_evidence),
                        ],
                        [
                            relation(
                                "外圈剥落", "indicated_by", "自由概念丙", first_evidence
                            )
                        ],
                    )
                ]
            ),
            graph_store,
        ),
    )

    # 共享概念节点不随任一篇文献的重跑而消失。
    shared = graph_store.concepts[SHARED_FAULT_MODE_ID]
    assert shared.doc_id is None
    assert graph_store.evidence_ids_for(shared.id) == {first_evidence, second_evidence}

    # 第一篇旧贡献被删干净，第二篇原样保留。
    assert f"{first_doc}:concept:symptom:自由概念甲" not in graph_store.concepts
    assert f"{first_doc}:concept:symptom:自由概念丙" in graph_store.concepts
    assert f"{second_doc}:concept:symptom:自由概念乙" in graph_store.concepts

    first_relations = [
        relation
        for relation in graph_store.relations
        if relation.provenance.doc_id == first_doc
    ]
    second_relations = [
        relation
        for relation in graph_store.relations
        if relation.provenance.doc_id == second_doc
    ]
    assert [
        (relation.type, relation.tail_id) for relation in first_relations
    ] == [("indicated_by", f"{first_doc}:concept:symptom:自由概念丙")]
    assert [
        (relation.type, relation.tail_id) for relation in second_relations
    ] == [("indicated_by", f"{second_doc}:concept:symptom:自由概念乙")]

    # 第二篇的证据节点与边未被第一篇的重跑波及。
    assert graph_store.evidence_ids_for(f"{second_doc}:concept:symptom:自由概念乙") == {
        second_evidence
    }
    assert second_relations[0].provenance.evidence_ids == (second_evidence,)


def test_internal_types_from_model_output_are_queued_not_written(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 internal type fixture"
    write_paper(tmp_path, "paper.pdf", content)
    doc_id = doc_id_for(content)
    segment = evidence_id(doc_id)
    llm = ScriptedLLM(
        [
            extraction(
                [
                    entity("外圈剥落", "fault_mode", segment),
                    entity("原文段落", "evidence", segment),
                ],
                [relation("外圈剥落", "evidenced_by", "原文段落", segment)],
            )
        ]
    )
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps(llm, graph_store))

    document = report.documents[0]
    # 内部类型是模型越界，不是整篇文献的错误：合法抽取照常入库。
    assert document.error is None
    assert (document.entities, document.relations) == (1, 0)
    assert (document.rejected_entities, document.rejected_relations) == (1, 1)
    assert set(graph_store.concepts) == {SHARED_FAULT_MODE_ID}
    assert all(node.type != "evidence" for node in graph_store.concepts.values())
    assert graph_store.relations == []

    reviews = {
        (review.kind, review.value): review
        for review in document.pending_reviews
    }
    entity_review = reviews[(PendingKind.ENTITY, "evidence")]
    assert entity_review.reason is RejectionReason.INTERNAL_TYPE
    assert entity_review.frequency == 1

    relation_review = reviews[(PendingKind.RELATION, "evidenced_by")]
    assert relation_review.reason is RejectionReason.INTERNAL_TYPE
    assert relation_review.frequency == 1

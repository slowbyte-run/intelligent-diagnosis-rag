"""越界抽取的待审队列：拒绝入库，但按频次与原文例句进入报告。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest
from idr_rotating.pending import PendingKind, RejectionReason

from fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeParser, FakeVectorStore


def write_pdf(directory: Path, name: str, content: bytes) -> None:
    (directory / name).write_bytes(content)


def test_out_of_ontology_entities_and_relations_are_reported_with_examples(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 pending review fixture"
    write_pdf(tmp_path, "pending.pdf", content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "未知甲与未知乙都属于神秘现象，彼此之间存在玄妙联系；外圈剥落则不同。"
    segment_id = f"ev:{doc_id}:1:0-{len(text)}"
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "未知甲",
                    "type": "alien_type",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "未知乙",
                    "type": "alien_type",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "未知甲",
                    "type": "magic_link",
                    "tail": "未知乙",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "未知乙",
                    "type": "magic_link",
                    "tail": "未知甲",
                    "evidence_ids": [segment_id],
                    "confidence": 0.7,
                },
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    llm = FakeLLM(response)

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert (document.entities, document.relations) == (1, 0)
    assert (document.rejected_entities, document.rejected_relations) == (2, 2)
    # 本体越界是语义结果而不是格式错误，不触发重试。
    assert len(llm.prompts) == 1

    batch = graph_store.batches[0]
    assert {concept.name for concept in batch.concepts} == {"轴承外圈疲劳剥落"}
    assert batch.relations == ()

    reviews = {
        (review.kind, review.value): review for review in document.pending_reviews
    }
    entity_review = reviews[(PendingKind.ENTITY, "alien_type")]
    assert entity_review.reason is RejectionReason.TYPE_NOT_IN_ONTOLOGY
    assert entity_review.frequency == 2
    assert set(entity_review.names) == {"未知甲", "未知乙"}
    assert [example.text for example in entity_review.examples] == [text]

    relation_review = reviews[(PendingKind.RELATION, "magic_link")]
    assert relation_review.reason is RejectionReason.TYPE_NOT_IN_ONTOLOGY
    assert relation_review.frequency == 2
    assert [example.text for example in relation_review.examples] == [text]

    assert report.pending_reviews == document.pending_reviews


def test_relation_that_violates_domain_or_range_is_queued_not_written(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 relation mismatch fixture"
    write_pdf(tmp_path, "mismatch.pdf", content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落是故障模式，滚动轴承是部件；两者不能直接用表现为相连。"
    segment_id = f"ev:{doc_id}:1:0-{len(text)}"
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "滚动轴承",
                    "type": "component",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "滚动轴承",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert document.rejected_relations == 1
    assert graph_store.batches[0].relations == ()

    (review,) = document.pending_reviews
    assert review.kind is PendingKind.RELATION
    assert review.value == "indicated_by"
    assert review.reason is RejectionReason.DOMAIN_RANGE_MISMATCH
    assert review.frequency == 1
    assert review.examples[0].text == text
    assert review.examples[0].doc_id == doc_id
    assert review.examples[0].evidence_id == segment_id


def test_batch_report_merges_pending_frequency_across_documents(
    tmp_path: Path,
) -> None:
    first_content = b"%PDF-1.4 first pending document"
    second_content = b"%PDF-1.4 second pending document"
    write_pdf(tmp_path, "a.pdf", first_content)
    write_pdf(tmp_path, "b.pdf", second_content)
    first_doc_id = hashlib.sha256(first_content).hexdigest()[:16]
    second_doc_id = hashlib.sha256(second_content).hexdigest()[:16]
    text = "两篇文献都出现了本体之外的同一类神秘现象。"
    first_segment = f"ev:{first_doc_id}:1:0-{len(text)}"
    second_segment = f"ev:{second_doc_id}:1:0-{len(text)}"

    def response(name: str, segment_id: str) -> str:
        return json.dumps(
            {
                "entities": [
                    {
                        "name": name,
                        "type": "alien_type",
                        "evidence_ids": [segment_id],
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            },
            ensure_ascii=False,
        )

    llm = FakeLLM(
        [
            response("未知甲", first_segment),
            response("未知乙", second_segment),
        ]
    )
    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    (review,) = report.pending_reviews
    assert review.kind is PendingKind.ENTITY
    assert review.value == "alien_type"
    assert review.frequency == 2
    assert set(review.names) == {"未知甲", "未知乙"}
    assert {example.doc_id for example in review.examples} == {
        first_doc_id,
        second_doc_id,
    }


def test_ambiguous_alias_is_queued_instead_of_stored_as_raw_name(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 ambiguous alias fixture"
    write_pdf(tmp_path, "ambiguous.pdf", content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "塑性变形在本体里同时指向轴承与齿面两个实例。"
    segment_id = f"ev:{doc_id}:1:0-{len(text)}"
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "塑性变形",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert document.error is None
    assert document.entities == 0, "歧义别名不得以原文名字入库"
    assert document.rejected_entities == 1
    assert graph_store.batches[0].concepts == ()

    (review,) = document.pending_reviews
    assert review.kind is PendingKind.ENTITY
    assert review.reason is RejectionReason.AMBIGUOUS_ALIAS
    assert review.value == "fault_mode"
    assert "塑性变形" in review.label
    assert review.examples[0].text == text


def test_relation_with_an_unknown_endpoint_is_queued_not_written(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 unknown endpoint fixture"
    write_pdf(tmp_path, "endpoint.pdf", content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落与一个本体里不存在的概念被连在一起。"
    segment_id = f"ev:{doc_id}:1:0-{len(text)}"
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                }
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "不在抽取结果里的终点",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert document.error is None
    assert document.rejected_relations == 1
    assert graph_store.batches[0].relations == ()

    (review,) = document.pending_reviews
    assert review.kind is PendingKind.RELATION
    assert review.reason is RejectionReason.UNKNOWN_RELATION_ENDPOINT
    assert review.value == "indicated_by"
    assert "不在抽取结果里的终点" in review.label

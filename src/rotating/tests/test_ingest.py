"""入库门面的行为测试：只通过 ingest() 观察外部结果。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, IngestReport, ingest

from fakes import FakeGraphStore, FakeLLM, FakeParser


def content_doc_id(content: bytes) -> str:
    """文献标识由内容哈希推导，与文件名无关。"""
    return hashlib.sha256(content).hexdigest()[:16]


def evidence_id(doc_id: str, page: int, text: str) -> str:
    """证据段标识由文献标识、页码与字符区间稳定推导。"""
    return f"ev:{doc_id}:{page}:0-{len(text)}"


def write_pdf(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def test_single_document_ingest_writes_traceable_concepts(tmp_path: Path) -> None:
    content = b"%PDF-1.4 fake paper"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动。"
    segment_id = evidence_id(doc_id, 1, text)

    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.91,
                },
                {
                    "name": "BPFO 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "confidence": 0.88,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "BPFO 倍频",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(72.0, 72.0, 540.0, 720.0))]
        ),
        graph_store=graph_store,
    )

    report = ingest(tmp_path, deps=deps)

    assert isinstance(report, IngestReport)
    assert len(report.documents) == 1
    document = report.documents[0]
    assert document.doc_id == doc_id
    assert (document.evidence_segments, document.entities, document.relations) == (
        1,
        2,
        1,
    )
    assert document.error is None

    batch = graph_store.batches[0]
    assert [(segment.id, segment.page, segment.charspan, segment.bbox) for segment in batch.evidence] == [
        (segment_id, 1, CharSpan(0, len(text)), BBox(72.0, 72.0, 540.0, 720.0))
    ]

    concepts_by_id = {concept.id: concept for concept in batch.concepts}
    assert {concept.name for concept in batch.concepts} == {"外圈剥落", "BPFO 倍频"}
    assert {concept.type for concept in batch.concepts} == {"fault_mode", "symptom"}

    assert len(batch.evidence_links) == 2
    link_confidences = {
        (concepts_by_id[link.concept_id].name, link.confidence)
        for link in batch.evidence_links
    }
    assert link_confidences == {("外圈剥落", 0.91), ("BPFO 倍频", 0.88)}
    assert all(link.evidence_ids == (segment_id,) for link in batch.evidence_links)
    assert all(link.source == "extracted" for link in batch.evidence_links)

    (relation,) = batch.relations
    assert relation.type == "indicated_by"
    assert relation.evidence_ids == (segment_id,)
    assert relation.confidence == 0.8
    assert relation.source == "extracted"
    assert concepts_by_id[relation.head_id].name == "外圈剥落"
    assert concepts_by_id[relation.tail_id].name == "BPFO 倍频"


def test_extractions_that_fail_ontology_validation_are_not_written(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 fake paper with invalid extractions"
    write_pdf(tmp_path, "mixed.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为外圈故障频率；滚动轴承是部件。"
    segment_id = evidence_id(doc_id, 1, text)

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
                    "name": "外圈故障频率",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "滚动轴承",
                    "type": "component",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "神秘实体",
                    "type": "alien_type",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "无出处实体",
                    "type": "fault_mode",
                    "evidence_ids": [],
                    "confidence": 0.9,
                },
                {
                    "name": "假证据实体",
                    "type": "fault_mode",
                    "evidence_ids": ["ev:does-not-exist"],
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "外圈故障频率",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "magic_link",
                    "tail": "外圈故障频率",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "has_component",
                    "tail": "滚动轴承",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "滚动轴承",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "不存在",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
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
        ),
    )

    document = report.documents[0]
    assert (document.entities, document.relations) == (3, 1)
    assert (document.rejected_entities, document.rejected_relations) == (3, 4)

    batch = graph_store.batches[0]
    assert {concept.name for concept in batch.concepts} == {
        "外圈剥落",
        "外圈故障频率",
        "滚动轴承",
    }
    assert [relation.type for relation in batch.relations] == ["indicated_by"]
    assert len(batch.evidence_links) == 3


def test_reprocessing_same_document_yields_same_identifiers(tmp_path: Path) -> None:
    content = b"%PDF-1.4 stable identifiers"
    write_pdf(tmp_path, "stable.pdf", content)
    doc_id = content_doc_id(content)
    text = "内圈剥落表现为 BPFI 倍频。"
    segment_id = evidence_id(doc_id, 1, text)
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "内圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "BPFI 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "内圈剥落",
                    "type": "indicated_by",
                    "tail": "BPFI 倍频",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=graph_store,
    )

    ingest(tmp_path, deps=deps)
    ingest(tmp_path, deps=deps)

    first, second = graph_store.batches
    assert [segment.id for segment in first.evidence] == [
        segment.id for segment in second.evidence
    ]
    assert [concept.id for concept in first.concepts] == [
        concept.id for concept in second.concepts
    ]
    assert first.relations == second.relations


def test_limit_processes_only_the_first_documents(tmp_path: Path) -> None:
    write_pdf(tmp_path, "a.pdf", b"%PDF-1.4 first")
    write_pdf(tmp_path, "b.pdf", b"%PDF-1.4 second")
    text = "外圈剥落。"
    doc_id = content_doc_id(b"%PDF-1.4 first")
    segment_id = evidence_id(doc_id, 1, text)
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
            "relations": [],
        },
        ensure_ascii=False,
    )

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=FakeGraphStore(),
        ),
        limit=1,
    )

    assert [document.path.name for document in report.documents] == ["a.pdf"]

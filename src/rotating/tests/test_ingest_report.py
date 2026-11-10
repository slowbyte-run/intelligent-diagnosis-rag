"""批次报告在入库门面上的行为：模型调用成本、隔离、待审计与运行元数据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idr_core.llm import ModelDescriptor
from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest

from fakes import FakeEmbedder, FakeGraphStore, FakeParser, FakeVectorStore
from fakes import InstrumentedFakeLLM


def content_doc_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def evidence_id(doc_id: str, page: int, text: str) -> str:
    return f"ev:{doc_id}:{page}:0-{len(text)}"


def one_page(text: str) -> FakeParser:
    return FakeParser(
        [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
    )


def extraction_response(segment_id: str) -> str:
    return json.dumps(
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


def test_report_records_each_model_call_with_model_tokens_and_elapsed_time(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    text = "外圈剥落表现为 BPFO 倍频，且能量集中在高频段。"
    doc_id = content_doc_id(content)
    llm = InstrumentedFakeLLM(
        extraction_response(evidence_id(doc_id, 1, text)),
        model="deepseek-chat",
        input_tokens=1234,
        output_tokens=321,
    )

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=one_page(text),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    assert report.documents[0].error is None
    assert len(report.model_calls) == 1
    call = report.model_calls[0]
    assert call.document_id == doc_id
    assert call.purpose == "extraction"
    assert (call.model, call.input_tokens, call.output_tokens) == (
        "deepseek-chat",
        1234,
        321,
    )
    assert call.elapsed_seconds >= 0.0


def test_report_records_enough_to_reproduce_the_model_and_parameter_combination(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    text = "外圈剥落会在振动频谱上留下明显的特征频率。"
    doc_id = content_doc_id(content)
    parameters: dict[str, str | int | float | bool] = {
        "temperature": 0.0,
        "response_format": "json_object",
    }
    llm = InstrumentedFakeLLM(
        extraction_response(evidence_id(doc_id, 1, text)),
        model="deepseek-chat",
        parameters=parameters,
    )

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=one_page(text),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    run = report.run
    assert run is not None
    assert run.source_dir == tmp_path
    assert run.limit is None
    assert run.model == ModelDescriptor(model="deepseek-chat", parameters=parameters)
    assert run.ontology_version
    assert len(run.ontology_sha256) == 64
    assert run.started_at.tzinfo is not None
    assert run.started_at <= run.finished_at


def test_run_metadata_records_the_limit_that_was_applied(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 first")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 second")
    text = "外圈剥落会在振动频谱上留下明显的特征频率。"
    doc_id = content_doc_id(b"%PDF-1.4 first")
    llm = InstrumentedFakeLLM(extraction_response(evidence_id(doc_id, 1, text)))

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=one_page(text),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
        limit=1,
    )

    assert report.run is not None
    assert report.run.limit == 1
    assert [document.path.name for document in report.documents] == ["a.pdf"]


def test_document_without_text_layer_is_quarantined_instead_of_failed(
    tmp_path: Path,
) -> None:
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 scanned paper")
    llm = InstrumentedFakeLLM("{}")

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=one_page("   "),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert document.status == "quarantined"
    assert document.quarantine_reason is not None
    assert "文字层" in document.quarantine_reason
    assert document.error is None
    assert document.evidence_segments == 0
    assert (report.succeeded, report.failed, report.quarantined) == (0, 0, 1)
    # 隔离的文档不进入抽取，因此不产生模型调用成本。
    assert report.model_calls == ()


def test_ontology_rejected_extractions_become_pending_review_items(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 fake paper with out-of-ontology extractions"
    (tmp_path / "mixed.pdf").write_bytes(content)
    text = "外圈剥落与神秘实体并存，外圈故障频率也会出现。"
    segment_id = evidence_id(content_doc_id(content), 1, text)
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
                    "name": "神秘实体",
                    "type": "alien_type",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                },
                {
                    "name": "神秘实体",
                    "type": "alien_type",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "magic_link",
                    "tail": "外圈故障频率",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=InstrumentedFakeLLM(response),
            parser=one_page(text),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert document.error is None
    assert document.entities == 1
    assert (document.rejected_entities, document.rejected_relations) == (2, 1)

    entity_item, relation_item = document.pending_review
    assert (
        entity_item.kind,
        entity_item.label,
        entity_item.type,
        entity_item.reason,
        entity_item.occurrences,
    ) == ("entity", "神秘实体", "alien_type", "type_not_in_ontology", 2)
    assert entity_item.evidence_ids == (segment_id,)
    assert (
        relation_item.kind,
        relation_item.label,
        relation_item.type,
        relation_item.reason,
    ) == ("relation", "外圈剥落 → 外圈故障频率", "magic_link", "type_not_in_ontology")
    assert report.pending_review == 2

"""模型输出不合规时的有界重试：记录原因，但不中断整批。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest

from fakes import (
    FakeEmbedder,
    FakeGraphStore,
    FakeLLM,
    FakeParser,
    FakeVectorStore,
    InstrumentedFakeLLM,
)

TEXT = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
BAD_RESPONSE = "这不是 JSON"


def valid_response(doc_id: str) -> str:
    segment_id = f"ev:{doc_id}:1:0-{len(TEXT)}"
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


def deps_for(llm: FakeLLM, graph_store: FakeGraphStore) -> IngestDeps:
    return IngestDeps(
        llm=llm,
        parser=FakeParser(
            [ParsedPage(number=1, text=TEXT, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )


def test_failed_extraction_is_retried_a_bounded_number_of_times_and_recorded(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 malformed extraction")
    good_content = b"%PDF-1.4 good extraction"
    (tmp_path / "b.pdf").write_bytes(good_content)
    good_doc_id = hashlib.sha256(good_content).hexdigest()[:16]
    llm = FakeLLM([BAD_RESPONSE, BAD_RESPONSE, BAD_RESPONSE, valid_response(good_doc_id)])
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    failed, good = report.documents
    assert report.failed == 1
    assert failed.path.name == "a.pdf"
    assert failed.error is not None
    assert "ExtractionError" in failed.error
    assert "3 次" in failed.error
    assert failed.extraction_attempts == 3

    # 一篇失败不影响同批的下一篇。
    assert good.error is None
    assert good.extraction_attempts == 1
    assert good.entities == 1
    assert len(graph_store.batches) == 1
    assert len(llm.prompts) == 4
    assert "上一次输出无法解析" in llm.prompts[1]


def test_a_retry_that_succeeds_keeps_the_document(tmp_path: Path) -> None:
    content = b"%PDF-1.4 retry succeeds"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    llm = FakeLLM([BAD_RESPONSE, valid_response(doc_id)])
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    document = report.documents[0]
    assert document.error is None
    assert document.extraction_attempts == 2
    assert document.entities == 1
    assert len(llm.prompts) == 2
    assert len(graph_store.batches) == 1


def test_every_retry_is_recorded_as_its_own_model_call(tmp_path: Path) -> None:
    content = b"%PDF-1.4 always malformed"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    llm = InstrumentedFakeLLM(BAD_RESPONSE, model="fake-model")
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    document = report.documents[0]
    assert document.error is not None
    assert document.extraction_attempts == 3
    assert len(report.model_calls) == 3
    assert {call.document_id for call in report.model_calls} == {doc_id}
    assert {call.model for call in report.model_calls} == {"fake-model"}

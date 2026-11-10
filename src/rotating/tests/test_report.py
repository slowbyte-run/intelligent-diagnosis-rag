"""批次报告文件：一次运行一份、可留存、字段足以核对与复现。

报告落在语料表中的 ``data/output/reports/``（AGENTS.md），由命令行传入具体目录。
测试只通过文件名与 JSON 内容观察报告，不碰内部序列化函数。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from idr_core.llm import ModelDescriptor
from idr_rotating import (
    DocumentReport,
    DocumentType,
    IngestDeps,
    IngestReport,
    ModelCallRecord,
    PendingExample,
    PendingKind,
    PendingObservation,
    RejectionReason,
    RunMetadata,
    ingest,
    write_report,
)

from fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeParser, FakeVectorStore

STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def sample_report(source_dir: Path) -> IngestReport:
    """一份含成功、失败、隔离与待审计四类结果的运行报告。"""
    pending = PendingObservation(
        kind=PendingKind.ENTITY,
        value="alien_type",
        reason=RejectionReason.TYPE_NOT_IN_ONTOLOGY,
        doc_id="doc1",
        name="神秘实体",
        examples=(
            PendingExample(
                doc_id="doc1",
                evidence_id="ev:doc1:1:0-12",
                text="神秘实体在文献中出现。",
            ),
        ),
    )
    return IngestReport(
        documents=(
            DocumentReport(
                path=source_dir / "paper.pdf",
                doc_id="doc1",
                evidence_segments=3,
                entities=5,
                relations=2,
                rejected_entities=2,
                document_type=DocumentType.PAPER,
                pending=(pending, pending),
            ),
            DocumentReport(
                path=source_dir / "broken.pdf",
                doc_id="doc2",
                evidence_segments=0,
                entities=0,
                relations=0,
                error="ExtractionError: 模型输出不符合抽取契约",
            ),
            DocumentReport(
                path=source_dir / "scan.pdf",
                doc_id="doc3",
                evidence_segments=0,
                entities=0,
                relations=0,
                quarantine_reason="扫描件：未检测到可用的文字层",
            ),
            DocumentReport(
                path=source_dir / "done.pdf",
                doc_id="doc4",
                evidence_segments=0,
                entities=0,
                relations=0,
                skipped=True,
            ),
        ),
        model_calls=(
            ModelCallRecord(
                document_id="doc1",
                purpose="extraction",
                model="deepseek-chat",
                input_tokens=1000,
                output_tokens=250,
                elapsed_seconds=1.23456,
            ),
        ),
        run=RunMetadata(
            source_dir=source_dir,
            limit=None,
            started_at=STARTED_AT,
            finished_at=STARTED_AT,
            model=ModelDescriptor(
                model="deepseek-chat",
                parameters={"temperature": 0.0, "response_format": "json_object"},
            ),
            ontology_version="0.1",
            ontology_sha256="a" * 64,
            prompt_sha256="b" * 64,
        ),
    )


def test_report_file_has_counts_and_details_for_all_four_categories(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    report = sample_report(source_dir)

    path = write_report(report, output_dir=tmp_path / "reports")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run"]["id"] == path.stem.removeprefix("ingest-")
    assert payload["run"]["source_dir"] == str(source_dir)
    assert payload["run"]["model"] == {
        "model": "deepseek-chat",
        "parameters": {"temperature": 0.0, "response_format": "json_object"},
    }
    assert payload["run"]["ontology"] == {"version": "0.1", "sha256": "a" * 64}
    assert payload["run"]["prompt"] == {"sha256": "b" * 64}

    summary = payload["summary"]
    assert summary["documents"] == {
        "total": 4,
        "succeeded": 1,
        "failed": 1,
        "quarantined": 1,
        "skipped": 1,
    }
    assert summary["pending_review"] == {"items": 1, "occurrences": 2}
    assert (
        summary["evidence_segments"],
        summary["entities"],
        summary["relations"],
    ) == (3, 5, 2)
    assert summary["model_calls"] == {
        "count": 1,
        "usage_reported": 1,
        "input_tokens": 1000,
        "output_tokens": 250,
        "elapsed_seconds": 1.2346,
    }

    documents = payload["documents"]
    assert [document["status"] for document in documents] == [
        "succeeded",
        "failed",
        "quarantined",
        "skipped",
    ]
    assert [document["file_name"] for document in documents] == [
        "paper.pdf",
        "broken.pdf",
        "scan.pdf",
        "done.pdf",
    ]
    assert documents[0]["document_type"] == "paper"
    assert (documents[0]["evidence_segments"], documents[0]["entities"]) == (3, 5)
    assert documents[1]["error"] == "ExtractionError: 模型输出不符合抽取契约"
    assert documents[2]["quarantine_reason"] == "扫描件：未检测到可用的文字层"

    assert payload["pending_review"] == [
        {
            "document_id": "doc1",
            "file_name": "paper.pdf",
            "kind": "entity",
            "label": "神秘实体",
            "type": "alien_type",
            "reason": "type_not_in_ontology",
            "occurrences": 2,
            "evidence_ids": ["ev:doc1:1:0-12"],
            "examples": [
                {
                    "document_id": "doc1",
                    "evidence_id": "ev:doc1:1:0-12",
                    "text": "神秘实体在文献中出现。",
                }
            ],
        }
    ]
    assert payload["model_calls"] == [
        {
            "document_id": "doc1",
            "purpose": "extraction",
            "model": "deepseek-chat",
            "input_tokens": 1000,
            "output_tokens": 250,
            "elapsed_seconds": 1.2346,
        }
    ]


def test_report_file_names_distinguish_runs(tmp_path: Path) -> None:
    report = sample_report(tmp_path)

    first = write_report(report, output_dir=tmp_path / "reports")
    second = write_report(report, output_dir=tmp_path / "reports")

    assert first != second
    assert first.parent == second.parent == tmp_path / "reports"
    assert first.name.startswith("ingest-20260102T030405Z-")
    assert first.suffix == second.suffix == ".json"
    assert random_suffix(first.name) != random_suffix(second.name)


def random_suffix(name: str) -> str:
    """文件名里区分同一次运行的随机后缀。"""
    return name.removesuffix(".json").rsplit("-", maxsplit=1)[-1]


def test_empty_batch_still_writes_a_report_with_zero_counts(tmp_path: Path) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    report = ingest(
        source_dir,
        deps=IngestDeps(
            llm=FakeLLM("{}"),
            parser=FakeParser([]),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    path = write_report(report, output_dir=tmp_path / "reports")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["documents"] == {
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "quarantined": 0,
        "skipped": 0,
    }
    assert payload["summary"]["pending_review"] == {"items": 0, "occurrences": 0}
    empty = []
    assert payload["documents"] == payload["pending_review"] == empty
    assert payload["model_calls"] == empty

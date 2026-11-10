"""ingest 子命令的计数输出：用假实现驱动，不连服务也不掉模型。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.config import Config
from idr_core.embedding import EMBEDDING_DIM
from idr_core.graph import GraphBatch
from idr_core.llm.embedding import OpenAICompatibleEmbedder
from idr_core.models import BBox
from idr_core.parsing import ParsedDocument, ParsedPage
from idr_core.vector_store.pgvector import PgVectorStore
from idr_rotating import IngestDeps

from idr_cli.ingest import build_deps, run_ingest
from idr_cli.main import main


class FakeParser:
    def __init__(self, pages: Sequence[ParsedPage]) -> None:
        self._pages = tuple(pages)

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(pages=self._pages)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response


class FakeGraphStore:
    def __init__(self) -> None:
        self.batches: list[GraphBatch] = []

    def write_document(self, batch: GraphBatch) -> None:
        self.batches.append(batch)


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIM for _ in texts]


class FakeVectorStore:
    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        pass

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None:
        return None


def test_ingest_prints_per_document_and_total_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落表现为 BPFO 倍频。"
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
                    "name": "BPFO 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
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
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    exit_code = run_ingest(tmp_path, deps=deps, reports_dir=tmp_path / "reports")

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paper.pdf：证据段 1，实体 2，关系 1" in captured.out
    assert "合计：证据段 1，实体 2，关系 1" in captured.out


def test_main_dispatches_ingest_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落。"
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
            "relations": [],
        },
        ensure_ascii=False,
    )
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    for name, value in {
        "DEEPSEEK_BASE_URL": "https://example.invalid",
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_MODEL": "test-model",
        "SILICONFLOW_BASE_URL": "https://example.invalid",
        "SILICONFLOW_API_KEY": "sk-test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIM": "1024",
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "DATABASE_URL": "postgresql://idr:secret@127.0.0.1:1/idr",
    }.items():
        monkeypatch.setenv(name, value)

    exit_code = main(
        ["ingest", str(tmp_path), "--reports-dir", str(tmp_path / "reports")],
        deps_factory=lambda config: deps,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paper.pdf：证据段 1，实体 1，关系 0" in captured.out


def test_build_deps_wires_the_embedder_and_vector_store() -> None:
    config = Config.model_validate(
        {
            "DEEPSEEK_BASE_URL": "https://example.invalid",
            "DEEPSEEK_API_KEY": "sk-test",
            "DEEPSEEK_MODEL": "test-model",
            "SILICONFLOW_BASE_URL": "https://example.invalid/v1",
            "SILICONFLOW_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "test-embedding",
            "EMBEDDING_DIM": "1024",
            "NEO4J_URI": "bolt://127.0.0.1:1",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
            "DATABASE_URL": "postgresql://idr:secret@127.0.0.1:1/idr",
        }
    )

    deps = build_deps(config)

    assert isinstance(deps.embedder, OpenAICompatibleEmbedder)
    assert isinstance(deps.vector_store, PgVectorStore)


def test_ingest_writes_a_batch_report_for_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落表现为 BPFO 倍频。"
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
                    "name": "BPFO 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
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
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    reports_dir = tmp_path / "reports"

    exit_code = run_ingest(tmp_path, deps=deps, reports_dir=reports_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    (report_file,) = list(reports_dir.glob("ingest-*.json"))
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["summary"]["documents"] == {
        "total": 1,
        "succeeded": 1,
        "failed": 0,
        "quarantined": 0,
    }
    assert (
        payload["documents"][0]["evidence_segments"],
        payload["documents"][0]["entities"],
        payload["documents"][0]["relations"],
    ) == (1, 2, 1)
    assert str(report_file) in captured.out


def test_empty_input_directory_still_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_dir = tmp_path / "input"
    empty_dir.mkdir()
    deps = IngestDeps(
        llm=FakeLLM("{}"),
        parser=FakeParser([]),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    reports_dir = tmp_path / "reports"

    exit_code = run_ingest(empty_dir, deps=deps, reports_dir=reports_dir)

    assert exit_code == 0
    (report_file,) = list(reports_dir.glob("ingest-*.json"))
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["summary"]["documents"]["total"] == 0
    assert payload["documents"] == []
    assert not capsys.readouterr().err

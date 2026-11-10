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
from idr_core.state import DocumentProcessingState
from idr_core.state.postgres import PostgresStateStore
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
    text = "外圈剥落表现为 BPFO 倍频，能量集中在高频段。"
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

    exit_code = run_ingest(tmp_path, deps=deps)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paper.pdf：证据段 1，实体 2，关系 1" in captured.out
    assert "合计：证据段 1，实体 2，关系 1" in captured.out


def test_ingest_reports_quarantined_scanned_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 scanned pages")
    deps = IngestDeps(
        llm=FakeLLM('{"entities": [], "relations": []}'),
        parser=FakeParser(
            [ParsedPage(number=1, text="", bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    exit_code = run_ingest(tmp_path, deps=deps)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "scan.pdf：已隔离" in captured.out
    assert "扫描件" in captured.out


def test_main_dispatches_ingest_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落会表现为 BPFO 倍频振动，且能量集中在高频段。"
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

    exit_code = main(["ingest", str(tmp_path)], deps_factory=lambda config: deps)

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
    assert isinstance(deps.state_store, PostgresStateStore)


class FakeStateStore:
    """内存版处理状态，供 CLI 行为测试使用。"""

    def __init__(self) -> None:
        self.states: dict[str, DocumentProcessingState] = {}

    def get(self, doc_id: str) -> DocumentProcessingState | None:
        return self.states.get(doc_id)

    def record(self, state: DocumentProcessingState) -> None:
        self.states[state.doc_id] = state


def make_stateful_deps(state_store: FakeStateStore) -> IngestDeps:
    return IngestDeps(
        llm=FakeLLM('{"entities": [], "relations": []}'),
        parser=FakeParser(
            [
                ParsedPage(
                    number=1,
                    text="外圈剥落表现为 BPFO 倍频，能量集中在高频段。",
                    bbox=BBox(0.0, 0.0, 595.0, 842.0),
                )
            ]
        ),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        state_store=state_store,
    )


def test_repeated_run_reports_skipped_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 b")
    deps = make_stateful_deps(FakeStateStore())

    assert run_ingest(tmp_path, deps=deps) == 0
    capsys.readouterr()
    assert run_ingest(tmp_path, deps=deps) == 0

    captured = capsys.readouterr()
    assert "a.pdf：已跳过（此前已入库）" in captured.out
    assert "b.pdf：已跳过（此前已入库）" in captured.out
    assert "合计：证据段 0，实体 0，关系 0，隔离 0，跳过 2" in captured.out


def test_main_limit_processes_only_the_first_documents(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 b")
    deps = make_stateful_deps(FakeStateStore())
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
        ["ingest", str(tmp_path), "--limit", "1"],
        deps_factory=lambda config: deps,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "a.pdf：" in captured.out
    assert "b.pdf" not in captured.out


def test_limit_must_be_a_positive_integer(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["ingest", str(tmp_path), "--limit", "0"])

    assert exit_info.value.code == 2

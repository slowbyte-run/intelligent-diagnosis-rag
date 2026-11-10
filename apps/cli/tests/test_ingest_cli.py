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

    exit_code = run_ingest(tmp_path, deps=deps, reports_dir=tmp_path / "reports")

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

    exit_code = run_ingest(tmp_path, deps=deps, reports_dir=tmp_path / "reports")

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


def test_ingest_writes_a_batch_report_for_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = b"%PDF-1.4 fake paper"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外圈剥落表现为 BPFO 倍频，且能量集中在高频段。"
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
        "skipped": 0,
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
    assert payload["summary"]["documents"] == {
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "quarantined": 0,
        "skipped": 0,
    }
    assert payload["documents"] == []
    assert not capsys.readouterr().err


def test_repeated_run_reports_skipped_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 b")
    deps = make_stateful_deps(FakeStateStore())
    reports_dir = tmp_path / "reports"

    assert run_ingest(tmp_path, deps=deps, reports_dir=reports_dir) == 0
    capsys.readouterr()
    assert run_ingest(tmp_path, deps=deps, reports_dir=reports_dir) == 0

    captured = capsys.readouterr()
    assert "a.pdf：已跳过（此前已入库）" in captured.out
    assert "b.pdf：已跳过（此前已入库）" in captured.out
    assert "分类：成功 0，失败 0，隔离 0，跳过 2，待审计 0" in captured.out
    assert "合计：证据段 0，实体 0，关系 0" in captured.out

    # 跳过也是报告里单独计数的一类，不能混进成功或失败。
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in reports_dir.glob("ingest-*.json")
    ]
    assert any(
        payload["summary"]["documents"]["skipped"] == 2 for payload in reports
    )


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
        [
            "ingest",
            str(tmp_path),
            "--limit",
            "1",
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
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


def test_ontology_error_is_reported_to_the_user(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake paper")
    broken = tmp_path / "broken-ontology.yaml"
    broken.write_text(
        "version: 0.1.0\ndomain: rotating\n"
        "entity_types:\n  fault_mode:\n    description: 缺少 label\n"
        "relation_types: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "idr_rotating.ontology.DEFAULT_ONTOLOGY_PATH", broken
    )
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
    assert exit_code == 1
    assert "本体" in captured.err
    assert "Traceback" not in captured.err


def test_ingest_prints_pending_reviews_with_frequency_and_examples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = b"%PDF-1.4 pending review output"
    (tmp_path / "pending.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "未知甲属于神秘类型，且与未知乙存在玄妙联系；外圈剥落则不同。"
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
                    "tail": "未知甲",
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

    exit_code = run_ingest(
        tmp_path, deps=deps, reports_dir=tmp_path / "reports"
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "待审队列" in captured.out
    assert "alien_type" in captured.out
    assert "magic_link" in captured.out
    assert "出现 1 次" in captured.out
    assert text in captured.out

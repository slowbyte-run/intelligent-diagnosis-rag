"""eval 子命令：读金标准集、调用 ask、写三项指标报告。

这里用一张合成的一页 PDF 与内存假图存储跑完整链路，不依赖 Neo4j、网络或
真实语料；真实 PDF 的逐字校验由 ``src/rotating/tests/test_citation_check.py``
用语料里的 PDF 覆盖。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.documents import document_id
from idr_core.graph import Concept, EvidenceRecord, FactEdge
from idr_core.parsing.pdf import page_text
from idr_rotating import AskDeps

from idr_cli.eval import run_eval
from idr_cli.main import main

OUTER_RACE = "concept:bearing_outer_race_fatigue_spalling"
BPFO = "concept:bpfo_harmonics"
PDF_TEXT = "Bearing outer race defect appears at BPFO."


def one_page_pdf(text: str) -> bytes:
    """生成一页、一段文字的合法最小 PDF（与核心包测试里的构造方式一致）。"""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_offset = len(document)
    document += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    document += b"0000000000 65535 f \n"
    for offset in offsets:
        document += f"{offset:010d} 00000 n \n".encode("ascii")
    document += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(document)


class FakeGraphReader:
    def __init__(
        self,
        *,
        doc_id: str,
        evidence_id: str,
    ) -> None:
        self._doc_id = doc_id
        self._evidence_id = evidence_id
        self._concepts = {
            OUTER_RACE: Concept(id=OUTER_RACE, name="轴承外圈疲劳剥落", type="fault_mode"),
            BPFO: Concept(id=BPFO, name="外圈故障频率及其倍频", type="symptom"),
        }
        self._evidence = EvidenceRecord(
            id=evidence_id, doc_id=doc_id, page=1, text=PDF_TEXT
        )

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        if concept_id != OUTER_RACE or "indicated_by" not in relation_types:
            return ()
        return (
            FactEdge(
                head_id=OUTER_RACE,
                relation="indicated_by",
                tail_id=BPFO,
                evidence_ids=(self._evidence_id,),
                confidence=0.9,
                doc_id=self._doc_id,
            ),
        )

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        return tuple(
            self._concepts[concept_id]
            for concept_id in concept_ids
            if concept_id in self._concepts
        )

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        return tuple(
            self._evidence for evidence_id in evidence_ids if evidence_id == self._evidence.id
        )


def corpus_with_pdf(directory: Path) -> tuple[Path, str, str, str]:
    """写一张合成 PDF 并返回 (文件, doc_id, evidence_id, 页面文字)。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "paper.pdf"
    content = one_page_pdf(PDF_TEXT)
    path.write_bytes(content)
    doc_id = document_id(content)
    text = page_text(path, 1)
    evidence_id = f"ev:{doc_id}:1:0-{len(text)}"
    return path, doc_id, evidence_id, text


def write_golden(path: Path, evidence_id: str) -> Path:
    payload = {
        "schema_version": 1,
        "questions": [
            {
                "id": "T-01",
                "class": "factual",
                "question": "轴承外圈疲劳剥落有什么征兆？",
                "expected_concepts": [OUTER_RACE, BPFO],
                "expected_evidence_ids": [evidence_id],
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_run_eval_reports_three_metrics_and_writes_an_archive_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_dir = tmp_path / "corpus"
    _, doc_id, evidence_id, text = corpus_with_pdf(corpus_dir)
    assert text == PDF_TEXT
    golden = write_golden(tmp_path / "golden.json", evidence_id)
    report_path = tmp_path / "reports" / "eval.json"

    exit_code = run_eval(
        deps=AskDeps(graph_store=FakeGraphReader(doc_id=doc_id, evidence_id=evidence_id)),
        corpus_dir=corpus_dir,
        golden_path=golden,
        out_path=report_path,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索命中率" in output
    assert "引用准确率" in output
    assert "拒答正确率" in output
    payload = json.loads(report_path.read_text("utf-8"))
    assert payload["metrics"]["retrieval_hit_rate"] == 1.0
    assert payload["metrics"]["citation_accuracy"] == 1.0
    assert payload["metrics"]["refusal_correctness"] == 1.0
    assert payload["cases"][0]["id"] == "T-01"


def test_run_eval_fails_cleanly_when_the_corpus_directory_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json", "ev:0000000000000000:1:0-1")

    exit_code = run_eval(
        deps=AskDeps(graph_store=FakeGraphReader(doc_id="x", evidence_id="y")),
        corpus_dir=tmp_path / "missing",
        golden_path=golden,
        out_path=tmp_path / "report.json",
    )

    assert exit_code == 1
    assert "语料目录" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_main_dispatches_eval_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    corpus_dir = tmp_path / "corpus"
    _, doc_id, evidence_id, _ = corpus_with_pdf(corpus_dir)
    golden = write_golden(tmp_path / "golden.json", evidence_id)
    report_path = tmp_path / "eval-report.json"

    exit_code = main(
        [
            "eval",
            "--golden",
            str(golden),
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(report_path),
        ],
        ask_deps_factory=lambda config, source_dir: AskDeps(
            graph_store=FakeGraphReader(doc_id=doc_id, evidence_id=evidence_id)
        ),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索命中率" in output
    assert report_path.exists()

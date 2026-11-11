"""ask 子命令：人类可读输出、JSON 形态与文献名解析。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.documents import DocumentNames, document_id
from idr_core.graph import Concept, EvidenceRecord, FactEdge
from idr_rotating import AskDeps

from idr_cli.ask import CorpusDocumentNames, run_ask
from idr_cli.main import main

DOC_ID = "doc-a"
OUTER_RACE = "concept:bearing_outer_race_fatigue_spalling"
BPFO = "concept:bpfo_harmonics"
EVIDENCE_ID = "ev:doc-a:4:0-26"


class FakeGraphReader:
    """按标识返回预置的读模型；CLI 测试不在图谱语义上重复造实现。"""

    def __init__(
        self,
        *,
        facts: Sequence[FactEdge] = (),
        concepts: Sequence[Concept] = (),
        evidence: Sequence[EvidenceRecord] = (),
    ) -> None:
        self._facts = tuple(facts)
        self._concepts = {concept.id: concept for concept in concepts}
        self._evidence = {segment.id: segment for segment in evidence}

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        return tuple(
            fact
            for fact in self._facts
            if fact.head_id == concept_id and fact.relation in relation_types
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
            self._evidence[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in self._evidence
        )


class FakeDocumentNames:
    def __init__(self, names: dict[str, str]) -> None:
        self._names = dict(names)

    def name_for(self, doc_id: str) -> str | None:
        return self._names.get(doc_id)


def deps_with_one_fact(
    *, document_names: DocumentNames | None = None
) -> AskDeps:
    outer = Concept(id=OUTER_RACE, name="轴承外圈疲劳剥落", type="fault_mode")
    bpfo = Concept(id=BPFO, name="外圈故障频率及其倍频", type="symptom")
    segment = EvidenceRecord(
        id=EVIDENCE_ID,
        doc_id=DOC_ID,
        page=4,
        text="外圈剥落会在频谱上出现 BPFO 及其倍频。",
    )
    graph = FakeGraphReader(
        facts=(
            FactEdge(
                head_id=OUTER_RACE,
                relation="indicated_by",
                tail_id=BPFO,
                evidence_ids=(EVIDENCE_ID,),
                confidence=0.8,
                doc_id=DOC_ID,
            ),
        ),
        concepts=(outer, bpfo),
        evidence=(segment,),
    )
    return AskDeps(
        graph_store=graph,
        document_names=(
            FakeDocumentNames({DOC_ID: "论文A.pdf"})
            if document_names is None
            else document_names
        ),
    )


def test_ask_prints_conclusions_with_document_and_page(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask("轴承外圈疲劳剥落有什么征兆", deps=deps_with_one_fact())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索路径：图命中" in output
    assert "命中概念：轴承外圈疲劳剥落（故障模式" in output
    assert "—表现为→ 外圈故障频率及其倍频（置信度 0.80）" in output
    assert "依据：论文A.pdf 第 4 页：外圈剥落会在频谱上出现 BPFO 及其倍频。" in output


def test_ask_prints_machine_readable_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask(
        "轴承外圈疲劳剥落有什么征兆", deps=deps_with_one_fact(), as_json=True
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == 1
    assert payload["question"] == "轴承外圈疲劳剥落有什么征兆"
    assert payload["path"] == "graph"
    assert payload["detail"] is None
    assert payload["focus"]["concept"]["id"] == OUTER_RACE
    assert payload["focus"]["alias"] == "轴承外圈疲劳剥落"
    assert payload["focus"]["kind"] == "substring"
    assert payload["facts"] == [
        {
            "relation": "indicated_by",
            "head": {
                "id": OUTER_RACE,
                "name": "轴承外圈疲劳剥落",
                "type": "fault_mode",
            },
            "tail": {
                "id": BPFO,
                "name": "外圈故障频率及其倍频",
                "type": "symptom",
            },
            "confidence": 0.8,
            "evidence_ids": [EVIDENCE_ID],
        }
    ]
    assert payload["citations"] == [
        {
            "evidence_id": EVIDENCE_ID,
            "doc_id": DOC_ID,
            "document": "论文A.pdf",
            "page": 4,
            "text": "外圈剥落会在频谱上出现 BPFO 及其倍频。",
        }
    ]


def test_ask_prints_the_reason_for_an_empty_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask(
        "今天天气怎么样",
        deps=AskDeps(graph_store=FakeGraphReader()),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索路径：拒答" in output
    assert "没有可回答的依据" in output
    assert "匹配" in output


def test_corpus_document_names_maps_content_hash_to_file_name(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 fake corpus file"
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)

    names = CorpusDocumentNames(tmp_path)

    assert names.name_for(document_id(content)) == "paper.pdf"
    assert names.name_for("not-a-real-id") is None


def test_corpus_document_names_is_empty_for_a_missing_directory(
    tmp_path: Path,
) -> None:
    assert CorpusDocumentNames(tmp_path / "missing").name_for(DOC_ID) is None


def test_main_dispatches_ask_subcommand(
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
    received: list[Path] = []

    def ask_deps_factory(config: object, source_dir: Path) -> AskDeps:
        received.append(source_dir)
        return deps_with_one_fact()

    exit_code = main(
        ["ask", "轴承外圈疲劳剥落有什么征兆", "--corpus-dir", str(tmp_path)],
        ask_deps_factory=ask_deps_factory,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert received == [tmp_path]
    assert "检索路径：图命中" in output

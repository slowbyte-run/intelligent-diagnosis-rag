"""ask 子命令：人类可读输出、JSON 形态与文献名解析。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.documents import DocumentNames, document_id
from idr_core.graph import Concept, EvidenceRecord, FactEdge
from idr_core.llm.rerank import RerankResponse, RerankScore
from idr_core.vector_store import VectorMatch
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
    assert payload["schema_version"] == 4
    assert payload["question"] == "轴承外圈疲劳剥落有什么征兆"
    assert payload["path"] == "graph"
    assert payload["detail"] is None
    assert payload["truncated"] is False
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
            "path": [
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
            ],
        }
    ]
    assert payload["citations"] == [
        {
            "evidence_id": EVIDENCE_ID,
            "doc_id": DOC_ID,
            "document": "论文A.pdf",
            "page": 4,
            "text": "外圈剥落会在频谱上出现 BPFO 及其倍频。",
            "score": None,
        }
    ]
    assert payload["model_calls"] == []


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
    assert "拒答原因：问题里的概念没有识别出来（no_concept）" in output
    assert "没有可回答的依据" in output
    assert "匹配" in output


def test_ask_prints_no_evidence_when_a_concept_has_no_basis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask(
        "轴承外圈疲劳剥落有什么征兆",
        deps=AskDeps(graph_store=FakeGraphReader()),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索路径：拒答" in output
    assert "拒答原因：概念已识别，但没有找到可用的证据（no_evidence）" in output
    assert "不在图谱中" in output


def test_ask_puts_the_refusal_reason_in_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask(
        "今天天气怎么样",
        deps=AskDeps(graph_store=FakeGraphReader()),
        as_json=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == 4
    assert payload["path"] == "refusal"
    assert payload["refusal_reason"] == "no_concept"
    assert payload["facts"] == []
    assert payload["citations"] == []


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


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeVectorReader:
    def __init__(self, evidence_ids: Sequence[str]) -> None:
        self._evidence_ids = tuple(evidence_ids)

    def search_similar(
        self, vector: Sequence[float], *, limit: int
    ) -> tuple[VectorMatch, ...]:
        return tuple(
            VectorMatch(evidence_id=evidence_id, distance=0.1)
            for evidence_id in self._evidence_ids[:limit]
        )


class FakeReranker:
    def __init__(self, score: float = 0.83) -> None:
        self._score = score

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        return RerankResponse(
            scores=tuple(
                RerankScore(index=index, score=self._score)
                for index, _ in enumerate(documents)
            ),
            model="fake-rerank",
            input_tokens=9,
            output_tokens=0,
        )


def deps_with_vector_only_basis() -> AskDeps:
    segment = EvidenceRecord(
        id=EVIDENCE_ID,
        doc_id=DOC_ID,
        page=4,
        text="周期性冲击通常来自滚动体缺陷，需要用包络解调确认。",
    )
    return AskDeps(
        graph_store=FakeGraphReader(evidence=(segment,)),
        document_names=FakeDocumentNames({DOC_ID: "论文A.pdf"}),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorReader([EVIDENCE_ID]),
        reranker=FakeReranker(),
    )


def test_ask_prints_reranked_evidence_for_the_vector_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask(
        "振动信号里出现周期性冲击应该检查什么",
        deps=deps_with_vector_only_basis(),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索路径：向量兜底" in output
    assert "文献证据（图路径未命中，按重排相关度排序）：" in output
    assert (
        "1. 论文A.pdf 第 4 页（相关度 0.83）："
        "周期性冲击通常来自滚动体缺陷，需要用包络解调确认。" in output
    )


def test_build_ask_deps_wires_graph_vectors_embedder_and_reranker(
    tmp_path: Path,
) -> None:
    """真实依赖必须一次接齐：少接一件，向量兜底就永远不会发生。"""
    from idr_core.config import load_config
    from idr_core.llm.embedding import OpenAICompatibleEmbedder
    from idr_core.llm.rerank import OpenAICompatibleReranker
    from idr_core.vector_store.pgvector import PgVectorStore

    from idr_cli.ask import build_ask_deps

    config = load_config(
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

    deps = build_ask_deps(config, tmp_path)

    assert isinstance(deps.embedder, OpenAICompatibleEmbedder)
    assert isinstance(deps.vector_store, PgVectorStore)
    assert isinstance(deps.reranker, OpenAICompatibleReranker)
    assert deps.document_names is not None
    assert deps.document_names.name_for("not-a-real-id") is None


def test_ask_reports_a_broken_retrieval_configuration(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import idr_cli.ask as ask_module
    from idr_rotating import RetrievalSettingsError

    def blow_up(question: str, *, deps: AskDeps) -> object:
        raise RetrievalSettingsError("检索配置读取失败：missing.yaml")

    monkeypatch.setattr(ask_module, "ask", blow_up)

    exit_code = run_ask("今天天气怎么样", deps=deps_with_one_fact())

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "检索配置错误" in captured.err
    assert "missing.yaml" in captured.err


def deps_with_two_hop_chain() -> AskDeps:
    """滚动轴承 → 轴承外圈疲劳剥落 → 外圈故障频率及其倍频。"""
    bearing = Concept(id="concept:rolling_bearing", name="滚动轴承", type="component")
    outer = Concept(id=OUTER_RACE, name="轴承外圈疲劳剥落", type="fault_mode")
    bpfo = Concept(id=BPFO, name="外圈故障频率及其倍频", type="symptom")
    fault_evidence = EvidenceRecord(
        id="ev:doc-a:3:0-15",
        doc_id=DOC_ID,
        page=3,
        text="滚动轴承常见外圈疲劳剥落。",
    )
    symptom_evidence = EvidenceRecord(
        id=EVIDENCE_ID,
        doc_id=DOC_ID,
        page=4,
        text="外圈剥落会在频谱上出现 BPFO 及其倍频。",
    )
    graph = FakeGraphReader(
        facts=(
            FactEdge(
                head_id=bearing.id,
                relation="has_fault_mode",
                tail_id=outer.id,
                evidence_ids=(fault_evidence.id,),
                confidence=0.7,
                doc_id=DOC_ID,
            ),
            FactEdge(
                head_id=outer.id,
                relation="indicated_by",
                tail_id=bpfo.id,
                evidence_ids=(symptom_evidence.id,),
                confidence=0.6,
                doc_id=DOC_ID,
            ),
        ),
        concepts=(bearing, outer, bpfo),
        evidence=(fault_evidence, symptom_evidence),
    )
    return AskDeps(
        graph_store=graph,
        document_names=FakeDocumentNames({DOC_ID: "论文A.pdf"}),
    )


def test_ask_prints_the_full_two_hop_route_with_evidence_for_each_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask("滚动轴承有什么征兆", deps=deps_with_two_hop_chain())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "滚动轴承 —具有故障模式→ 轴承外圈疲劳剥落 "
        "—表现为→ 外圈故障频率及其倍频（置信度 0.60）"
    ) in output
    assert "依据（滚动轴承 → 轴承外圈疲劳剥落）：论文A.pdf 第 3 页：" in output
    assert (
        "依据（轴承外圈疲劳剥落 → 外圈故障频率及其倍频）：论文A.pdf 第 4 页："
    ) in output


INNER_RACE = "concept:bearing_inner_race_fatigue_spalling"
BPFI = "concept:bpfi_harmonics"
SPECTRUM = "concept:fft_spectrum_analysis"
OUTER_SYMPTOM_ID = "ev:doc-b:2:0-18"
INNER_SYMPTOM_ID = "ev:doc-b:2:30-48"
OUTER_DETECTION_ID = "ev:doc-b:3:0-15"
INNER_DETECTION_ID = "ev:doc-b:3:20-35"


def deps_with_two_fault_modes() -> AskDeps:
    """两个同级故障模式：各有征兆，共用检测手段。"""
    outer = Concept(id=OUTER_RACE, name="轴承外圈疲劳剥落", type="fault_mode")
    inner = Concept(id=INNER_RACE, name="轴承内圈疲劳剥落", type="fault_mode")
    bpfo = Concept(id=BPFO, name="外圈故障频率及其倍频", type="symptom")
    bpfi = Concept(id=BPFI, name="内圈故障频率及其倍频", type="symptom")
    spectrum = Concept(id=SPECTRUM, name="频谱分析", type="detection_method")
    evidence = (
        EvidenceRecord(
            id=OUTER_SYMPTOM_ID,
            doc_id=DOC_ID,
            page=2,
            text="外圈剥落会出现 BPFO 及其倍频。",
        ),
        EvidenceRecord(
            id=INNER_SYMPTOM_ID,
            doc_id=DOC_ID,
            page=2,
            text="内圈剥落会出现 BPFI 及其倍频。",
        ),
        EvidenceRecord(
            id=OUTER_DETECTION_ID,
            doc_id=DOC_ID,
            page=3,
            text="外圈剥落常用频谱分析确认。",
        ),
        EvidenceRecord(
            id=INNER_DETECTION_ID,
            doc_id=DOC_ID,
            page=3,
            text="内圈剥落也常用频谱分析确认。",
        ),
    )
    graph = FakeGraphReader(
        facts=(
            FactEdge(
                head_id=OUTER_RACE,
                relation="indicated_by",
                tail_id=BPFO,
                evidence_ids=(OUTER_SYMPTOM_ID,),
                confidence=0.8,
                doc_id=DOC_ID,
            ),
            FactEdge(
                head_id=INNER_RACE,
                relation="indicated_by",
                tail_id=BPFI,
                evidence_ids=(INNER_SYMPTOM_ID,),
                confidence=0.8,
                doc_id=DOC_ID,
            ),
            FactEdge(
                head_id=OUTER_RACE,
                relation="detected_by",
                tail_id=SPECTRUM,
                evidence_ids=(OUTER_DETECTION_ID,),
                confidence=0.7,
                doc_id=DOC_ID,
            ),
            FactEdge(
                head_id=INNER_RACE,
                relation="detected_by",
                tail_id=SPECTRUM,
                evidence_ids=(INNER_DETECTION_ID,),
                confidence=0.7,
                doc_id=DOC_ID,
            ),
        ),
        concepts=(outer, inner, bpfo, bpfi, spectrum),
        evidence=evidence,
    )
    return AskDeps(
        graph_store=graph,
        document_names=FakeDocumentNames({DOC_ID: "论文B.pdf"}),
    )


def test_ask_prints_comparison_partitions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_ask(
        "轴承外圈剥落和轴承内圈剥落怎么区分？", deps=deps_with_two_fault_modes()
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "检索路径：图命中" in output
    assert "对比：轴承外圈疲劳剥落" in output
    assert "共同点：" in output
    assert "仅「轴承外圈疲劳剥落」有：" in output
    assert "仅「轴承内圈疲劳剥落」有：" in output
    assert "—表现为→ 外圈故障频率及其倍频" in output
    assert "—表现为→ 内圈故障频率及其倍频" in output
    assert "依据（轴承外圈疲劳剥落 → 频谱分析）：论文B.pdf 第 3 页" in output
    assert "依据（轴承内圈疲劳剥落 → 频谱分析）：论文B.pdf 第 3 页" in output
    assert "论文B.pdf 第 2 页" in output

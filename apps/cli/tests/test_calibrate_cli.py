"""calibrate 子命令：单遍采样、阈值推荐、报告留档与配置写回。

采样用内存假服务，不连 Neo4j/Postgres/网络；真实校准由维护者手动跑
``idr calibrate``，结果写进 ``config/retrieval.yaml`` 并留档。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.graph import Concept, EvidenceRecord, FactEdge
from idr_core.llm import EmbeddingResponse
from idr_core.llm.rerank import RerankResponse, RerankScore
from idr_core.vector_store import VectorMatch
from idr_rotating import AskDeps
from idr_rotating.calibration import VectorSample
from idr_rotating.evaluation import GoldenQuestion
from idr_rotating.retrieval import RetrievalSettings

from idr_cli.calibrate import collect_samples, run_calibrate
from idr_cli.main import main

ANCHOR_EVIDENCE = "ev:0000000000000001:1:0-3"
OTHER_EVIDENCE = "ev:0000000000000002:1:0-3"
ANCHOR_TEXT = "外圈剥落的证据。"
OTHER_TEXT = "另外一段候选。"


class FakeGraphReader:
    def __init__(self) -> None:
        self._evidence = {
            ANCHOR_EVIDENCE: EvidenceRecord(
                id=ANCHOR_EVIDENCE, doc_id="doc-a", page=1, text=ANCHOR_TEXT
            ),
            OTHER_EVIDENCE: EvidenceRecord(
                id=OTHER_EVIDENCE, doc_id="doc-a", page=1, text=OTHER_TEXT
            ),
        }

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        return ()

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        return ()

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        return tuple(
            self._evidence[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in self._evidence
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2] for _ in texts]

    def embed_with_usage(self, texts: Sequence[str]) -> EmbeddingResponse:
        return EmbeddingResponse(
            vectors=self.embed(texts), model="fake-embedding", input_tokens=3
        )


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
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls = 0

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        self.calls += 1
        return RerankResponse(
            scores=tuple(
                RerankScore(index=index, score=self._scores.get(text, 0.0))
                for index, text in enumerate(documents)
            ),
            model="fake-rerank",
            input_tokens=5,
        )


def two_questions() -> tuple[GoldenQuestion, GoldenQuestion]:
    answerable = GoldenQuestion(
        id="T-1",
        question_class="factual",
        question="轴承外圈疲劳剥落有什么征兆？",
        expected_concepts=("concept:bearing_outer_race_fatigue_spalling",),
        expected_evidence_ids=(ANCHOR_EVIDENCE,),
    )
    refusal = GoldenQuestion(
        id="R-1",
        question_class="factual",
        question="汽轮机有什么故障模式？",
        expected_concepts=(),
        expected_evidence_ids=(),
        expect_refusal=True,
        expected_refusal_reason="no_evidence",
    )
    return answerable, refusal


def write_golden(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "questions": [
            {
                "id": "T-1",
                "class": "factual",
                "question": "轴承外圈疲劳剥落有什么征兆？",
                "expected_concepts": [
                    "concept:bearing_outer_race_fatigue_spalling"
                ],
                "expected_evidence_ids": [ANCHOR_EVIDENCE],
            },
            {
                "id": "R-1",
                "class": "factual",
                "question": "汽轮机有什么故障模式？",
                "expect_refusal": True,
                "expected_refusal_reason": "no_evidence",
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_config(path: Path, threshold: float = 0.35) -> Path:
    path.write_text(
        "# 检索参数：注释要保留\n"
        "vector_top_k: 20\n"
        f"vector_min_rerank_score: {threshold}\n"
        "vector_max_results: 5\n",
        encoding="utf-8",
    )
    return path


def fixed_sampler(
    questions: Sequence[GoldenQuestion],
    deps: AskDeps,
    settings: RetrievalSettings,
) -> tuple[VectorSample, ...]:
    assert [question.id for question in questions] == ["T-1", "R-1"]
    return (
        VectorSample(
            question_id="T-1",
            expect_refusal=False,
            expected_evidence_ids=(ANCHOR_EVIDENCE,),
            scored=((ANCHOR_EVIDENCE, 0.90),),
        ),
        VectorSample(
            question_id="R-1",
            expect_refusal=True,
            expected_evidence_ids=(),
            scored=((OTHER_EVIDENCE, 0.70),),
        ),
    )


def test_collect_samples_sees_every_candidate_despite_the_current_threshold() -> None:
    embedder = FakeEmbedder()
    reranker = FakeReranker({ANCHOR_TEXT: 0.9, OTHER_TEXT: 0.2})
    deps = AskDeps(
        graph_store=FakeGraphReader(),
        embedder=embedder,
        vector_store=FakeVectorReader((ANCHOR_EVIDENCE, OTHER_EVIDENCE)),
        reranker=reranker,
    )
    settings = RetrievalSettings(
        vector_top_k=2,
        vector_min_rerank_score=0.95,
        vector_max_results=2,
    )

    samples = collect_samples(two_questions(), deps, settings)

    assert embedder.calls == 2
    assert reranker.calls == 2
    assert samples[0].scored == ((ANCHOR_EVIDENCE, 0.9), (OTHER_EVIDENCE, 0.2))
    assert samples[0].top_score == 0.9
    # 阈值 0.95 会筛掉 0.2 的候选；采样必须看到它，否则分布是不完整的。
    assert samples[1].scored == ((ANCHOR_EVIDENCE, 0.9), (OTHER_EVIDENCE, 0.2))
    assert settings.vector_min_rerank_score == 0.95


def test_run_calibrate_reports_the_recommendation_and_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    report = tmp_path / "reports" / "calibration.json"

    exit_code = run_calibrate(
        deps=AskDeps(graph_store=FakeGraphReader()),
        golden_path=golden,
        out_path=report,
        sampler=fixed_sampler,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "推荐阈值：0.80" in output
    assert "拒答对照：1/1 正确拒答" in output
    assert "期望证据：1/1 保留" in output
    assert "报告：" in output
    payload = json.loads(report.read_text("utf-8"))
    assert payload["recommendation"]["threshold"] == 0.8
    assert payload["recommendation"]["all_refusals_correct"] is True
    assert len(payload["samples"]) == 2
    assert payload["samples"][1]["top_score"] == 0.7


def test_run_calibrate_can_write_the_threshold_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    config = write_config(tmp_path / "retrieval.yaml")

    exit_code = run_calibrate(
        deps=AskDeps(graph_store=FakeGraphReader()),
        golden_path=golden,
        out_path=tmp_path / "report.json",
        settings_path=config,
        write_config=True,
        sampler=fixed_sampler,
    )

    assert exit_code == 0
    text = config.read_text(encoding="utf-8")
    assert "# 检索参数：注释要保留" in text
    assert "vector_min_rerank_score: 0.80" in text
    assert "vector_min_rerank_score_calibrated: true" in text
    assert "已写入" in capsys.readouterr().out


def test_run_calibrate_fails_cleanly_without_vector_dependencies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")

    exit_code = run_calibrate(
        deps=AskDeps(graph_store=FakeGraphReader()),
        golden_path=golden,
        out_path=tmp_path / "report.json",
    )

    assert exit_code == 1
    assert "校准失败" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_main_dispatches_calibrate_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import idr_cli.main as main_module

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

    captured: dict[str, object] = {}

    def fake_run_calibrate(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(main_module, "run_calibrate", fake_run_calibrate)
    config = tmp_path / "retrieval.yaml"

    exit_code = main(
        [
            "calibrate",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "report.json"),
            "--write",
        ],
        ask_deps_factory=lambda config, source_dir: AskDeps(
            graph_store=FakeGraphReader()
        ),
    )

    assert exit_code == 0
    assert captured["settings_path"] == config
    assert captured["write_config"] is True
    assert captured["out_path"] == tmp_path / "report.json"
    assert "校准" not in capsys.readouterr().err

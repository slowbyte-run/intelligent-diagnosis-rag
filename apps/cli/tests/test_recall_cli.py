"""recall 子命令：只量向量召回，输出 recall@k 与逐题名次。

全部用内存假实现，不连 Postgres / 网络；真实召回诊断由维护者手动跑
``idr recall``，报告写进 ``data/output/reports/``。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.graph import Concept, EvidenceRecord, FactEdge
from idr_core.vector_store import VectorMatch
from idr_rotating import AskDeps
from idr_rotating.retrieval import RetrievalSettings

from idr_cli.main import main
from idr_cli.recall import run_recall

ANCHOR_EVIDENCE = "ev:0000000000000001:1:0-3"
OTHER_EVIDENCE = "ev:0000000000000002:1:0-3"


class FakeGraphReader:
    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        return ()

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        return ()

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        return ()

    def evidence_by_page(self, doc_id: str, page: int) -> tuple[EvidenceRecord, ...]:
        return ()


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeTextSearch:
    def __init__(self, evidence_ids: Sequence[str]) -> None:
        self._evidence_ids = tuple(evidence_ids)
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def search_evidence(
        self, terms: Sequence[str], *, limit: int
    ) -> tuple[str, ...]:
        self.calls.append((tuple(terms), limit))
        return self._evidence_ids[:limit]


class FakeVectorReader:
    def __init__(self, evidence_ids: Sequence[str]) -> None:
        self._evidence_ids = tuple(evidence_ids)
        self.limits: list[int] = []

    def search_similar(
        self, vector: Sequence[float], *, limit: int
    ) -> tuple[VectorMatch, ...]:
        self.limits.append(limit)
        return tuple(
            VectorMatch(evidence_id=evidence_id, distance=0.1)
            for evidence_id in self._evidence_ids[:limit]
        )


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


def ask_deps(
    *,
    matches: Sequence[str],
    query_instruction: str = "",
) -> tuple[AskDeps, FakeEmbedder, FakeVectorReader]:
    embedder = FakeEmbedder()
    store = FakeVectorReader(matches)
    deps = AskDeps(
        graph_store=FakeGraphReader(),
        embedder=embedder,
        vector_store=store,
        retrieval_settings=RetrievalSettings(
            query_instruction=query_instruction
        ),
    )
    return deps, embedder, store


def test_recall_reports_every_configured_k_and_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    report = tmp_path / "reports" / "recall.json"
    # 绑定证据排在第 26 名：recall@20 漏掉、recall@50 接住，名次要如实报出来。
    deps, embedder, store = ask_deps(
        matches=(
            *(f"ev:00000000000000{index:02d}:1:0-1" for index in range(25)),
            ANCHOR_EVIDENCE,
        )
    )

    exit_code = run_recall(
        deps=deps, golden_path=golden, out_path=report, top_k=100
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "recall@20" in output
    assert "recall@50" in output
    assert "recall@100" in output
    assert "T-1" in output and "第 26 名" in output
    assert len(embedder.texts) == 4  # 2 个变体 × 2 道题
    assert store.limits[:2] == [100, 100]

    payload = json.loads(report.read_text("utf-8"))
    assert payload["question_count"] == 2
    assert payload["answerable_count"] == 1
    assert [item["variant"] for item in payload["variants"]] == [
        "raw",
        "configured",
    ]
    assert payload["variants"][0]["summary"]["20"]["hits"] == 0
    assert payload["variants"][0]["summary"]["50"]["hits"] == 1


def test_a_configured_instruction_changes_what_is_embedded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    deps, embedder, _ = ask_deps(
        matches=(ANCHOR_EVIDENCE,), query_instruction="crosslingual"
    )

    exit_code = run_recall(
        deps=deps,
        golden_path=golden,
        out_path=tmp_path / "report.json",
        variants="raw,configured",
    )

    assert exit_code == 0
    assert embedder.texts[0] == "轴承外圈疲劳剥落有什么征兆？"
    assert embedder.texts[2].startswith("Instruct: ")
    assert "\nQuery:轴承外圈疲劳剥落有什么征兆？" in embedder.texts[2]
    capsys.readouterr()


def test_only_ks_that_fit_inside_top_k_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    deps, _, _ = ask_deps(matches=(ANCHOR_EVIDENCE,))

    exit_code = run_recall(
        deps=deps, golden_path=golden, out_path=tmp_path / "r.json", top_k=30
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "recall@20" in output
    assert "recall@50" not in output


def test_an_unknown_variant_fails_before_any_model_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    deps, embedder, store = ask_deps(matches=(ANCHOR_EVIDENCE,))

    exit_code = run_recall(
        deps=deps,
        golden_path=golden,
        out_path=tmp_path / "r.json",
        variants="raw,translate",
    )

    assert exit_code == 1
    assert "未知的查询变体" in capsys.readouterr().err
    assert embedder.texts == []
    assert store.limits == []
    assert not (tmp_path / "r.json").exists()


def test_missing_vector_dependencies_fail_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")

    exit_code = run_recall(
        deps=AskDeps(graph_store=FakeGraphReader()), golden_path=golden
    )

    assert exit_code == 1
    assert "recall 需要嵌入与向量检索依赖" in capsys.readouterr().err


def test_main_dispatches_recall_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    def fake_run_recall(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(main_module, "run_recall", fake_run_recall)

    exit_code = main(
        [
            "recall",
            "--golden",
            str(tmp_path / "golden.json"),
            "--out",
            str(tmp_path / "report.json"),
            "--top-k",
            "50",
            "--variants",
            "raw,enriched",
        ],
        ask_deps_factory=lambda config, source_dir: AskDeps(
            graph_store=FakeGraphReader()
        ),
    )

    assert exit_code == 0
    assert captured["golden_path"] == tmp_path / "golden.json"
    assert captured["out_path"] == tmp_path / "report.json"
    assert captured["top_k"] == 50
    assert captured["variants"] == "raw,enriched"


def test_lexical_flag_fuses_term_hits_into_the_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    deps, _, _ = ask_deps(matches=(OTHER_EVIDENCE, ANCHOR_EVIDENCE))
    deps = AskDeps(
        graph_store=deps.graph_store,
        embedder=deps.embedder,
        vector_store=deps.vector_store,
        text_search=FakeTextSearch((ANCHOR_EVIDENCE,)),
        retrieval_settings=deps.retrieval_settings,
    )

    exit_code = run_recall(
        deps=deps,
        golden_path=golden,
        out_path=tmp_path / "report.json",
        variants="raw",
        lexical=True,
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "report.json").read_text("utf-8"))
    variant = payload["variants"][0]
    assert variant["variant"] == "raw+lexical"
    first = variant["questions"][0]
    assert first["nearest_rank"] == 1
    assert first["retrieved"][0]["lexical"] is True
    assert first["retrieved"][1]["lexical"] is False
    capsys.readouterr()


def test_lexical_flag_without_a_text_search_port_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    deps, _, _ = ask_deps(matches=(ANCHOR_EVIDENCE,))

    exit_code = run_recall(
        deps=deps, golden_path=golden, out_path=tmp_path / "r.json", lexical=True
    )

    assert exit_code == 1
    assert "词面诊断需要 text_search" in capsys.readouterr().err

"""向量召回诊断：只嵌入 + pgvector 检索，不重排。

诊断回答的是「绑定的期望证据进没进 top-k」而不是「答案对不对」：每道可答题
只嵌入一次，按向量距离取前 ``max_k`` 条，再看期望证据最早出现在第几名。
重排与阈值都不参与，否则又回到「分数低」与「没召回」分不清的老问题。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from idr_core.vector_store import VectorMatch
from idr_rotating.evaluation import GoldenQuestion
from idr_rotating.recall_diagnostic import (
    diagnose_recall,
    recall_report_to_dict,
)


class FakeEmbedder:
    """按文本返回定长向量，并记录被嵌入的文本。"""

    def __init__(self, vectors: dict[str, Sequence[float]] | None = None) -> None:
        self._vectors = dict(vectors or {})
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [list(self._vectors.get(text, [0.0])) for text in texts]


class FakeVectorStore:
    """按标识返回固定距离，并记录每次查询的条数。"""

    def __init__(self, matches: Sequence[VectorMatch]) -> None:
        self._matches = tuple(matches)
        self.limits: list[int] = []

    def search_similar(
        self, vector: Sequence[float], *, limit: int
    ) -> tuple[VectorMatch, ...]:
        self.limits.append(limit)
        return self._matches[:limit]


def question(
    question_id: str,
    *,
    expected: tuple[str, ...],
    refusal: bool = False,
    question_class: str = "factual",
) -> GoldenQuestion:
    return GoldenQuestion(
        id=question_id,
        question_class=question_class,
        question=f"问题 {question_id}",
        expected_concepts=() if refusal else ("concept:bearing_outer_race_fatigue_spalling",),
        expected_evidence_ids=expected,
        expect_refusal=refusal,
    )


def matches(*ids: str) -> tuple[VectorMatch, ...]:
    return tuple(
        VectorMatch(evidence_id=evidence_id, distance=index / 100)
        for index, evidence_id in enumerate(ids)
    )


def test_recall_counts_questions_whose_bound_evidence_is_in_top_k() -> None:
    questions = (
        question("A", expected=("ev:a:1:0-1",)),
        question("B", expected=("ev:b:1:0-1",)),
        question("C", expected=("ev:missing:1:0-1",)),
    )
    retrieved = matches("ev:a:1:0-1", "ev:x:1:0-1", "ev:b:1:0-1")
    store = FakeVectorStore(retrieved)

    report, call = diagnose_recall(
        questions,
        embedder=FakeEmbedder(),
        vector_store=store,
        query_for=lambda item: item.question,
        variant="raw",
        max_k=3,
        ks=(1, 2, 3),
    )

    assert store.limits == [3, 3, 3]
    assert call.elapsed_seconds >= 0.0
    assert [(item.k, item.hits, item.answerable) for item in report.summary] == [
        (1, 1, 3),
        (2, 1, 3),
        (3, 2, 3),
    ]
    assert report.summary[0].rate == pytest.approx(1 / 3)


def test_nearest_rank_reports_how_far_a_miss_is() -> None:
    questions = (question("A", expected=("ev:a:1:0-1",)),)
    store = FakeVectorStore(matches("ev:x:1:0-1", "ev:y:1:0-1", "ev:a:1:0-1"))

    report, _ = diagnose_recall(
        questions,
        embedder=FakeEmbedder(),
        vector_store=store,
        query_for=lambda item: item.question,
        variant="raw",
        max_k=3,
        ks=(1, 3),
    )

    case = report.cases[0]
    assert case.nearest_rank == 3
    assert case.hit_at(2) is False
    assert case.hit_at(3) is True
    assert case.matched_at(3) == ("ev:a:1:0-1",)


def test_refusal_controls_are_reported_but_not_scored() -> None:
    questions = (
        question("A", expected=("ev:a:1:0-1",)),
        question("R", expected=(), refusal=True),
    )

    report, _ = diagnose_recall(
        questions,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(matches("ev:a:1:0-1")),
        query_for=lambda item: item.question,
        variant="raw",
        max_k=1,
        ks=(1,),
    )

    assert [case.question_id for case in report.cases] == ["A", "R"]
    assert report.summary[0].answerable == 1
    assert report.summary[0].hits == 1


def test_the_query_builder_decides_what_is_embedded() -> None:
    embedder = FakeEmbedder()
    questions = (question("A", expected=("ev:a:1:0-1",)),)

    diagnose_recall(
        questions,
        embedder=embedder,
        vector_store=FakeVectorStore(matches("ev:a:1:0-1")),
        query_for=lambda item: f"改写：{item.question}",
        variant="enriched",
        max_k=1,
        ks=(1,),
    )

    assert embedder.texts == ["改写：问题 A"]


def test_a_k_beyond_the_retrieval_limit_is_rejected() -> None:
    """要报 recall@50 就必须真的取 50 条，不能拿 20 条的结果冒充。"""
    with pytest.raises(ValueError, match="max_k"):
        diagnose_recall(
            (question("A", expected=("ev:a:1:0-1",)),),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(matches()),
            query_for=lambda item: item.question,
            variant="raw",
            max_k=20,
            ks=(50,),
        )


def test_no_questions_is_rejected_instead_of_reporting_zero() -> None:
    with pytest.raises(ValueError, match="没有"):
        diagnose_recall(
            (),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(matches()),
            query_for=lambda item: item.question,
            variant="raw",
            max_k=10,
            ks=(10,),
        )


def test_report_serialisation_keeps_the_evidence_and_the_rank() -> None:
    questions = (
        question("A", expected=("ev:a:1:0-1",), question_class="comparative"),
        question("R", expected=(), refusal=True),
    )

    report, _ = diagnose_recall(
        questions,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(matches("ev:x:1:0-1", "ev:a:1:0-1")),
        query_for=lambda item: item.question,
        variant="raw",
        max_k=2,
        ks=(1, 2),
    )

    payload = recall_report_to_dict(report)

    assert payload["schema_version"] == 1
    assert payload["variant"] == "raw"
    assert payload["summary"]["1"] == {
        "k": 1,
        "hits": 0,
        "answerable": 1,
        "rate": 0.0,
    }
    assert payload["summary"]["2"] == {
        "k": 2,
        "hits": 1,
        "answerable": 1,
        "rate": 1.0,
    }
    first, refusal = payload["questions"]
    assert first["id"] == "A"
    assert first["nearest_rank"] == 2
    assert first["matched_evidence_ids"] == ["ev:a:1:0-1"]
    assert [item["evidence_id"] for item in first["retrieved"]] == [
        "ev:x:1:0-1",
        "ev:a:1:0-1",
    ]
    assert refusal["expect_refusal"] is True
    assert refusal["nearest_rank"] is None


def test_lexical_candidates_are_fused_and_marked_in_the_report() -> None:
    questions = (question("A", expected=("ev:a:1:0-1",)),)

    report, _ = diagnose_recall(
        questions,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(
            matches("ev:x:1:0-1", "ev:a:1:0-1")
        ),
        query_for=lambda item: item.question,
        variant="configured+lexical",
        max_k=2,
        ks=(1, 2),
        lexical_for=lambda item: ("ev:a:1:0-1",),
    )

    # 词面第 1（1/61）压过向量第 1（1/61）的同分，按标识排序后仍在第 1。
    case = report.cases[0]
    assert [item.evidence_id for item in case.retrieved] == [
        "ev:a:1:0-1",
        "ev:x:1:0-1",
    ]
    assert case.retrieved[0].distance == 0.01
    assert case.retrieved[0].lexical is True
    assert report.summary[0].hits == 1

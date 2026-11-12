"""向量召回诊断：只嵌入 + pgvector 检索，不重排、不过阈值。

用途是把「期望证据为什么没进答案」拆成两段里更靠前的那一段：

- **召回**：问题嵌入一次，pgvector 顺序扫描取前 ``max_k`` 条。诊断只看这一步
  有没有把金标准绑定的证据段带进来，以及最近的绑定证据排第几；
- **重排**：不在本模块的度量范围内。召回没带进来的证据，重排分数再高也救不
  回来；先量召回，才不会把召回问题误判成阈值问题。

每道题只做一次嵌入、一次向量检索，因此一次诊断的成本是「金标准题数 × 1 次
嵌入」。报告只写数字与逐题名次，不写结论；结论由读报告的人下。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from idr_core.embedding import Embedder
from idr_core.llm import ModelCall, embed_texts
from idr_core.vector_store import VectorReader
from idr_rotating.evaluation import GoldenQuestion
from idr_rotating.vector_recall import fuse_rankings

RECALL_SCHEMA_VERSION = 1
DEFAULT_RECALL_KS = (20, 50, 100)
DEFAULT_MAX_K = 100


@dataclass(frozen=True, slots=True)
class RecallSummary:
    """某个 k 上的召回：分母只算可答题，拒答对照不参与。"""

    k: int
    hits: int
    answerable: int

    @property
    def rate(self) -> float | None:
        if self.answerable == 0:
            return None
        return self.hits / self.answerable


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """一条召回候选；向量命中带距离，词面命中带标记，两者可以同时为真。"""

    evidence_id: str
    distance: float | None
    lexical: bool = False


@dataclass(frozen=True, slots=True)
class RecallCase:
    """一道题的召回结果：查询文本、完整候选列表与绑定证据的名次。"""

    question_id: str
    question_class: str
    expect_refusal: bool
    expected_evidence_ids: tuple[str, ...]
    query: str
    retrieved: tuple[RetrievedEvidence, ...]

    @property
    def nearest_rank(self) -> int | None:
        """最近的绑定证据在候选列表里的名次（1 起）；没召回时为 ``None``。"""
        expected = set(self.expected_evidence_ids)
        return next(
            (
                index + 1
                for index, match in enumerate(self.retrieved)
                if match.evidence_id in expected
            ),
            None,
        )

    @property
    def nearest_distance(self) -> float | None:
        rank = self.nearest_rank
        return None if rank is None else self.retrieved[rank - 1].distance

    def matched_at(self, k: int) -> tuple[str, ...]:
        """前 k 条里命中的绑定证据，按名次排列。"""
        expected = set(self.expected_evidence_ids)
        return tuple(
            match.evidence_id
            for match in self.retrieved[:k]
            if match.evidence_id in expected
        )

    def hit_at(self, k: int) -> bool:
        return bool(self.matched_at(k))


@dataclass(frozen=True, slots=True)
class RecallReport:
    """一次查询变体的完整诊断结果。"""

    variant: str
    max_k: int
    ks: tuple[int, ...]
    cases: tuple[RecallCase, ...]

    @property
    def summary(self) -> tuple[RecallSummary, ...]:
        answerable = [case for case in self.cases if not case.expect_refusal]
        return tuple(
            RecallSummary(
                k=k,
                hits=sum(1 for case in answerable if case.hit_at(k)),
                answerable=len(answerable),
            )
            for k in self.ks
        )

    def summary_by_class(self, k: int) -> dict[str, RecallSummary]:
        """按题型分组；拒答对照不参与，分母为零时 rate 为 ``None``。"""
        groups: dict[str, list[RecallCase]] = {}
        for case in self.cases:
            if not case.expect_refusal:
                groups.setdefault(case.question_class, []).append(case)
        return {
            name: RecallSummary(
                k=k,
                hits=sum(1 for case in cases if case.hit_at(k)),
                answerable=len(cases),
            )
            for name, cases in groups.items()
        }


QueryFor = Callable[[GoldenQuestion], str]
LexicalFor = Callable[[GoldenQuestion], Sequence[str]]


def diagnose_recall(
    questions: Sequence[GoldenQuestion],
    *,
    embedder: Embedder,
    vector_store: VectorReader,
    query_for: QueryFor,
    variant: str,
    max_k: int = DEFAULT_MAX_K,
    ks: Sequence[int] = DEFAULT_RECALL_KS,
    lexical_for: LexicalFor | None = None,
) -> tuple[RecallReport, ModelCall]:
    """对每道题嵌入一次查询文本，取前 ``max_k`` 条并统计各 k 的召回。

    给了 ``lexical_for`` 时，把它的词面候选按名次与向量候选做 RRF 融合（与
    线上 ``fuse_rankings`` 是同一个函数），``variant`` 由调用方补上词面标记。
    """
    if not questions:
        raise ValueError("没有可诊断的题目：至少需要一道金标准题")
    if not ks:
        raise ValueError("至少要报告一个 k")
    if max_k < 1:
        raise ValueError(f"max_k 必须是正整数，收到 {max_k}")
    if any(k < 1 for k in ks):
        raise ValueError(f"k 必须是正整数，收到 {ks}")
    deepest = max(ks)
    if deepest > max_k:
        raise ValueError(
            f"要报告的 recall@{deepest} 超过检索上限 max_k={max_k}；"
            "否则会把「没取到」误报成「没召回」"
        )

    ordered_ks = tuple(dict.fromkeys(int(k) for k in ks))
    queries = [query_for(question) for question in questions]
    vectors, call = embed_texts(embedder, queries)
    if len(vectors) != len(questions):
        raise ValueError(
            f"嵌入服务返回 {len(vectors)} 个向量，请求了 {len(questions)} 个查询"
        )

    cases = tuple(
        _recall_case(
            question,
            query,
            vector,
            vector_store,
            max_k=max_k,
            lexical_ids=() if lexical_for is None else lexical_for(question),
        )
        for question, query, vector in zip(questions, queries, vectors, strict=True)
    )
    return (
        RecallReport(
            variant=variant, max_k=max_k, ks=ordered_ks, cases=cases
        ),
        call,
    )


def _recall_case(
    question: GoldenQuestion,
    query: str,
    vector: Sequence[float],
    vector_store: VectorReader,
    *,
    max_k: int,
    lexical_ids: Sequence[str],
) -> RecallCase:
    matches = vector_store.search_similar(vector, limit=max_k)
    distances = {match.evidence_id: match.distance for match in matches}
    vector_ids = tuple(match.evidence_id for match in matches)
    candidate_ids = (
        fuse_rankings(vector_ids, lexical_ids, limit=max_k)
        if lexical_ids
        else vector_ids
    )
    return RecallCase(
        question_id=question.id,
        question_class=question.question_class,
        expect_refusal=question.expect_refusal,
        expected_evidence_ids=question.expected_evidence_ids,
        query=query,
        retrieved=tuple(
            RetrievedEvidence(
                evidence_id=evidence_id,
                distance=distances.get(evidence_id),
                lexical=evidence_id in set(lexical_ids),
            )
            for evidence_id in candidate_ids
        ),
    )


def recall_report_to_dict(report: RecallReport) -> dict[str, Any]:
    """报告的 JSON 形态；字段名是留档与后续对比的契约。"""
    return {
        "schema_version": RECALL_SCHEMA_VERSION,
        "variant": report.variant,
        "max_k": report.max_k,
        "ks": list(report.ks),
        "summary": {
            str(item.k): {
                "k": item.k,
                "hits": item.hits,
                "answerable": item.answerable,
                "rate": _rounded(item.rate),
            }
            for item in report.summary
        },
        "questions": [_case_to_dict(case) for case in report.cases],
    }


def _case_to_dict(case: RecallCase) -> dict[str, Any]:
    return {
        "id": case.question_id,
        "class": case.question_class,
        "expect_refusal": case.expect_refusal,
        "query": case.query,
        "expected_evidence_ids": list(case.expected_evidence_ids),
        "nearest_rank": case.nearest_rank,
        "nearest_distance": case.nearest_distance,
        "matched_evidence_ids": list(
            case.matched_at(len(case.retrieved))
        ),
        "retrieved": [
            {
                "evidence_id": match.evidence_id,
                "distance": (
                    None
                    if match.distance is None
                    else _distance(match.distance)
                ),
                "lexical": match.lexical,
            }
            for match in case.retrieved
        ],
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _distance(value: float) -> float:
    return round(value, 6)

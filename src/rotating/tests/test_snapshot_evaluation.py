"""快照检索评估：不连 Neo4j、不联网、同一快照两次给出同一组指标。

这些测试跑提交进版本库的真实快照与真实金标准集，验证评估入口的外部行为：
三项指标的名字与口径、金标准集绑定的校验和、结构上不可引用的期望证据如何
归因，以及同一份输入两次运行的完全一致性。图读端口的单元行为另见
``src/core/tests/test_snapshot_reader.py``。
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from idr_core.graph.snapshot import load_snapshot
from idr_core.graph.snapshot_reader import SnapshotGraphReader
from idr_rotating import AskDeps, ask
from idr_rotating.answer import AnswerPath
from idr_rotating.citation_check import CitationCheck
from idr_rotating.evaluation import (
    GoldenSet,
    SnapshotBindingError,
    load_golden_set,
)
from idr_rotating.snapshot_eval import (
    evaluate_snapshot,
    snapshot_report_to_dict,
)
from idr_rotating.snapshot_export import DEFAULT_SNAPSHOT_PATH


def run_report(**overrides: Any) -> dict[str, Any]:
    golden = load_golden_set()
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    report = evaluate_snapshot(golden, snapshot, **overrides)
    return snapshot_report_to_dict(report)


def test_snapshot_evaluation_reports_three_explicitly_named_metrics() -> None:
    payload = run_report(generated_at="2026-11-13T00:00:00+00:00")

    assert payload["graph_frozen"] is True
    assert payload["graph_label"] == "图谱已固定"
    assert payload["graph"]["mode"] == "snapshot"
    assert payload["graph"]["checksum"] == load_snapshot(DEFAULT_SNAPSHOT_PATH).checksum
    assert payload["golden"]["checksum"] == payload["graph"]["checksum"]

    metrics = payload["metrics"]
    assert metrics["question_count"] == 50
    assert metrics["answerable_count"] == 48
    assert metrics["retrieval_hit_rate"] is not None
    assert metrics["citation_text_consistency"] is not None
    assert metrics["refusal_correctness"] is not None
    # 「引用准确率」在快照路径下必须叫它真正度量的东西；不能沿用回源 PDF 的名字。
    assert "citation_accuracy" not in metrics
    assert payload["citation_pdf_verifiable"]["checked"] is False
    assert payload["citation_pdf_verifiable"]["citation_pdf_accuracy"] is None


def test_snapshot_evaluation_never_opens_a_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有 Neo4j、没有嵌入与重排依赖，评估跑在快照上就不该联网。"""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("快照评估不允许联网")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    payload = run_report()

    assert payload["metrics"]["retrieval_hit_rate"] is not None


def test_two_runs_on_the_same_snapshot_are_identical_except_timestamp() -> None:
    first = run_report(generated_at="2026-11-13T00:00:00+00:00")
    second = run_report(generated_at="2026-11-13T00:00:01+00:00")

    first.pop("generated_at")
    second.pop("generated_at")

    assert first["metrics"] == second["metrics"]
    assert first["attribution"] == second["attribution"]
    assert first["cases"] == second["cases"]
    assert first["result_digest"] == second["result_digest"]


def test_expected_evidence_only_on_internal_edges_is_attributed_not_a_plain_miss() -> None:
    payload = run_report()
    attribution = payload["attribution"]

    assert len(attribution["unreachable_evidence"]) == 6
    assert "F-21" in attribution["unreachable_question_ids"]
    assert attribution["reachable_answerable_count"] == 47

    unreachable_ids = {
        item["evidence_id"] for item in attribution["unreachable_evidence"]
    }
    assert "ev:906bd81774ba8b1a:33:5848-6279" in unreachable_ids

    f21 = next(case for case in payload["cases"] if case["id"] == "F-21")
    assert f21["hit"] is False
    assert f21["miss_attribution"] == "expected_evidence_only_on_internal_edges"
    assert f21["unreachable_expected_evidence"] == ["ev:4a8dc5af5de2cfe4:19:3961-4466"]

    p01 = next(case for case in payload["cases"] if case["id"] == "P-01")
    assert p01["unreachable_expected_evidence"] == [
        "ev:4a8dc5af5de2cfe4:20:2446-2660"
    ]


def test_unreachable_questions_are_excluded_from_the_reachable_denominator() -> None:
    payload = run_report()

    assert (
        payload["attribution"]["retrieval_hit_rate_excluding_unreachable"]
        is not None
    )
    assert payload["metrics"]["answerable_count"] == 48
    assert payload["attribution"]["reachable_answerable_count"] == 47


def test_pdf_citation_verification_is_reported_separately_when_available() -> None:
    def pdf_verifier(citation: dict[str, Any]) -> CitationCheck:
        return CitationCheck(ok=True, reason="引文与源 PDF 该页逐字一致", offset=0)

    payload = run_report(pdf_verifier=pdf_verifier)
    pdf = payload["citation_pdf_verifiable"]

    assert pdf["checked"] is True
    assert pdf["citations_total"] == payload["metrics"]["citations_total"]
    assert pdf["citations_verified"] == pdf["citations_total"]
    assert pdf["citation_pdf_accuracy"] == 1.0
    assert pdf["unavailable_reason"] is None


def test_pdf_verification_failures_do_not_change_the_text_consistency_metric() -> None:
    def pdf_verifier(citation: dict[str, Any]) -> CitationCheck:
        return CitationCheck(ok=False, reason="引文未在源 PDF 该页逐字出现")

    payload = run_report(pdf_verifier=pdf_verifier)

    assert payload["metrics"]["citation_text_consistency"] == 1.0
    assert payload["citation_pdf_verifiable"]["citation_pdf_accuracy"] == 0.0
    assert payload["citation_pdf_verifiable"]["failures"]


def test_checksum_mismatch_fails_before_any_scoring() -> None:
    golden = load_golden_set()
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    mismatched = GoldenSet(
        path=golden.path,
        snapshot_checksum="sha256:" + "0" * 64,
        questions=golden.questions,
    )

    with pytest.raises(SnapshotBindingError, match="snapshot.checksum"):
        evaluate_snapshot(mismatched, snapshot)


def test_the_committed_snapshot_baseline_is_locked() -> None:
    """图谱冻结后，这个基线只应随「显式重新导出 + 重新绑定」而变化。

    数字来自 ``docs/retrieval-eval.md``；改快照或改检索逻辑时，这些断言会
    先红，提醒把新基线写进文档，而不是让报告悄悄漂移。
    """
    payload = run_report()
    metrics = payload["metrics"]

    assert (metrics["retrieval_hits"], metrics["answerable_count"]) == (37, 48)
    assert metrics["retrieval_hit_rate"] == 0.7708
    assert (metrics["citations_text_consistent"], metrics["citations_total"]) == (
        843,
        843,
    )
    assert metrics["citation_text_consistency"] == 1.0
    assert (metrics["refusal_correct"], metrics["question_count"]) == (47, 50)
    assert metrics["refusal_correctness"] == 0.94
    assert {
        name: (group["retrieval_hits"], group["answerable"])
        for name, group in metrics["by_class"].items()
    } == {
        "factual": (20, 24),
        "comparative": (9, 12),
        "cross_document": (8, 12),
    }
    assert payload["attribution"]["reachable_hits"] == 37
    assert payload["attribution"]["retrieval_hit_rate_excluding_unreachable"] == 0.7872


def test_snapshot_reader_drives_the_existing_ask_facade_unchanged() -> None:
    """#32 的验收：快照通过既有 GraphReader 端口驱动 ask，调用方式不变。"""
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    evidence_ids = {record.id for record in snapshot.evidence}

    answer = ask(
        "离心泵包含哪些部件？",
        deps=AskDeps(graph_store=SnapshotGraphReader(snapshot)),
    )

    assert answer.path is AnswerPath.GRAPH
    assert answer.citations
    assert all(
        citation.evidence_id in evidence_ids for citation in answer.citations
    )

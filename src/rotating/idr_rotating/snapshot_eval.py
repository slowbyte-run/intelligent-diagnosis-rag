"""快照检索评估：只读冻结图谱，给出可重复、可归因的三项指标。

检索评估与抽取评估分开（父规格 #30）：这里的数据源只有提交进版本库的
``graph_snapshot.json``，不连 Neo4j、不调模型、不读 PDF。同一个问题在同一份
快照上永远得到同一个答案，分数变化只可能来自检索或回答逻辑。

三件事必须显式，不能藏在默认值里：

1. **图已固定。** 报告顶部写 ``graph_frozen`` / ``graph_label`` 与快照内容
   校验和；金标准集记录它绑定的校验和，不符时评估在打分前失败。
2. **引用准确率拆成两个名字。** 快照带证据段正文但不带 PDF，因此默认只给
   ``citation_text_consistency``（引文是否为快照原文的逐字片段、页码与文献是否
   对得上）；回源 PDF 的 ``citation_pdf_verifiable`` 只在显式注入 PDF 校验器
   时计算，否则为 ``null`` 并写明原因。两者不是同一个数，报告不合并。
3. **结构上不可引用的期望证据单列。** 有些金标准证据只挂在概念→证据内部边上，
   图路径不可能引用它们；报告按快照实际的关系边算出这份清单，并把「期望证据
   全部不可达」的题从「可达口径命中率」的分母中排除，同时保留全量口径。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idr_core.graph.snapshot import (
    SNAPSHOT_FORMAT,
    SNAPSHOT_SCHEMA_VERSION,
    GraphSnapshot,
)
from idr_core.graph.snapshot_reader import SnapshotGraphReader
from idr_rotating.answer import answer_to_dict
from idr_rotating.ask import AskDeps, ask
from idr_rotating.citation_check import CitationCheck, SnapshotCitationVerifier
from idr_rotating.evaluation import (
    CaseResult,
    GoldenQuestion,
    GoldenSet,
    EvaluationReport,
    check_snapshot_binding,
    evaluate_golden,
    report_to_dict,
)
from idr_rotating.snapshot_export import DEFAULT_SNAPSHOT_PATH

SNAPSHOT_EVALUATION_SCHEMA_VERSION = 1
SNAPSHOT_EVALUATION_KIND = "retrieval-snapshot-eval"

# 报告里逐项写清「这个数度量什么、不能拿它当什么」。
SNAPSHOT_METRIC_DEFINITIONS = {
    "retrieval_hit_rate": (
        "可答题中，至少命中一个期望证据段（跨文献题还要求每个期望文献至少一条"
        "引用）的比例；分母是可答题数。与历史 50 题基线同口径，用于对照。"
    ),
    "citation_text_consistency": (
        "答案实际输出的全部引用中，引文是快照里所引证据段原文的逐字片段、且"
        "文献标识与页码与快照记录一致的比例（micro）；没有输出引用时为 null。"
        "它度量「引用拼装 vs 冻结图谱」的自洽性，不是回源 PDF；PDF 口径见"
        " citation_pdf_verifiable。"
    ),
    "refusal_correctness": (
        "全部题中，答案路径是否为拒答与 expect_refusal 一致的比例；同时惩罚"
        "该答却拒与该拒却答。"
    ),
    "retrieval_hit_rate_excluding_unreachable": (
        "从可答题中排除「期望证据全部只存在于概念→证据内部边」的题之后，"
        "按同样口径重算的命中率。这些题在冻结图谱上没有关系边可走，图路径"
        "不可能命中；单列是为了不把它们混进普通未命中。"
    ),
    "citation_pdf_verifiable": (
        "可选口径：显式提供语料目录时，把答案引用回源 PDF 在所标页逐字核对的"
        "比例（micro）；未提供时为 null 并在 unavailable_reason 写明原因。"
        "它验证的是快照里的证据正文是否忠实于 PDF，与检索是否可重复无关。"
    ),
}


@dataclass(frozen=True, slots=True)
class UnreachableEvidence:
    """一条只挂在概念→证据内部边上、没有任何关系边引用的期望证据。"""

    evidence_id: str
    question_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotAttribution:
    """未命中归因：哪些期望证据在冻结图谱上结构性不可达。"""

    unreachable_evidence: tuple[UnreachableEvidence, ...]
    unreachable_question_ids: tuple[str, ...]
    partially_unreachable_question_ids: tuple[str, ...]
    reachable_answerable_count: int
    reachable_hits: int
    retrieval_hit_rate_excluding_unreachable: float | None


@dataclass(frozen=True, slots=True)
class PdfCitationVerification:
    """可选的 PDF 回源口径；未启用时 ``checked=False`` 且比率为 ``null``。"""

    checked: bool
    citations_total: int
    citations_verified: int
    citation_pdf_accuracy: float | None
    unavailable_reason: str | None
    failures: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SnapshotEvaluationReport:
    """一次快照评估的完整结果：来源、指标、归因与逐题明细。"""

    generated_at: str
    golden: GoldenSet
    snapshot_path: Path
    snapshot: GraphSnapshot
    base: EvaluationReport
    attribution: SnapshotAttribution
    pdf: PdfCitationVerification


def evaluate_snapshot(
    golden: GoldenSet,
    snapshot: GraphSnapshot,
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    pdf_verifier: Callable[[dict[str, Any]], CitationCheck] | None = None,
    generated_at: str | None = None,
) -> SnapshotEvaluationReport:
    """在金标准集绑定的快照上跑检索评估；不需要图数据库、模型或网络。

    ``pdf_verifier`` 是可选的回源 PDF 校验函数（通常是
    :class:`SourcePdfVerifier.verify`）。不传时只计算快照口径，并在报告里
    写明 PDF 口径未启用——不会把「没测」当成「通过」。
    """
    check_snapshot_binding(golden, snapshot, snapshot_path=snapshot_path)

    reader = SnapshotGraphReader(snapshot)
    verifier = SnapshotCitationVerifier(snapshot)
    emitted: list[dict[str, Any]] = []

    def answer_for(question: str) -> Mapping[str, Any]:
        payload = answer_to_dict(ask(question, deps=AskDeps(graph_store=reader)))
        citations = payload.get("citations")
        if isinstance(citations, list):
            emitted.extend(
                dict(citation)
                for citation in citations
                if isinstance(citation, Mapping)
            )
        return payload

    base = evaluate_golden(
        golden.questions,
        answer_for=answer_for,
        verify_citation=verifier.verify,
        generated_at=generated_at,
    )
    return SnapshotEvaluationReport(
        generated_at=base.generated_at,
        golden=golden,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        base=base,
        attribution=attribute_unreachable(golden.questions, snapshot, base.cases),
        pdf=_pdf_verification(emitted, pdf_verifier),
    )


def attribute_unreachable(
    questions: Sequence[GoldenQuestion],
    snapshot: GraphSnapshot,
    cases: Sequence[CaseResult],
) -> SnapshotAttribution:
    """按快照实际导出的关系边，算出结构性不可达的期望证据。

    可达 = 至少被一条导出的关系边引用。闭包里的证据正文可能只挂在概念→证据
    内部边上（导出不导这类边），图路径没有边可走，命中在结构上就不可能。
    """
    cited = {
        evidence_id for edge in snapshot.edges for evidence_id in edge.evidence_ids
    }
    unreachable_questions: dict[str, list[str]] = {}
    unreachable_question_ids: list[str] = []
    partially_unreachable: list[str] = []
    for question in questions:
        if question.expect_refusal:
            continue
        missing = [
            evidence_id
            for evidence_id in question.expected_evidence_ids
            if evidence_id not in cited
        ]
        if not missing:
            continue
        unreachable_questions[question.id] = missing
        if len(missing) == len(question.expected_evidence_ids):
            unreachable_question_ids.append(question.id)
        else:
            partially_unreachable.append(question.id)

    by_evidence: dict[str, list[str]] = {}
    for question_id, evidence_ids in unreachable_questions.items():
        for evidence_id in evidence_ids:
            by_evidence.setdefault(evidence_id, []).append(question_id)

    reachable_cases = [
        case
        for case in cases
        if not case.expect_refusal and case.id not in unreachable_question_ids
    ]
    reachable_hits = sum(1 for case in reachable_cases if case.hit)
    return SnapshotAttribution(
        unreachable_evidence=tuple(
            UnreachableEvidence(
                evidence_id=evidence_id,
                question_ids=tuple(sorted(question_ids)),
            )
            for evidence_id, question_ids in sorted(by_evidence.items())
        ),
        unreachable_question_ids=tuple(sorted(unreachable_question_ids)),
        partially_unreachable_question_ids=tuple(sorted(partially_unreachable)),
        reachable_answerable_count=len(reachable_cases),
        reachable_hits=reachable_hits,
        retrieval_hit_rate_excluding_unreachable=_ratio(
            reachable_hits, len(reachable_cases)
        ),
    )


def _pdf_verification(
    citations: Sequence[dict[str, Any]],
    verifier: Callable[[dict[str, Any]], CitationCheck] | None,
) -> PdfCitationVerification:
    total = len(citations)
    if verifier is None:
        return PdfCitationVerification(
            checked=False,
            citations_total=total,
            citations_verified=0,
            citation_pdf_accuracy=None,
            unavailable_reason=(
                "本次评估只读快照，未提供语料目录；快照里没有 PDF。"
                "回源 PDF 校验需要显式传 --corpus-dir"
            ),
        )
    failures: list[tuple[str, str]] = []
    for citation in citations:
        check = verifier(citation)
        if not check.ok:
            failures.append((str(citation.get("evidence_id", "")), check.reason))
    return PdfCitationVerification(
        checked=True,
        citations_total=total,
        citations_verified=total - len(failures),
        citation_pdf_accuracy=_ratio(total - len(failures), total),
        unavailable_reason=None,
        failures=tuple(failures),
    )


def snapshot_report_to_dict(report: SnapshotEvaluationReport) -> dict[str, Any]:
    """报告的 JSON 形态；``result_digest`` 让两次运行可以一眼比对。"""
    metrics = report.base.metrics
    cases = report_to_dict(report.base)["cases"]
    _annotate_cases(cases, report.attribution)

    metrics_payload: dict[str, Any] = {
        "question_count": metrics.question_count,
        "answerable_count": metrics.answerable_count,
        "retrieval_hits": metrics.retrieval_hits,
        "retrieval_hit_rate": _rounded(metrics.retrieval_hit_rate),
        "citations_total": metrics.citations_total,
        "citations_text_consistent": metrics.citations_verified,
        "citation_text_consistency": _rounded(metrics.citation_accuracy),
        "refusal_correct": metrics.refusal_correct,
        "refusal_correctness": _rounded(metrics.refusal_correctness),
        "path_counts": metrics.path_counts,
        "by_class": {
            name: {
                "total": group.total,
                "answerable": group.answerable,
                "retrieval_hits": group.retrieval_hits,
                "retrieval_hit_rate": _rounded(group.retrieval_hit_rate),
            }
            for name, group in metrics.by_class.items()
        },
    }
    attribution_payload: dict[str, Any] = {
        "unreachable_evidence": [
            {
                "evidence_id": item.evidence_id,
                "question_ids": list(item.question_ids),
            }
            for item in report.attribution.unreachable_evidence
        ],
        "unreachable_question_ids": list(report.attribution.unreachable_question_ids),
        "partially_unreachable_question_ids": list(
            report.attribution.partially_unreachable_question_ids
        ),
        "reachable_answerable_count": report.attribution.reachable_answerable_count,
        "reachable_hits": report.attribution.reachable_hits,
        "retrieval_hit_rate_excluding_unreachable": _rounded(
            report.attribution.retrieval_hit_rate_excluding_unreachable
        ),
        "note": (
            f"这 {len(report.attribution.unreachable_evidence)} 条期望证据只挂在"
            "概念→证据内部边上；快照不导出这类边，图路径不可能引用它们。"
            "正文已包含以便核对，未命中不计作普通漏检。"
        ),
    }
    pdf_payload: dict[str, Any] = {
        "checked": report.pdf.checked,
        "citations_total": report.pdf.citations_total,
        "citations_verified": report.pdf.citations_verified,
        "citation_pdf_accuracy": _rounded(report.pdf.citation_pdf_accuracy),
        "unavailable_reason": report.pdf.unavailable_reason,
        "failures": [
            {"evidence_id": evidence_id, "reason": reason}
            for evidence_id, reason in report.pdf.failures
        ],
    }

    digest = _digest(
        {
            "metrics": metrics_payload,
            "attribution": attribution_payload,
            "citation_pdf_verifiable": pdf_payload,
            "cases": cases,
        }
    )
    return {
        "schema_version": SNAPSHOT_EVALUATION_SCHEMA_VERSION,
        "kind": SNAPSHOT_EVALUATION_KIND,
        "generated_at": report.generated_at,
        "graph_frozen": True,
        "graph_label": "图谱已固定",
        "graph": {
            "mode": "snapshot",
            "format": SNAPSHOT_FORMAT,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "path": str(report.snapshot_path),
            "checksum": report.snapshot.checksum,
            "concepts": len(report.snapshot.concepts),
            "edges": len(report.snapshot.edges),
            "evidence": len(report.snapshot.evidence),
            "closure": {
                "seed_concepts": len(report.snapshot.closure.seed_concept_ids),
                "relation_types": list(report.snapshot.closure.relation_types),
                "structural_relations": list(
                    report.snapshot.closure.structural_relations
                ),
                "max_hops": report.snapshot.closure.max_hops,
            },
        },
        "golden": {
            "path": str(report.golden.path),
            "checksum": report.golden.snapshot_checksum,
            "questions": len(report.golden.questions),
        },
        "metric_definitions": SNAPSHOT_METRIC_DEFINITIONS,
        "metrics": metrics_payload,
        "attribution": attribution_payload,
        "citation_pdf_verifiable": pdf_payload,
        "result_digest": digest,
        "cases": cases,
    }


def _annotate_cases(
    cases: Sequence[dict[str, Any]], attribution: SnapshotAttribution
) -> None:
    """给逐题明细补上「这条期望证据结构上不可达」，并标出未命中类型。"""
    by_question: dict[str, list[str]] = {}
    for item in attribution.unreachable_evidence:
        for question_id in item.question_ids:
            by_question.setdefault(question_id, []).append(item.evidence_id)
    blocked = set(attribution.unreachable_question_ids)
    for case in cases:
        unreachable = by_question.get(case["id"], [])
        case["unreachable_expected_evidence"] = unreachable
        if case["hit"] is None or case["hit"]:
            case["miss_attribution"] = None
        elif case["id"] in blocked:
            case["miss_attribution"] = "expected_evidence_only_on_internal_edges"
        elif unreachable:
            case["miss_attribution"] = "partially_unreachable_expected_evidence"
        else:
            case["miss_attribution"] = "ordinary_miss"


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _digest(payload: Mapping[str, Any]) -> str:
    """对结果内容算 SHA-256；时间戳不在其中，两次运行可直接比对。"""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()

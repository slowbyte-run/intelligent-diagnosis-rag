"""eval 子命令：一条命令跑完金标准集，输出三项指标并留档报告。

流程只有三步：读金标准集 → 逐题调用 ``ask`` → 把每条引用拿回源 PDF 逐字校验。
评估本身只读：图谱只走 ``GraphReader`` 的读方法，语料只被打开读取；唯一写出
的是报告文件（默认 ``data/output/reports/``）。报告里的三项指标定义写在
``idr_rotating.evaluation.METRIC_DEFINITIONS``，随报告一起留档。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from idr_rotating import AskDeps, OntologyError, ask
from idr_rotating.answer import answer_to_dict
from idr_rotating.citation_check import SourcePdfVerifier
from idr_rotating.evaluation import (
    CaseResult,
    ClassMetrics,
    EvaluationError,
    EvaluationReport,
    GoldenSetError,
    evaluate_golden,
    load_golden_set,
    report_to_dict,
)

DEFAULT_REPORTS_DIR = Path("data/output/reports")

_PATH_LABELS = {
    "graph": "图命中",
    "vector": "向量兜底",
    "refusal": "拒答",
}


def run_eval(
    *,
    deps: AskDeps,
    corpus_dir: Path,
    golden_path: Path | None = None,
    out_path: Path | None = None,
) -> int:
    """跑完整套评估；配置或语料缺失时返回 1，指标不达标不算失败（这是度量）。"""
    try:
        questions = load_golden_set(golden_path)
        verifier = SourcePdfVerifier(corpus_dir)
    except (GoldenSetError, ValueError) as error:
        print(f"评估配置错误：{error}", file=sys.stderr)
        return 1

    try:
        report = evaluate_golden(
            questions,
            answer_for=lambda question: answer_to_dict(ask(question, deps=deps)),
            verify_citation=verifier.verify,
        )
    except (EvaluationError, OntologyError) as error:
        print(f"评估失败：{error}", file=sys.stderr)
        return 1

    target = _report_path(out_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report_to_dict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"无法写评估报告 {target}：{error}", file=sys.stderr)
        return 1

    print(render_report(report, target))
    return 0


def _report_path(out_path: Path | None) -> Path:
    if out_path is not None:
        return out_path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORTS_DIR / f"eval-{stamp}-{uuid4().hex[:8]}.json"


def render_report(report: EvaluationReport, report_path: Path) -> str:
    """人类可读摘要；报告文件才是完整明细。"""
    metrics = report.metrics
    lines = [
        f"金标准评估：{metrics.question_count} 题"
        f"（可答 {metrics.answerable_count}，拒答对照 "
        f"{metrics.question_count - metrics.answerable_count}）",
        _metric_line(
            "检索命中率",
            metrics.retrieval_hits,
            metrics.answerable_count,
            metrics.retrieval_hit_rate,
        ),
        _metric_line(
            "引用准确率",
            metrics.citations_verified,
            metrics.citations_total,
            metrics.citation_accuracy,
        ),
        _metric_line(
            "拒答正确率",
            metrics.refusal_correct,
            metrics.question_count,
            metrics.refusal_correctness,
        ),
        "按题型：" + "，".join(
            f"{name} {group.retrieval_hits}/{group.answerable}"
            for name, group in _class_labels(metrics.by_class)
        ),
        "逐题：",
    ]
    lines.extend(_case_line(case) for case in report.cases)
    lines.append(f"报告：{report_path}")
    return "\n".join(lines)


def _class_labels(
    by_class: dict[str, ClassMetrics],
) -> list[tuple[str, ClassMetrics]]:
    labels = {
        "factual": "事实型",
        "comparative": "对比型",
        "cross_document": "跨文献型",
    }
    return [(labels.get(name, name), group) for name, group in by_class.items()]


def _metric_line(label: str, passed: int, total: int, rate: float | None) -> str:
    shown = "N/A" if rate is None else f"{rate:.1%}"
    return f"{label}：{passed}/{total} = {shown}"


def _case_line(case: CaseResult) -> str:
    passed = case.refusal_correct if case.expect_refusal else bool(case.hit)
    mark = "✓" if passed else "✗"
    path = _PATH_LABELS.get(case.path, case.path)
    citations = f"引用 {case.citations_verified}/{case.citations_total}"
    if case.expect_refusal:
        verdict = "正确拒答" if passed else "应拒未拒"
    elif case.hit:
        verdict = "命中"
    else:
        reasons: list[str] = []
        if not case.concept_ok:
            reasons.append(f"缺概念 {len(case.missing_concepts)}")
        if not case.evidence_ok:
            reasons.append("期望证据未命中")
        if not case.document_ok:
            reasons.append(f"缺文献 {len(case.missing_documents)}")
        if case.citation_failures:
            reasons.append(f"引用校验失败 {len(case.citation_failures)}")
        verdict = "未命中" + (f"（{'；'.join(reasons)}）" if reasons else "")
    return f"{mark} {case.id} [{case.question_class}] {path} | {verdict} | {citations}"

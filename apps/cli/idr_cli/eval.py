"""eval 子命令：一条命令跑完金标准集，输出三项指标并留档报告。

默认路径只读提交进版本库的**图谱快照**：不连 Neo4j、不调模型、不读 PDF，
几秒内跑完，同一份快照两次给出同一组指标。金标准集记录它绑定的快照校验和，
不符时评估直接失败；报告显式标注「图谱已固定」，并把「引用文本自洽」与
「回源 PDF」拆成两个名字（快照里没有 PDF，见 ``docs/retrieval-eval.md``）。

``--live-graph`` 保留旧的实时图谱路径：从 Neo4j 与 PDF 复盘历史 50 题基线，
也用来确认提交的快照确实是线上一张图的导出。两条路径写不同格式的报告
（快照报告带结果摘要；旧路径保持原有字段），不混在一起比较。

评估本身只读：快照路径只读文件；旧路径只走 ``GraphReader`` 的读方法与语料
文件。唯一写出的是报告文件（默认 ``data/output/reports/``）。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from idr_core.graph.snapshot import SnapshotError, load_snapshot
from idr_rotating import AskDeps, OntologyError, ask
from idr_rotating.answer import answer_to_dict
from idr_rotating.citation_check import SourcePdfVerifier
from idr_rotating.evaluation import (
    CaseResult,
    ClassMetrics,
    EvaluationError,
    EvaluationReport,
    GoldenSetError,
    SnapshotBindingError,
    evaluate_golden,
    load_golden_set,
    report_to_dict,
)
from idr_rotating.snapshot_eval import (
    evaluate_snapshot,
    snapshot_report_to_dict,
)
from idr_rotating.snapshot_export import DEFAULT_SNAPSHOT_PATH

DEFAULT_REPORTS_DIR = Path("data/output/reports")

_PATH_LABELS = {
    "graph": "图命中",
    "vector": "向量兜底",
    "refusal": "拒答",
}

_MISS_LABELS = {
    "expected_evidence_only_on_internal_edges": "期望证据只存在于概念→证据内部边",
    "partially_unreachable_expected_evidence": "部分期望证据只存在于内部边",
    "ordinary_miss": "普通未命中",
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
        questions = load_golden_set(golden_path).questions
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

    target = _report_path(out_path, prefix="eval")
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


def run_snapshot_eval(
    *,
    golden_path: Path | None = None,
    snapshot_path: Path | None = None,
    out_path: Path | None = None,
    corpus_dir: Path | None = None,
) -> int:
    """在冻结快照上评估；``corpus_dir`` 给了才额外做回源 PDF 校验。

    快照路径不读配置、不连图数据库；``corpus_dir`` 只影响可选的 PDF 口径，
    它的结果单列，不改变快照口径的三项指标。
    """
    target_snapshot = (
        DEFAULT_SNAPSHOT_PATH if snapshot_path is None else Path(snapshot_path)
    )
    try:
        golden = load_golden_set(golden_path)
        snapshot = load_snapshot(target_snapshot)
        pdf_verifier = (
            None
            if corpus_dir is None
            else SourcePdfVerifier(Path(corpus_dir)).verify
        )
    except (GoldenSetError, SnapshotError, ValueError) as error:
        print(f"评估配置错误：{error}", file=sys.stderr)
        return 1

    try:
        report = evaluate_snapshot(
            golden,
            snapshot,
            snapshot_path=target_snapshot,
            pdf_verifier=pdf_verifier,
        )
    except SnapshotBindingError as error:
        print(f"评估失败：{error}", file=sys.stderr)
        return 1
    except (EvaluationError, OntologyError) as error:
        print(f"评估失败：{error}", file=sys.stderr)
        return 1

    payload = snapshot_report_to_dict(report)
    target = _report_path(out_path, prefix="eval-snapshot")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"无法写评估报告 {target}：{error}", file=sys.stderr)
        return 1

    print(render_snapshot_report(payload, target))
    return 0


def _report_path(out_path: Path | None, *, prefix: str) -> Path:
    if out_path is not None:
        return out_path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORTS_DIR / f"{prefix}-{stamp}-{uuid4().hex[:8]}.json"


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


def render_snapshot_report(payload: Mapping[str, Any], report_path: Path) -> str:
    """快照评估的人类可读摘要；每个指标都写明它度量的是什么。"""
    metrics = payload["metrics"]
    graph = payload["graph"]
    lines = [
        f"快照检索评估：{metrics['question_count']} 题"
        f"（可答 {metrics['answerable_count']}，拒答对照 "
        f"{metrics['question_count'] - metrics['answerable_count']}）",
        f"{payload['graph_label']}：{graph['checksum']}"
        f"（{graph['concepts']} 个概念 / {graph['edges']} 条关系边 / "
        f"{graph['evidence']} 段证据）",
        _metric_line(
            "检索命中率",
            metrics["retrieval_hits"],
            metrics["answerable_count"],
            metrics["retrieval_hit_rate"],
        ),
        _metric_line(
            "引用文本自洽率（快照原文逐字，非回源 PDF）",
            metrics["citations_text_consistent"],
            metrics["citations_total"],
            metrics["citation_text_consistency"],
        ),
        _metric_line(
            "拒答正确率",
            metrics["refusal_correct"],
            metrics["question_count"],
            metrics["refusal_correctness"],
        ),
        "按题型：" + "，".join(
            f"{_class_name(name)} {group['retrieval_hits']}/{group['answerable']}"
            for name, group in metrics["by_class"].items()
        ),
    ]
    attribution = payload["attribution"]
    if attribution["unreachable_evidence"]:
        affected = sorted(
            {
                question_id
                for item in attribution["unreachable_evidence"]
                for question_id in item["question_ids"]
            }
        )
        blocked = attribution["unreachable_question_ids"]
        suffix = f"（{'、'.join(blocked)}）" if blocked else ""
        lines.append(
            f"结构上无法引用的期望证据："
            f"{len(attribution['unreachable_evidence'])} 条，"
            f"涉及 {len(affected)} 道题；其中 {len(blocked)} 道题的期望证据"
            f"全部只在概念→证据内部边上{suffix}，未命中已单列"
        )
        lines.append(
            _metric_line(
                "可达口径检索命中率（排除全部不可达的题）",
                attribution["reachable_hits"],
                attribution["reachable_answerable_count"],
                attribution["retrieval_hit_rate_excluding_unreachable"],
            )
        )
    pdf = payload["citation_pdf_verifiable"]
    if pdf["checked"]:
        lines.append(
            _metric_line(
                "引用回源 PDF 准确率（另测，验证快照证据正文）",
                pdf["citations_verified"],
                pdf["citations_total"],
                pdf["citation_pdf_accuracy"],
            )
        )
    else:
        lines.append(f"回源 PDF：未校验（{pdf['unavailable_reason']}）")
    lines.append(
        f"结果摘要：{payload['result_digest']}（同一快照两次运行应完全相同）"
    )
    lines.append("逐题：")
    lines.extend(_snapshot_case_line(case) for case in payload["cases"])
    lines.append(f"报告：{report_path}")
    return "\n".join(lines)


def _class_name(name: str) -> str:
    return {
        "factual": "事实型",
        "comparative": "对比型",
        "cross_document": "跨文献型",
    }.get(name, name)


def _snapshot_case_line(case: Mapping[str, Any]) -> str:
    passed = (
        case["refusal_correct"] if case["expect_refusal"] else bool(case["hit"])
    )
    mark = "✓" if passed else "✗"
    path = _PATH_LABELS.get(case["path"], case["path"])
    citations = f"引用 {case['citations']['verified']}/{case['citations']['total']}"
    if case["expect_refusal"]:
        verdict = "正确拒答" if passed else "应拒未拒"
    elif case["hit"]:
        verdict = "命中"
    else:
        reasons: list[str] = []
        if not case["concept_ok"]:
            reasons.append(f"缺概念 {len(case['missing_concepts'])}")
        if not case["evidence_ok"]:
            reasons.append(
                _MISS_LABELS.get(case["miss_attribution"], "期望证据未命中")
            )
        if not case["document_ok"]:
            reasons.append(f"缺文献 {len(case['missing_documents'])}")
        if case["citations"]["failures"]:
            reasons.append(f"引用校验失败 {len(case['citations']['failures'])}")
        verdict = "未命中" + (f"（{'；'.join(reasons)}）" if reasons else "")
    return f"{mark} {case['id']} [{case['class']}] {path} | {verdict} | {citations}"

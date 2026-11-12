"""recall 子命令：只量向量召回，不重排、不过阈值。

这是 #20 的诊断工具：它回答「绑定的期望证据进没进 top-k」，而不是「答案对
不对」。每道题一次嵌入、一次 pgvector 检索，因此可以反复跑；重排与阈值都不
参与，召回的损失不会和分数问题混在一起。

默认跑两个变体：``raw``（原始问句）与 ``configured``（``retrieval.yaml`` 当前
的查询改写）。其余变体（``enriched``、``instruct:<预设>``、
``enriched+instruct:<预设>``）用于比较；报告只写数字，结论由读报告的人下。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from idr_rotating import AskDeps
from idr_rotating.evaluation import GoldenQuestion, GoldenSetError, load_golden_set
from idr_rotating.ontology import Ontology, load_ontology
from idr_rotating.query_rewrite import (
    INSTRUCTION_PRESETS,
    apply_instruction,
    build_retrieval_query,
    enrich_question,
    lexical_terms,
)
from idr_rotating.recall_diagnostic import (
    DEFAULT_MAX_K,
    DEFAULT_RECALL_KS,
    RecallReport,
    diagnose_recall,
    recall_report_to_dict,
)
from idr_rotating.retrieval import (
    RetrievalSettings,
    RetrievalSettingsError,
    load_retrieval_settings,
)

DEFAULT_REPORTS_DIR = Path("data/output/reports")
DEFAULT_VARIANTS = "raw,configured"


def run_recall(
    *,
    deps: AskDeps,
    golden_path: Path | None = None,
    out_path: Path | None = None,
    top_k: int = DEFAULT_MAX_K,
    variants: str = DEFAULT_VARIANTS,
    lexical: bool = False,
) -> int:
    """对每个查询变体跑一遍召回诊断，打印摘要并写 JSON 报告。

    ``lexical=True`` 时把词面候选与向量候选按 RRF 融合后再统计，用来核对
    线上 ``lexical_recall`` 的收益；融合同样不经过重排。
    """
    if deps.embedder is None or deps.vector_store is None:
        print(
            "诊断失败：recall 需要嵌入与向量检索依赖，当前配置缺其中至少一个",
            file=sys.stderr,
        )
        return 1
    try:
        questions = load_golden_set(golden_path)
        settings = deps.retrieval_settings or load_retrieval_settings()
        ontology = load_ontology()
        builders = _query_builders(
            _parse_variants(variants), ontology=ontology, settings=settings
        )
        ks = tuple(k for k in DEFAULT_RECALL_KS if k <= top_k)
        if not ks:
            raise ValueError(
                f"top-k={top_k} 小于最小的报告口径 recall@{min(DEFAULT_RECALL_KS)}"
            )
        if lexical and deps.text_search is None:
            raise ValueError("词面诊断需要 text_search 依赖，当前配置没有注入")
    except (GoldenSetError, RetrievalSettingsError, ValueError) as error:
        print(f"诊断配置错误：{error}", file=sys.stderr)
        return 1

    def lexical_for(question: GoldenQuestion) -> tuple[str, ...]:
        if deps.text_search is None:
            return ()
        terms = lexical_terms(question.question, ontology)
        if not terms:
            return ()
        return deps.text_search.search_evidence(terms, limit=top_k)

    reports: list[RecallReport] = []
    for name, query_for in builders.items():
        report, _ = diagnose_recall(
            questions,
            embedder=deps.embedder,
            vector_store=deps.vector_store,
            query_for=query_for,
            variant=f"{name}+lexical" if lexical else name,
            max_k=top_k,
            ks=ks,
            lexical_for=lexical_for if lexical else None,
        )
        reports.append(report)

    target = _report_path(out_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(
                        timespec="seconds"
                    ),
                    "question_count": len(questions),
                    "answerable_count": sum(
                        1 for question in questions if not question.expect_refusal
                    ),
                    "variants": [
                        recall_report_to_dict(report) for report in reports
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"无法写召回报告 {target}：{error}", file=sys.stderr)
        return 1

    print(render_recall(reports, target))
    return 0


def _parse_variants(spec: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip() for part in spec.split(",") if part.strip()))
    if not names:
        raise ValueError("至少需要一个查询变体")
    return names


def _query_builders(
    names: tuple[str, ...],
    *,
    ontology: Ontology,
    settings: RetrievalSettings,
) -> dict[str, Callable[[GoldenQuestion], str]]:
    """把变体名映射成查询构造函数；名字拼错在跑模型调用之前就失败。"""
    builders: dict[str, Callable[[GoldenQuestion], str]] = {}
    for name in names:
        if name == "raw":
            builders[name] = lambda question: question.question
        elif name == "enriched":
            builders[name] = lambda question: enrich_question(
                question.question, ontology
            )
        elif name == "configured":
            builders[name] = lambda question: build_retrieval_query(
                question.question, settings=settings, ontology=ontology
            )
        elif name.startswith("instruct:"):
            builders[name] = _instructed(name)
        elif name.startswith("enriched+instruct:"):
            builders[name] = _enriched_instructed(name, ontology)
        else:
            raise ValueError(f"未知的查询变体：{name}")
    return builders


def _instructed(name: str) -> Callable[[GoldenQuestion], str]:
    preset = name.removeprefix("instruct:")
    _check_preset(preset)
    return lambda question: apply_instruction(question.question, preset)


def _enriched_instructed(
    name: str, ontology: Ontology
) -> Callable[[GoldenQuestion], str]:
    preset = name.removeprefix("enriched+instruct:")
    _check_preset(preset)
    return lambda question: apply_instruction(
        enrich_question(question.question, ontology), preset
    )


def _check_preset(preset: str) -> None:
    if preset not in INSTRUCTION_PRESETS:
        choices = "、".join(sorted(INSTRUCTION_PRESETS))
        raise ValueError(f"未知的检索指令预设：{preset!r}（可选：{choices}）")


def _report_path(out_path: Path | None) -> Path:
    if out_path is not None:
        return out_path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORTS_DIR / f"recall-{stamp}-{uuid4().hex[:8]}.json"


def render_recall(reports: list[RecallReport], report_path: Path) -> str:
    """人类可读摘要：每个变体一行 recall，逐题只列没进前 k 的绑定证据。"""
    lines: list[str] = []
    for report in reports:
        lines.append(f"查询变体：{report.variant}")
        for item in report.summary:
            rate = "N/A" if item.rate is None else f"{item.rate:.1%}"
            lines.append(
                f"  recall@{item.k}：{item.hits}/{item.answerable} = {rate}"
            )
        misses = [
            case
            for case in report.cases
            if not case.expect_refusal and not case.hit_at(report.ks[0])
        ]
        lines.append(
            f"  未进 recall@{report.ks[0]} 的题（最近绑定证据名次）："
        )
        if not misses:
            lines.append("    无")
        for case in misses:
            rank = "未召回" if case.nearest_rank is None else f"第 {case.nearest_rank} 名"
            lines.append(f"    {case.question_id} [{case.question_class}] {rank}")
        lines.append("")
    lines.append(f"报告：{report_path}")
    return "\n".join(lines)

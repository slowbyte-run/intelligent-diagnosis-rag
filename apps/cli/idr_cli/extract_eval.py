"""``idr extract-eval``：在真实语料标注集上重复评估抽取，报分布而不是单点。

单次评估只是一次抽样：同一输入重复抽取，产出会变，召回率与精度也跟着变。
这个命令默认重复 3 次，报告每次运行的分数、命中的标注断言集合的两两重合度，
以及召回率与精度的极差。标注集是领域资产；真实模型调用按文献×次数计，
``--limit`` 控制文献数，``--runs`` 控制重复次数。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idr_core.config import Config
from idr_core.llm import LLMClient
from idr_rotating.extraction import ExtractionError
from idr_rotating.extraction_distribution import (
    EvaluationRun,
    ExtractionDistribution,
    measure_extraction_distribution,
)
from idr_rotating.extraction_evaluation import (
    DEFAULT_ANNOTATIONS_PATH,
    AnnotationSet,
    AnnotationSetError,
    ExtractionScore,
    load_annotations,
)
from idr_rotating.ontology import load_ontology
from idr_rotating.variance import MetricSpread

EXTRACTION_REPORT_SCHEMA_VERSION = 2

METRIC_DEFINITIONS = {
    "recall": (
        "期望断言（实体 name+type；关系 head+type+tail）出现在输出中的比例；"
        "分母是去重后的期望断言数。"
    ),
    "precision": (
        "输出断言中与标注完全一致的比例；分母是去重后的输出断言数，"
        "输出里多出来的断言都算精度损失。"
    ),
    "missing": "期望断言未命中，且同名实体或同端点的关系完全没有出现的条数。",
    "type_mismatch": "同名实体或同端点关系出现了，但类型不符的条数。",
    "endpoint_mismatch": "关系类型相同，但头端或尾端与标注不符的条数。",
    "overlap": (
        "两次运行各自命中的标注断言集合的 Jaccard 系数；"
        "衡量两次运行是否抽到了同一批期望断言。"
    ),
    "range": "多次运行中该指标的最大值减最小值，即极差。",
}


def run_extract_eval(
    *,
    llm: LLMClient,
    annotations_path: Path | None = None,
    limit: int | None = None,
    runs: int = 3,
    out: Path | None = None,
    model: str = "unknown",
) -> int:
    """加载标注集、重复抽取并评分；格式错误与解析失败都给出可读的错误。"""
    ontology = load_ontology()
    try:
        annotations = load_annotations(ontology, annotations_path)
    except AnnotationSetError as error:
        print(f"标注集错误：{error}", file=sys.stderr)
        return 1

    if runs < 2:
        print("重复次数至少为 2，否则没有分布可言", file=sys.stderr)
        return 1

    if limit is not None:
        if limit < 1:
            print("--limit 必须是正整数", file=sys.stderr)
            return 1
        annotations = annotations.with_documents(annotations.documents()[:limit])

    try:
        distribution = measure_extraction_distribution(
            llm, ontology, annotations, runs=runs
        )
    except ExtractionError as error:
        print(f"抽取失败：{error}", file=sys.stderr)
        return 1

    _print_report(annotations, distribution, model)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                _report_dict(annotations, distribution, model),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告：{out}")
    return 0


def _print_report(
    annotations: AnnotationSet, distribution: ExtractionDistribution, model: str
) -> None:
    print(f"标注集：{annotations.name or DEFAULT_ANNOTATIONS_PATH.name}")
    print(f"模型：{model}")
    print(
        f"文献：{len(annotations.documents())}  "
        f"证据段：{len(annotations.segments)}  "
        f"标注：实体 {len(annotations.expected_entities)} 条 / "
        f"关系 {len(annotations.expected_relations)} 条"
    )
    print("（同文献内同名同类型断言去重后参与评分）")
    print(
        f"重复次数：{len(distribution.runs)}"
        "（单次运行只是一次抽样，不要当能力估计）"
    )
    print()
    print("每次运行：")
    for run in distribution.runs:
        score = run.score
        print(
            f"  #{run.run}  期望 {score.expected_assertions:>3}  "
            f"输出 {score.output_assertions:>3}  "
            f"命中 {score.matched_assertions:>3}  "
            f"召回 {_percent(run.recall)}  精度 {_percent(run.precision)}  "
            f"漏抽 {score.missing_entities + score.missing_relations}  "
            f"类型不符 "
            f"{score.type_mismatch_entities + score.type_mismatch_relations}  "
            f"端点不符 {score.endpoint_mismatch_relations}"
        )

    print()
    print(f"召回率：{_spread_line(distribution.recall)}")
    print(f"精度：{_spread_line(distribution.precision)}")
    print(f"命中标注集合两两重合（Jaccard 均值）：{_overlap_text(distribution)}")
    expected = distribution.runs[0].score.expected_assertions
    print(
        f"每次运行都命中的断言：{len(distribution.consistent_hits)}/{expected}"
        f"（{_percent(_ratio(len(distribution.consistent_hits), expected))}）"
    )
    print(
        f"至少一次命中的断言：{len(distribution.ever_hits)}/{expected}"
        f"（{_percent(_ratio(len(distribution.ever_hits), expected))}）"
    )
    print(
        "所有运行的命中集合完全一致："
        f"{'是' if distribution.identical_hits else '否'}"
    )


def _spread_line(spread: MetricSpread) -> str:
    if spread.spread is None:
        return "—（有运行的分数没有分母，无法比较极差）"
    return (
        f"最低 {_percent(spread.minimum)}  最高 {_percent(spread.maximum)}  "
        f"极差 {spread.spread * 100:.1f} 个百分点"
    )


def _overlap_text(distribution: ExtractionDistribution) -> str:
    mean = distribution.mean_overlap
    if mean is None:
        return "—"
    return f"{mean:.3f}（{len(distribution.pairwise_overlap)} 对）"


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _report_dict(
    annotations: AnnotationSet,
    distribution: ExtractionDistribution,
    model: str,
) -> dict[str, Any]:
    expected = distribution.runs[0].score.expected_assertions
    return {
        "schema_version": EXTRACTION_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "runs": len(distribution.runs),
        "annotation_set": {
            "name": annotations.name,
            "segments": len(annotations.segments),
            "documents": len(annotations.documents()),
            "annotated_entities": len(annotations.expected_entities),
            "annotated_relations": len(annotations.expected_relations),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "distribution": {
            "recall": _spread_dict(distribution.recall),
            "precision": _spread_dict(distribution.precision),
            "pairwise_overlap": [
                _rounded(value) for value in distribution.pairwise_overlap
            ],
            "mean_overlap": _rounded(distribution.mean_overlap),
            "identical_hits": distribution.identical_hits,
            "expected_assertions": expected,
            "consistent_hit_assertions": len(distribution.consistent_hits),
            "ever_hit_assertions": len(distribution.ever_hits),
            "flaky_hit_assertions": len(distribution.flaky_hits),
        },
        "run_details": [_run_dict(run) for run in distribution.runs],
    }


def _run_dict(run: EvaluationRun) -> dict[str, Any]:
    return {
        "run": run.run,
        "metrics": _metrics_dict(run.score),
        "by_document": [
            {
                "doc_id": document.doc_id,
                "expected_entities": document.expected_entities,
                "expected_relations": document.expected_relations,
                "output_entities": document.output_entities,
                "output_relations": document.output_relations,
                "matched_entities": document.matched_entities,
                "matched_relations": document.matched_relations,
            }
            for document in run.score.documents
        ],
        "hit_assertions": sorted(run.hit_keys),
        "cases": [
            {
                "kind": outcome.kind,
                "evidence_id": outcome.evidence_id,
                "label": outcome.label,
                "status": outcome.status,
                "detail": outcome.detail,
            }
            for outcome in run.score.outcomes
        ],
        "extras": [
            {
                "kind": extra.kind,
                "doc_id": extra.doc_id,
                "label": extra.label,
                "evidence_ids": list(extra.evidence_ids),
            }
            for extra in run.score.extras
        ],
    }


def _metrics_dict(score: ExtractionScore) -> dict[str, Any]:
    return {
        "expected_entities": score.expected_entities,
        "expected_relations": score.expected_relations,
        "expected_assertions": score.expected_assertions,
        "output_entities": score.output_entities,
        "output_relations": score.output_relations,
        "output_assertions": score.output_assertions,
        "matched_entities": score.matched_entities,
        "matched_relations": score.matched_relations,
        "matched_assertions": score.matched_assertions,
        "recall": _rounded(score.recall),
        "precision": _rounded(score.precision),
        "missing_entities": score.missing_entities,
        "missing_relations": score.missing_relations,
        "type_mismatch_entities": score.type_mismatch_entities,
        "type_mismatch_relations": score.type_mismatch_relations,
        "endpoint_mismatch_relations": score.endpoint_mismatch_relations,
        "extra_entities": score.extra_entities,
        "extra_relations": score.extra_relations,
    }


def _spread_dict(spread: MetricSpread) -> dict[str, Any]:
    return {
        "values": [_rounded(value) for value in spread.values],
        "min": _rounded(spread.minimum),
        "max": _rounded(spread.maximum),
        "range": _rounded(spread.spread),
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def build_extract_eval_llm(config: Config) -> LLMClient:
    """真实模型客户端；与入库用的是同一套参数。"""
    from idr_core.llm.openai_compat import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
        model=config.deepseek_model,
        extra_headers=config.llm_extra_headers,
    )

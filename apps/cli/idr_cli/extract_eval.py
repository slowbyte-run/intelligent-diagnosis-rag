"""``idr extract-eval``：在真实语料标注集上评估抽取，不碰图谱与检索。

输出断言级召回率与精度，并把漏抽、类型不符、端点不符分开计数。标注集是领域
资产；真实模型调用按文献计，``--limit`` 可以只跑前几篇控制成本。
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
from idr_rotating.extraction_evaluation import (
    DEFAULT_ANNOTATIONS_PATH,
    AnnotationSetError,
    AnnotationSet,
    ExtractionScore,
    load_annotations,
    evaluate_extraction,
)
from idr_rotating.ontology import load_ontology

EXTRACTION_REPORT_SCHEMA_VERSION = 1

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
}


def run_extract_eval(
    *,
    llm: LLMClient,
    annotations_path: Path | None = None,
    limit: int | None = None,
    out: Path | None = None,
    model: str = "unknown",
) -> int:
    """加载标注集、逐篇抽取并评分；格式错误与解析失败都给出可读的错误。"""
    ontology = load_ontology()
    try:
        annotations = load_annotations(ontology, annotations_path)
    except AnnotationSetError as error:
        print(f"标注集错误：{error}", file=sys.stderr)
        return 1

    if limit is not None:
        if limit < 1:
            print("--limit 必须是正整数", file=sys.stderr)
            return 1
        annotations = annotations.with_documents(annotations.documents()[:limit])

    try:
        score = evaluate_extraction(llm, ontology, annotations)
    except ExtractionError as error:
        print(f"抽取失败：{error}", file=sys.stderr)
        return 1

    _print_report(annotations, score, model)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                _report_dict(annotations, score, model),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告：{out}")
    return 0


def _print_report(
    annotations: AnnotationSet, score: ExtractionScore, model: str
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
    print()
    print(
        f"期望断言：实体 {score.expected_entities}  "
        f"关系 {score.expected_relations}  合计 {score.expected_assertions}"
    )
    print(
        f"输出断言：实体 {score.output_entities}  "
        f"关系 {score.output_relations}  合计 {score.output_assertions}"
    )
    print(
        f"断言级召回率：{_percent(score.recall)}"
        f"（命中 {score.matched_assertions} / 期望 {score.expected_assertions}）"
    )
    print(
        f"断言级精度：{_percent(score.precision)}"
        f"（命中 {score.matched_assertions} / 输出 {score.output_assertions}）"
    )
    print()
    print(
        f"漏抽：实体 {score.missing_entities}  关系 {score.missing_relations}"
    )
    print(
        f"类型不符：实体 {score.type_mismatch_entities}  "
        f"关系 {score.type_mismatch_relations}"
    )
    print(f"端点不符：关系 {score.endpoint_mismatch_relations}")
    print(f"精度损失（输出多出的断言）：实体 {score.extra_entities}  关系 {score.extra_relations}")
    print()

    print("按文献：")
    for document in score.documents:
        expected = document.expected_entities + document.expected_relations
        output = document.output_entities + document.output_relations
        matched = document.matched_entities + document.matched_relations
        recall = None if expected == 0 else matched / expected
        precision = None if output == 0 else matched / output
        print(
            f"  {document.doc_id}  期望 {expected:>3}  输出 {output:>3}  "
            f"命中 {matched:>3}  召回 {_percent(recall)}  精度 {_percent(precision)}"
        )

    if score.outcomes:
        misses = [
            outcome
            for outcome in score.outcomes
            if outcome.status != "matched"
        ]
        if misses:
            print()
            print(f"未命中断言（{len(misses)} 条，最多显示 20 条）：")
            for outcome in misses[:20]:
                print(
                    f"  [{_status_label(outcome.status)}] {outcome.evidence_id}  "
                    f"{outcome.label}"
                    + (f"（{outcome.detail}）" if outcome.detail else "")
                )
            if len(misses) > 20:
                print(f"  ……其余 {len(misses) - 20} 条见报告 JSON")


def _status_label(status: str) -> str:
    return {
        "missing": "漏抽",
        "type_mismatch": "类型不符",
        "endpoint_mismatch": "端点不符",
    }.get(status, status)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _report_dict(
    annotations: AnnotationSet, score: ExtractionScore, model: str
) -> dict[str, Any]:
    return {
        "schema_version": EXTRACTION_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "annotation_set": {
            "name": annotations.name,
            "segments": len(annotations.segments),
            "documents": len(annotations.documents()),
            "annotated_entities": len(annotations.expected_entities),
            "annotated_relations": len(annotations.expected_relations),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "metrics": {
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
        },
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
            for document in score.documents
        ],
        "cases": [
            {
                "kind": outcome.kind,
                "evidence_id": outcome.evidence_id,
                "label": outcome.label,
                "status": outcome.status,
                "detail": outcome.detail,
            }
            for outcome in score.outcomes
        ],
        "extras": [
            {
                "kind": extra.kind,
                "doc_id": extra.doc_id,
                "label": extra.label,
                "evidence_ids": list(extra.evidence_ids),
            }
            for extra in score.extras
        ],
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

"""calibrate 子命令：用金标准集校准向量兜底的拒答阈值。

采样只有一遍：每道金标准题强制走向量路径（阈值临时设为 0），拿回全部候选的
重排分数；随后在内存里对同一份分数扫所有候选阈值，不按阈值重复调用模型。
采样本身会花 2 次模型调用/题（嵌入 + 重排），因此这是维护者命令，不是线上
请求路径的一部分。

推荐值与理由由 ``idr_rotating.calibration`` 计算，本模块只负责接服务、留档与
可选的配置写回。``--write`` 写回数值和校准标记两行，注释与其余键保持不变。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from idr_rotating import AskDeps
from idr_rotating.calibration import (
    CalibrationError,
    CalibrationResult,
    VectorSample,
    calibration_report_to_dict,
    recommend_threshold,
    write_threshold,
)
from idr_rotating.evaluation import (
    GoldenQuestion,
    GoldenSetError,
    load_golden_set,
)
from idr_rotating.retrieval import (
    DEFAULT_RETRIEVAL_PATH,
    RetrievalSettings,
    RetrievalSettingsError,
    load_retrieval_settings,
)
from idr_rotating.vector_recall import recall_evidence

DEFAULT_REPORTS_DIR = Path("data/output/reports")

Sampler = Callable[
    [Sequence[GoldenQuestion], AskDeps, RetrievalSettings],
    tuple[VectorSample, ...],
]


def collect_samples(
    questions: Sequence[GoldenQuestion],
    deps: AskDeps,
    settings: RetrievalSettings,
) -> tuple[VectorSample, ...]:
    """每题跑一遍「嵌入 + 向量召回 + 重排」，把全部候选与分数留下来。

    采样时的阈值临时设为 0，且把结果上限放宽到召回数：校准要看到完整的分数
    分布，不能被当前配置的阈值或条数截断。
    """
    if (
        deps.embedder is None
        or deps.vector_store is None
        or deps.reranker is None
    ):
        raise CalibrationError(
            "校准需要向量兜底依赖：嵌入、向量检索与重排三者都要注入"
        )
    scan_settings = settings.model_copy(
        update={
            "vector_min_rerank_score": 0.0,
            "vector_max_results": settings.vector_top_k,
        }
    )
    samples: list[VectorSample] = []
    for question in questions:
        recall = recall_evidence(
            question.question,
            embedder=deps.embedder,
            vector_store=deps.vector_store,
            reranker=deps.reranker,
            graph_store=deps.graph_store,
            settings=scan_settings,
        )
        samples.append(
            VectorSample(
                question_id=question.id,
                expect_refusal=question.expect_refusal,
                expected_evidence_ids=question.expected_evidence_ids,
                scored=tuple(
                    (item.segment.id, item.score) for item in recall.segments
                ),
            )
        )
    return tuple(samples)


def run_calibrate(
    *,
    deps: AskDeps,
    golden_path: Path | None = None,
    out_path: Path | None = None,
    settings_path: Path | None = None,
    write_config: bool = False,
    sampler: Sampler = collect_samples,
) -> int:
    """采样、扫描、留档；``--write`` 时把推荐阈值写回检索配置。"""
    try:
        questions = load_golden_set(golden_path)
        settings = deps.retrieval_settings or load_retrieval_settings(settings_path)
    except (GoldenSetError, RetrievalSettingsError) as error:
        print(f"校准配置错误：{error}", file=sys.stderr)
        return 1

    try:
        samples = sampler(questions, deps, settings)
        result = recommend_threshold(samples)
    except CalibrationError as error:
        print(f"校准失败：{error}", file=sys.stderr)
        return 1

    target = _report_path(out_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(
                        timespec="seconds"
                    ),
                    "settings_file": str(
                        settings_path or DEFAULT_RETRIEVAL_PATH
                    ),
                    **calibration_report_to_dict(samples, result),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"无法写校准报告 {target}：{error}", file=sys.stderr)
        return 1

    print(render_calibration(result, target))

    if write_config:
        config_path = settings_path or DEFAULT_RETRIEVAL_PATH
        try:
            written = write_threshold(config_path, result.threshold)
        except (CalibrationError, RetrievalSettingsError) as error:
            print(f"无法写回检索配置：{error}", file=sys.stderr)
            return 1
        print(
            f"已写入 {config_path}：vector_min_rerank_score="
            f"{written.vector_min_rerank_score:.2f}，"
            "vector_min_rerank_score_calibrated=true"
        )
    return 0


def _report_path(out_path: Path | None) -> Path:
    if out_path is not None:
        return out_path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORTS_DIR / f"calibration-{stamp}-{uuid4().hex[:8]}.json"


def render_calibration(result: CalibrationResult, report_path: Path) -> str:
    """人类可读摘要：推荐值、被取舍掉的东西、完整扫描表。"""
    point = result.recommended
    lines = [
        "阈值校准（向量兜底，金标准集）",
        f"推荐阈值：{point.threshold:.2f}",
        f"  拒答对照：{point.refusal_correct}/{point.refusal_total} 正确拒答",
        f"  期望证据：{point.vector_hits}/{point.answerable_total} 保留",
        f"  有候选可答：{point.answered}/{point.answerable_total}",
        f"  离最近观测分数：{point.margin:.3f}",
        "选择理由：",
    ]
    lines.extend(f"  - {note}" for note in result.notes)
    lines.append(
        "扫描表（推荐 / 阈值 / 拒答 / 期望证据 / 有候选 / 余量；"
        "阈值是每个分数空档的代表点，不是全部分数）："
    )
    for candidate in result.table:
        mark = "→" if candidate.threshold == point.threshold else " "
        lines.append(
            f"{mark} {candidate.threshold:.3f} | "
            f"{candidate.refusal_correct}/{candidate.refusal_total} | "
            f"{candidate.vector_hits}/{candidate.answerable_total} | "
            f"{candidate.answered}/{candidate.answerable_total} | "
            f"{candidate.margin:.3f}"
        )
    lines.append(f"报告：{report_path}")
    return "\n".join(lines)

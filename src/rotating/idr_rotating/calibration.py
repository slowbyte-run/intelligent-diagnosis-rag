"""阈值校准：把金标准集的重排分数分布变成一个可解释、可复现的阈值。

校准只做两件事：先采样（每道题强制走向量路径，拿到全部候选的分数），再离线扫
阈值。评分只用金标准集里已经绑定好的证据段标识，不引入裁判模型；阈值扫描是
一次性的，不会为每个候选阈值重跑一遍检索。

选择规则是**拒绝优先**：

1. 硬约束：阈值必须高于每一道拒答对照题的最高分，否则「该拒却答」；
2. 稳定性：阈值离最近的**关键分数**至少 ``STABILITY_MARGIN``。关键分数只有
   两类：每道题的最高分（决定「还有没有候选」）与可答题绑定证据段的分数
   （决定「期望证据保不保留」）；其余候选分数不改变结论，不参与切分。重排
   分数在重复调用之间会抖动，贴着边界的点下次调用就可能翻面；
3. 在满足前两条的点里，选期望证据保留最多、离关键分数最远的点。

如果没有任何阈值能同时完成这两件事，结果里写明被牺牲的是哪一个目标——单阈值
就是分不开时，不假装分得开。

本模块只做纯计算与配置文本写回，不连任何服务；采样在命令行层完成。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from idr_rotating.retrieval import RetrievalSettings, load_retrieval_settings

CALIBRATION_SCHEMA_VERSION = 1

# 阈值与最近关键分数之间的最小余量。重排服务是生成式模型，同一对「查询—证据」
# 在重复调用里会出现约 0.01 的分数抖动；取两倍作为稳定边界。
STABILITY_MARGIN = 0.02

_SCORE_LINE = re.compile(
    r"^(vector_min_rerank_score:[ \t]*)([0-9]*\.?[0-9]+)([^\n]*)$", re.MULTILINE
)
_CALIBRATED_LINE = re.compile(
    r"^(vector_min_rerank_score_calibrated:[ \t]*)(true|false)([^\n]*)$",
    re.MULTILINE,
)


class CalibrationError(ValueError):
    """校准输入或配置写回不合法；消息面向人阅读。"""


@dataclass(frozen=True, slots=True)
class VectorSample:
    """一道金标准题在向量兜底路径上的全部候选与分数。

    ``scored`` 是一遍召回 + 重排后的全部候选（不过阈值），顺序不重要；校准在
    这上面离线扫任意阈值，因此不需要为每个阈值重复调用模型。
    """

    question_id: str
    expect_refusal: bool
    expected_evidence_ids: tuple[str, ...]
    scored: tuple[tuple[str, float], ...]

    @property
    def top_score(self) -> float | None:
        """最高重排分；没有召回候选时为 ``None``。"""
        return max((score for _, score in self.scored), default=None)


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """一个候选阈值在整张金标准集上的后果。"""

    threshold: float
    # 阈值到最近关键重排分数的距离；越大越不容易被下一次调用的抖动翻面。
    margin: float
    refusal_correct: int
    refusal_total: int
    # 期望证据仍在阈值之上的可答题数，与评估口径一致：至少命中一个绑定证据段。
    vector_hits: int
    answerable_total: int
    # 阈值之上至少有一条候选的可答题数；衡量「向量路径还答不答得出来」。
    answered: int

    @property
    def refusal_complete(self) -> bool:
        return self.refusal_correct == self.refusal_total


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """一次校准的推荐点、完整扫描表与选择理由。"""

    recommended: ThresholdPoint
    table: tuple[ThresholdPoint, ...]
    notes: tuple[str, ...]
    stability_margin: float = STABILITY_MARGIN

    @property
    def threshold(self) -> float:
        return self.recommended.threshold

    @property
    def all_refusals_correct(self) -> bool:
        return self.recommended.refusal_complete


def recommend_threshold(
    samples: Sequence[VectorSample],
    *,
    stability_margin: float = STABILITY_MARGIN,
) -> CalibrationResult:
    """扫描金标准集的分数分布，返回推荐阈值、完整结果表与选择理由。"""
    if not samples:
        raise CalibrationError("没有校准样本：至少需要一道题的重排结果")
    if not any(item.expect_refusal for item in samples):
        raise CalibrationError("没有拒答对照题：无法确定阈值上界")
    if not any(not item.expect_refusal for item in samples):
        raise CalibrationError("没有可答题：无法评估阈值保留了多少证据")
    if stability_margin < 0.0:
        raise CalibrationError("稳定性余量不能为负数")

    scores = _critical_scores(samples)
    if not scores:
        raise CalibrationError("所有题目都没有重排分数：无法校准阈值")

    points = tuple(
        _evaluate(threshold, samples, scores)
        for threshold in _candidate_thresholds(scores)
    )
    feasible = [point for point in points if point.refusal_complete]
    stable = [point for point in feasible if point.margin >= stability_margin]

    if stable:
        recommended = max(stable, key=_preference)
    elif feasible:
        # 没有满足稳定余量的点：仍以拒答为硬约束，再尽量多留证据。
        recommended = max(feasible, key=_preference)
    else:
        recommended = max(points, key=_infeasible_preference)

    return CalibrationResult(
        recommended=recommended,
        table=tuple(sorted(points, key=lambda point: -point.threshold)),
        notes=_notes(
            samples,
            points,
            recommended,
            feasible=bool(feasible),
            stability_margin=stability_margin,
        ),
        stability_margin=stability_margin,
    )


def _preference(point: ThresholdPoint) -> tuple[int, float, float]:
    """稳定可行点之间的偏好：先多留证据，再离关键分数更远，最后取更低阈值。"""
    return (point.vector_hits, point.margin, -point.threshold)


def _infeasible_preference(point: ThresholdPoint) -> tuple[int, int, float, float]:
    """连拒答都做不到时的最不坏选择：先多拒答，再少误伤可答题。"""
    return (
        point.refusal_correct,
        point.vector_hits,
        point.margin,
        -point.threshold,
    )


def _critical_scores(samples: Sequence[VectorSample]) -> list[float]:
    """只保留「会改变校准结论」的分数，作为候选阈值与余量的边界。

    一道题的 20 条候选里大多数分数对结论没有影响：能改变结论的只有它的最高分
    （决定「还有没有候选」）与它绑定的证据段分数（决定「期望证据保不保留」）。
    把每条候选都当边界会让几百个低分噪声切碎分数轴，稳定余量永远凑不满。
    """
    scores: set[float] = set()
    for item in samples:
        for evidence_id, score in item.scored:
            if not math.isfinite(score):
                raise CalibrationError(
                    f"{item.question_id} 的候选 {evidence_id} 分数不是有限数值：{score}"
                )
        top = item.top_score
        if top is not None:
            scores.add(float(top))
        if item.expect_refusal:
            continue
        expected = set(item.expected_evidence_ids)
        scores.update(
            float(score)
            for evidence_id, score in item.scored
            if evidence_id in expected
        )
    return sorted(scores)


def _candidate_thresholds(scores: Sequence[float]) -> tuple[float, ...]:
    """取每个「行为区间」的代表点。

    阈值 ``t`` 的行为只由「有哪些分数 >= t」决定，因此区间
    ``(s[i], s[i+1]]`` 里任意一点等价；取中点让离两侧观测分数的余量最大。
    最上面与最下面的区间也取中点，除非它贴着 0/1 边界。
    """
    candidates: list[float] = []
    if scores[0] > 0.0:
        candidates.append(_representative(0.0, scores[0]))
    for lower, upper in zip(scores, scores[1:]):
        if upper > lower:
            candidates.append(_representative(lower, upper))
    if scores[-1] < 1.0:
        candidates.append(_representative(scores[-1], 1.0))
    return tuple(dict.fromkeys(candidates))


def _representative(lower: float, upper: float) -> float:
    """区间内的代表阈值，保留两位小数且不越出区间。"""
    midpoint = (lower + upper) / 2
    for digits in (2, 4):
        rounded = round(midpoint, digits)
        if lower < rounded <= upper:
            return rounded
    return midpoint


def _evaluate(
    threshold: float, samples: Sequence[VectorSample], scores: Sequence[float]
) -> ThresholdPoint:
    refusal_correct = 0
    refusal_total = 0
    vector_hits = 0
    answerable_total = 0
    answered = 0

    for item in samples:
        kept = [evidence_id for evidence_id, score in item.scored if score >= threshold]
        if item.expect_refusal:
            refusal_total += 1
            refusal_correct += 0 if kept else 1
            continue
        answerable_total += 1
        answered += 1 if kept else 0
        expected = set(item.expected_evidence_ids)
        vector_hits += 1 if any(evidence_id in expected for evidence_id in kept) else 0

    margin = min(abs(threshold - score) for score in scores)
    return ThresholdPoint(
        threshold=threshold,
        margin=margin,
        refusal_correct=refusal_correct,
        refusal_total=refusal_total,
        vector_hits=vector_hits,
        answerable_total=answerable_total,
        answered=answered,
    )


def _notes(
    samples: Sequence[VectorSample],
    points: Sequence[ThresholdPoint],
    recommended: ThresholdPoint,
    *,
    feasible: bool,
    stability_margin: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    refusal_tops = [
        item.top_score
        for item in samples
        if item.expect_refusal and item.top_score is not None
    ]
    if refusal_tops:
        highest = max(refusal_tops)
        if feasible:
            notes.append(
                f"最高拒答分 {highest:.3f}：推荐阈值高于它，"
                f"{recommended.refusal_total} 道拒答对照才会全部拒答。"
            )
        else:
            notes.append(
                f"没有任何阈值能让全部拒答对照题都拒答（最高拒答分 {highest:.3f}）；"
                "已按拒答数优先选择最不坏的点。"
            )
    notes.append(
        f"推荐阈值 {recommended.threshold:.2f} 离最近的关键重排分数 "
        f"{recommended.margin:.3f}（稳定余量下限 {stability_margin:.3f}）。"
    )

    if feasible:
        alternatives = [
            point
            for point in points
            if point.refusal_complete
            and point.vector_hits > recommended.vector_hits
            and point.threshold < recommended.threshold
        ]
        if alternatives:
            best = max(alternatives, key=_preference)
            notes.append(
                f"存在更低的可行阈值 {best.threshold:.2f}，可多保留 "
                f"{best.vector_hits - recommended.vector_hits} 道题的期望证据，"
                f"但它离最近的关键分数只有 {best.margin:.3f}；"
                "重排分数在重复调用之间会抖动，贴着边界的点下次可能翻面，故未采用。"
            )

    lost = {
        item.question_id: [
            score
            for evidence_id, score in item.scored
            if evidence_id in set(item.expected_evidence_ids)
            and score < recommended.threshold
        ]
        for item in samples
        if not item.expect_refusal
    }
    lost = {key: value for key, value in lost.items() if value}
    if lost:
        lowest = min(min(value) for value in lost.values())
        notes.append(
            f"有 {len(lost)} 道可答题的期望证据低于推荐阈值（最低 {lowest:.3f}）；"
            "它们与拒答对照的分数区间重叠，单阈值无法同时保留。"
        )
    if recommended.vector_hits == 0:
        notes.append(
            "推荐阈值下没有任何可答题的期望证据被保留："
            "该校准只保证拒答，向量路径在这批题上不提供可核对证据。"
        )
    return tuple(notes)


def calibration_report_to_dict(
    samples: Sequence[VectorSample], result: CalibrationResult
) -> dict[str, object]:
    """校准的 JSON 留档形态；字段名是复现与对比的契约。"""
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "policy": {
            "description": (
                "先保证全部拒答对照题拒答；再要求阈值离关键分数至少 "
                f"{result.stability_margin:.3f}；最后选期望证据保留最多、余量最大的点。"
            ),
            "stability_margin": result.stability_margin,
        },
        "recommendation": {
            **_point_to_dict(result.recommended),
            "all_refusals_correct": result.all_refusals_correct,
        },
        "notes": list(result.notes),
        "table": [_point_to_dict(point) for point in result.table],
        "samples": [
            {
                "question_id": item.question_id,
                "expect_refusal": item.expect_refusal,
                "expected_evidence_ids": list(item.expected_evidence_ids),
                "top_score": item.top_score,
                "scores": [
                    {"evidence_id": evidence_id, "score": score}
                    for evidence_id, score in item.scored
                ],
            }
            for item in samples
        ],
    }


def _point_to_dict(point: ThresholdPoint) -> dict[str, object]:
    return {
        "threshold": point.threshold,
        "margin": point.margin,
        "refusal_correct": point.refusal_correct,
        "refusal_total": point.refusal_total,
        "vector_hits": point.vector_hits,
        "answerable_total": point.answerable_total,
        "answered": point.answered,
    }


def updated_settings_text(
    text: str, threshold: float, *, calibrated: bool = True
) -> str:
    """把阈值与校准标记写回配置文本，保留注释与其他键。

    只改两行：``vector_min_rerank_score`` 的数值与
    ``vector_min_rerank_score_calibrated`` 的布尔值；后者不存在时插在前者之后。
    写回结果必须能被 :class:`RetrievalSettings` 解析，否则不落盘。
    """
    if not 0.0 <= threshold <= 1.0:
        raise CalibrationError(f"阈值必须在 0 到 1 之间：{threshold}")
    if _SCORE_LINE.search(text) is None:
        raise CalibrationError(
            "检索配置缺少 vector_min_rerank_score 行，无法写回校准值"
        )

    updated = _SCORE_LINE.sub(
        lambda match: f"{match.group(1)}{threshold:.2f}{match.group(3)}",
        text,
        count=1,
    )
    flag = "true" if calibrated else "false"
    if _CALIBRATED_LINE.search(updated) is not None:
        updated = _CALIBRATED_LINE.sub(
            lambda match: f"{match.group(1)}{flag}{match.group(3)}",
            updated,
            count=1,
        )
    else:
        match = _SCORE_LINE.search(updated)
        assert match is not None
        insertion = f"\nvector_min_rerank_score_calibrated: {flag}"
        updated = updated[: match.end()] + insertion + updated[match.end() :]

    _validate_settings_text(updated)
    return updated


def write_threshold(
    path: Path, threshold: float, *, calibrated: bool = True
) -> RetrievalSettings:
    """校准命令的落盘动作：改写配置文本、写回、再按正常加载路径复读一遍。"""
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationError(
            f"检索配置读取失败：{path}（{error.strerror or error}）"
        ) from None
    updated = updated_settings_text(original, threshold, calibrated=calibrated)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as error:
        raise CalibrationError(
            f"检索配置写入失败：{path}（{error.strerror or error}）"
        ) from None
    return load_retrieval_settings(path)


def _validate_settings_text(text: str) -> None:
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise CalibrationError(f"写回后的检索配置不是合法 YAML：{error}") from None
    if not isinstance(payload, dict):
        raise CalibrationError("写回后的检索配置顶层必须是键值对")
    try:
        RetrievalSettings.model_validate(payload)
    except ValidationError as error:
        details = "、".join(
            f"{'.'.join(str(part) for part in item['loc']) or '顶层'}（{item['msg']}）"
            for item in error.errors()
        )
        raise CalibrationError(f"写回后的检索配置取值非法：{details}") from None

"""阈值校准：把金标准集的重排分数分布变成一个可解释、可复现的阈值。

校准只做两件事：先采样（每道题强制走向量路径，拿到全部候选的分数），再离线扫
阈值。评分只用金标准集里已经绑定好的证据段标识，不引入裁判模型。

选择规则是拒绝优先：阈值必须高于每一道拒答对照题的最高分，否则「该拒却答」；
在此前提下，在分数抖动之外选期望证据保留最多、离观测分数最远的那个点。若没有
任何阈值能同时满足两者，结果里写明是哪一个目标被牺牲，而不是假装都满足。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from idr_rotating.calibration import (
    CalibrationError,
    VectorSample,
    recommend_threshold,
    updated_settings_text,
    write_threshold,
)
from idr_rotating.retrieval import RetrievalSettings, load_retrieval_settings


def sample(
    question_id: str,
    scored: dict[str, float],
    *,
    refusal: bool = False,
    expected: tuple[str, ...] = (),
) -> VectorSample:
    return VectorSample(
        question_id=question_id,
        expect_refusal=refusal,
        expected_evidence_ids=expected,
        scored=tuple(scored.items()),
    )


def test_the_threshold_sits_above_the_highest_refusal_score() -> None:
    samples = (
        sample("A-1", {"ev:a": 0.9}, expected=("ev:a",)),
        sample("R-1", {"ev:r": 0.7}, refusal=True),
    )

    result = recommend_threshold(samples)

    assert result.recommended.threshold > 0.7
    assert result.recommended.threshold <= 0.9
    assert result.all_refusals_correct
    assert result.recommended.refusal_correct == 1
    assert result.recommended.vector_hits == 1


def test_a_stable_threshold_wins_over_a_razor_thin_one() -> None:
    """0.71 的证据几乎贴着拒答分 0.70；用约 0.80 换掉它，边界才经得起抖动。"""
    samples = (
        sample("A-1", {"ev:a": 0.90}, expected=("ev:a",)),
        sample("A-2", {"ev:b": 0.71}, expected=("ev:b",)),
        sample("A-3", {"ev:c": 0.95}, expected=("ev:c",)),
        sample("R-1", {"ev:r": 0.70}, refusal=True),
    )

    stable = recommend_threshold(samples)
    greedy = recommend_threshold(samples, stability_margin=0.0)

    assert stable.recommended.threshold > 0.71
    assert stable.recommended.vector_hits == 2
    assert stable.recommended.margin >= 0.02
    assert greedy.recommended.vector_hits == 3
    assert greedy.recommended.threshold <= 0.71


def test_cross_language_evidence_cannot_be_kept_without_breaking_refusal() -> None:
    samples = (
        sample("A-1", {"ev:a": 0.90}, expected=("ev:a",)),
        sample("A-2", {"ev:cross": 0.15}, expected=("ev:cross",)),
        sample("R-1", {"ev:r": 0.70}, refusal=True),
    )

    result = recommend_threshold(samples)

    assert result.recommended.threshold > 0.70
    assert result.recommended.vector_hits == 1
    assert any("0.150" in note for note in result.notes)
    assert any("单阈值" in note for note in result.notes)


def test_a_perfect_refusal_score_is_infeasible_and_says_so() -> None:
    samples = (
        sample("A-1", {"ev:a": 0.90}, expected=("ev:a",)),
        sample("R-1", {"ev:r": 1.0}, refusal=True),
    )

    result = recommend_threshold(samples)

    assert not result.all_refusals_correct
    assert result.recommended.refusal_correct == 0
    assert any("没有任何阈值" in note for note in result.notes)


def test_a_refusal_without_any_candidate_does_not_raise_the_threshold() -> None:
    samples = (
        sample("A-1", {"ev:a": 0.90}, expected=("ev:a",)),
        sample("R-1", {}, refusal=True),
    )

    result = recommend_threshold(samples)

    assert result.all_refusals_correct
    assert result.recommended.vector_hits == 1
    assert 0.0 < result.recommended.threshold <= 0.90


def test_irrelevant_candidates_do_not_shrink_the_stable_gap() -> None:
    """只把会改变结论的分数当边界：低分候选不应把稳定区间切碎。"""
    samples = (
        sample(
            "A-1",
            {"ev:a": 0.90, "ev:noise-1": 0.702, "ev:noise-2": 0.704},
            expected=("ev:a",),
        ),
        sample("R-1", {"ev:r": 0.70}, refusal=True),
    )

    result = recommend_threshold(samples)

    assert result.recommended.threshold > 0.70
    assert result.recommended.margin == pytest.approx(0.10)
    assert result.recommended.vector_hits == 1


def test_calibration_needs_both_controls_and_candidates() -> None:
    answerable = sample("A-1", {"ev:a": 0.9}, expected=("ev:a",))
    refusal = sample("R-1", {"ev:r": 0.7}, refusal=True)

    with pytest.raises(CalibrationError, match="样本"):
        recommend_threshold(())
    with pytest.raises(CalibrationError, match="拒答对照"):
        recommend_threshold((answerable,))
    with pytest.raises(CalibrationError, match="可答"):
        recommend_threshold((refusal,))
    with pytest.raises(CalibrationError, match="重排分数"):
        recommend_threshold(
            (
                VectorSample("A-1", False, ("ev:a",), ()),
                VectorSample("R-1", True, (), ()),
            )
        )


CONFIG_TEXT = """\
# 检索参数：注释必须保留
vector_top_k: 20
vector_min_rerank_score: 0.35  # 初值
vector_max_results: 5
"""


def test_writing_a_threshold_preserves_comments_and_marks_it_calibrated() -> None:
    updated = updated_settings_text(CONFIG_TEXT, 0.76)

    assert "# 检索参数：注释必须保留" in updated
    assert "vector_min_rerank_score: 0.76  # 初值" in updated
    assert "vector_min_rerank_score_calibrated: true" in updated
    settings = RetrievalSettings.model_validate(yaml.safe_load(updated))
    assert settings.vector_min_rerank_score == 0.76
    assert settings.vector_min_rerank_score_calibrated is True
    assert settings.vector_top_k == 20


def test_an_existing_calibration_flag_is_flipped_in_place() -> None:
    text = (
        "vector_min_rerank_score: 0.35\n"
        "vector_min_rerank_score_calibrated: false\n"
    )

    updated = updated_settings_text(text, 0.7)

    assert updated.count("vector_min_rerank_score_calibrated") == 1
    assert "vector_min_rerank_score_calibrated: true" in updated
    assert "vector_min_rerank_score: 0.70" in updated


def test_writing_a_threshold_rejects_bad_values_and_missing_lines() -> None:
    with pytest.raises(CalibrationError, match="0 到 1"):
        updated_settings_text(CONFIG_TEXT, 1.5)
    with pytest.raises(CalibrationError, match="vector_min_rerank_score"):
        updated_settings_text("vector_top_k: 20\n", 0.5)


def test_write_threshold_updates_the_file_and_reloads_it(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")

    settings = write_threshold(path, 0.76)

    assert settings == load_retrieval_settings(path)
    assert settings.vector_min_rerank_score == 0.76
    assert settings.vector_min_rerank_score_calibrated is True
    assert "# 检索参数" in path.read_text(encoding="utf-8")

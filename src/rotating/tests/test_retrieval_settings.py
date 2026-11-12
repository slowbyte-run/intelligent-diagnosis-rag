"""检索配置：向量兜底的初值与加载契约。

阈值这类参数已经用金标准集校准（#17）；这里只保证它来自配置文件、校准标记
与数值一起被记录、取值合法，写错了会明确失败，而不是悄悄用默认值继续跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from idr_rotating.retrieval import (
    DEFAULT_RETRIEVAL_PATH,
    RetrievalSettings,
    RetrievalSettingsError,
    load_retrieval_settings,
)


def test_the_packaged_configuration_records_the_calibrated_threshold() -> None:
    settings = load_retrieval_settings()

    assert settings == RetrievalSettings(
        vector_top_k=20,
        vector_min_rerank_score=0.78,
        vector_min_rerank_score_calibrated=True,
        vector_max_results=5,
        query_enrichment=False,
        query_instruction="",
        lexical_recall=False,
    )
    assert settings.vector_max_results <= settings.vector_top_k


def test_settings_without_provenance_are_treated_as_uncalibrated() -> None:
    assert RetrievalSettings().vector_min_rerank_score_calibrated is False


def test_an_unknown_query_instruction_preset_fails(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("query_instruction: translate\n")

    with pytest.raises(RetrievalSettingsError, match="预设"):
        load_retrieval_settings(path)


def test_the_query_instruction_can_be_disabled_with_an_empty_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("query_instruction: ''\n")

    assert load_retrieval_settings(path).query_instruction == ""


def test_query_enrichment_is_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("query_enrichment: true\n")

    assert load_retrieval_settings(path).query_enrichment is True
    assert RetrievalSettings().query_enrichment is False


def test_lexical_recall_is_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("lexical_recall: true\n")

    assert load_retrieval_settings(path).lexical_recall is True
    assert RetrievalSettings().lexical_recall is False


def test_a_configuration_file_can_mark_the_threshold_as_calibrated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        "vector_min_rerank_score: 0.6\nvector_min_rerank_score_calibrated: true\n",
        encoding="utf-8",
    )

    settings = load_retrieval_settings(path)

    assert settings.vector_min_rerank_score == 0.6
    assert settings.vector_min_rerank_score_calibrated is True


def test_a_non_boolean_calibration_flag_fails(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("vector_min_rerank_score_calibrated: maybe\n")

    with pytest.raises(RetrievalSettingsError, match="取值非法"):
        load_retrieval_settings(path)


def test_settings_can_be_overridden_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        "vector_top_k: 8\nvector_min_rerank_score: 0.6\nvector_max_results: 3\n",
        encoding="utf-8",
    )

    assert load_retrieval_settings(path) == RetrievalSettings(
        vector_top_k=8, vector_min_rerank_score=0.6, vector_max_results=3
    )


def test_missing_configuration_file_fails_explicitly(tmp_path: Path) -> None:
    with pytest.raises(RetrievalSettingsError, match="读取"):
        load_retrieval_settings(tmp_path / "missing.yaml")


def test_unknown_keys_fail_instead_of_being_ignored(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("vector_top_k: 8\nverctor_min_rerank_score: 0.6\n")

    with pytest.raises(RetrievalSettingsError, match="取值非法"):
        load_retrieval_settings(path)


def test_a_score_outside_the_rerank_range_fails(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("vector_min_rerank_score: 1.5\n")

    with pytest.raises(RetrievalSettingsError):
        load_retrieval_settings(path)


def test_more_results_than_recalled_candidates_fails(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("vector_top_k: 2\nvector_max_results: 5\n")

    with pytest.raises(RetrievalSettingsError, match="top_k"):
        load_retrieval_settings(path)


def test_the_default_file_is_the_packaged_one() -> None:
    assert DEFAULT_RETRIEVAL_PATH.name == "retrieval.yaml"
    assert DEFAULT_RETRIEVAL_PATH.parent.name == "config"
    assert DEFAULT_RETRIEVAL_PATH.is_file()

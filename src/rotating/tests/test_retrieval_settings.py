"""检索配置：向量兜底的初值与加载契约。

阈值这类参数先给初值，等金标准集跑出基线再校准（#17）；这里只保证它来自
配置文件、取值合法、写错了会明确失败，而不是悄悄用默认值继续跑。
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


def test_default_configuration_holds_the_uncalibrated_initial_values() -> None:
    settings = load_retrieval_settings()

    assert settings == RetrievalSettings(
        vector_top_k=20, vector_min_rerank_score=0.35, vector_max_results=5
    )
    assert settings.vector_max_results <= settings.vector_top_k


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

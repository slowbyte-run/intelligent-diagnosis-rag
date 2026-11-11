"""配置加载的公开契约：缺项一次列全，取值非法在启动时明确失败。"""

from __future__ import annotations

import pytest

from idr_core.config import ConfigError, load_config
from idr_core.embedding import EMBEDDING_DIM


def valid_env(**overrides: str) -> dict[str, str]:
    """一套完整的环境变量；只覆盖测试关心的项。"""
    env = {
        "DEEPSEEK_BASE_URL": "https://example.invalid",
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_MODEL": "test-model",
        "SILICONFLOW_BASE_URL": "https://example.invalid",
        "SILICONFLOW_API_KEY": "sk-test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIM": str(EMBEDDING_DIM),
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "DATABASE_URL": "postgresql://idr:secret@127.0.0.1:1/idr",
    }
    env.update(overrides)
    return env


def test_load_config_accepts_the_fixed_embedding_dimension() -> None:
    config = load_config(valid_env())

    assert config.embedding_dim == EMBEDDING_DIM


def test_load_config_rejects_an_embedding_dimension_other_than_the_fixed_one() -> None:
    with pytest.raises(ConfigError, match=f"{EMBEDDING_DIM}"):
        load_config(valid_env(EMBEDDING_DIM=str(EMBEDDING_DIM // 2)))


def test_load_config_names_every_missing_variable_at_once() -> None:
    env = valid_env()
    del env["DEEPSEEK_API_KEY"]
    del env["DATABASE_URL"]

    with pytest.raises(ConfigError) as error:
        load_config(env)

    assert "DEEPSEEK_API_KEY" in str(error.value)
    assert "DATABASE_URL" in str(error.value)


def test_load_config_parses_llm_extra_headers() -> None:
    config = load_config(
        valid_env(LLM_EXTRA_HEADERS='{"x-opencode-session": "idr"}')
    )

    assert config.llm_extra_headers == {"x-opencode-session": "idr"}


def test_load_config_rejects_malformed_llm_extra_headers() -> None:
    with pytest.raises(ConfigError):
        load_config(valid_env(LLM_EXTRA_HEADERS="not-json"))

"""配置：从环境变量加载，缺项时一次性列出全部缺失项。

环境变量名只在 Config 的字段别名里出现一次；缺项检查与取值都从模型派生，
新增一个变量只需在这里加一行和一个别名。
"""

import json
import os
from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from idr_core.embedding import EMBEDDING_DIM


class ConfigError(RuntimeError):
    """配置缺失或非法。消息面向人阅读，调用方不应打印堆栈。"""


DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    deepseek_base_url: str = Field(validation_alias="DEEPSEEK_BASE_URL")
    deepseek_api_key: str = Field(validation_alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(validation_alias="DEEPSEEK_MODEL")
    siliconflow_base_url: str = Field(validation_alias="SILICONFLOW_BASE_URL")
    siliconflow_api_key: str = Field(validation_alias="SILICONFLOW_API_KEY")
    embedding_model: str = Field(validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(validation_alias="EMBEDDING_DIM")
    rerank_model: str = Field(
        default=DEFAULT_RERANK_MODEL, validation_alias="RERANK_MODEL"
    )
    neo4j_uri: str = Field(validation_alias="NEO4J_URI")
    neo4j_user: str = Field(validation_alias="NEO4J_USER")
    neo4j_password: str = Field(validation_alias="NEO4J_PASSWORD")
    database_url: str = Field(validation_alias="DATABASE_URL")
    llm_extra_headers: dict[str, str] = Field(
        default_factory=dict, validation_alias="LLM_EXTRA_HEADERS"
    )

    @field_validator("llm_extra_headers", mode="before")
    @classmethod
    def _parse_extra_headers(cls, value: Any) -> dict[str, str]:
        """环境变量里以 JSON 对象提供额外的请求头。"""
        if value is None or value == "":
            return {}
        if isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items()}
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            raise ValueError("必须是 JSON 对象，例如 {\"x-header\": \"value\"}") from None
        if not isinstance(parsed, dict):
            raise ValueError("必须是 JSON 对象")
        return {str(k): str(v) for k, v in parsed.items()}

    @field_validator("embedding_dim")
    @classmethod
    def _fixed_embedding_dimension(cls, value: int) -> int:
        """ADR-0002 把向量维度钉死在 EMBEDDING_DIM；配置漂移在这里就失败。"""
        if value != EMBEDDING_DIM:
            raise ValueError(
                f"EMBEDDING_DIM 必须等于 {EMBEDDING_DIM}（ADR-0002 固定向量维度），"
                f"收到 {value}；换维度等于重建全部向量索引，不属于配置可调项"
            )
        return value

    @classmethod
    def required_env_names(cls) -> tuple[str, ...]:
        """必须由环境提供的变量名，按模型定义的顺序。"""
        names: list[str] = []
        for field in cls.model_fields.values():
            if not field.is_required():
                continue
            alias = field.validation_alias
            names.append(alias if isinstance(alias, str) else str(alias))
        return tuple(names)


def load_config(env: Mapping[str, str] | None = None) -> Config:
    source = os.environ if env is None else env

    missing = [
        name for name in Config.required_env_names() if not source.get(name, "").strip()
    ]
    if missing:
        raise ConfigError("缺少必需的环境变量：" + "、".join(missing))

    try:
        return Config.model_validate(dict(source))
    except ValidationError as error:
        details = "、".join(
            f"{'.'.join(str(part) for part in item['loc'])}（{item['msg']}）"
            for item in error.errors()
        )
        raise ConfigError(f"环境变量取值非法：{details}") from None

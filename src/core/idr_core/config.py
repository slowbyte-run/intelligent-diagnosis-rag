"""配置：从环境变量加载，缺项时一次性列出全部缺失项。"""

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigError(RuntimeError):
    """配置缺失或非法。消息面向人阅读，调用方不应打印堆栈。"""


REQUIRED_VARIABLES: tuple[str, ...] = (
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "SILICONFLOW_BASE_URL",
    "SILICONFLOW_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "DATABASE_URL",
)

DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    deepseek_base_url: str
    deepseek_api_key: str
    deepseek_model: str
    siliconflow_base_url: str
    siliconflow_api_key: str
    embedding_model: str
    embedding_dim: int
    rerank_model: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    database_url: str


def load_config(env: Mapping[str, str] | None = None) -> Config:
    source = os.environ if env is None else env

    missing = [name for name in REQUIRED_VARIABLES if not source.get(name, "").strip()]
    if missing:
        raise ConfigError("缺少必需的环境变量：" + "、".join(missing))

    try:
        return Config(
            deepseek_base_url=source["DEEPSEEK_BASE_URL"],
            deepseek_api_key=source["DEEPSEEK_API_KEY"],
            deepseek_model=source["DEEPSEEK_MODEL"],
            siliconflow_base_url=source["SILICONFLOW_BASE_URL"],
            siliconflow_api_key=source["SILICONFLOW_API_KEY"],
            embedding_model=source["EMBEDDING_MODEL"],
            embedding_dim=source["EMBEDDING_DIM"],
            rerank_model=source.get("RERANK_MODEL") or DEFAULT_RERANK_MODEL,
            neo4j_uri=source["NEO4J_URI"],
            neo4j_user=source["NEO4J_USER"],
            neo4j_password=source["NEO4J_PASSWORD"],
            database_url=source["DATABASE_URL"],
        )
    except ValidationError as exc:
        details = "、".join(
            f"{'.'.join(str(part) for part in error['loc'])}（{error['msg']}）"
            for error in exc.errors()
        )
        raise ConfigError(f"环境变量取值非法：{details}") from None

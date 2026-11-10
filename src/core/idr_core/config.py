"""配置：从环境变量加载，缺项时一次性列出全部缺失项。

环境变量名只在 Config 的字段别名里出现一次；缺项检查与取值都从模型派生，
新增一个变量只需在这里加一行和一个别名。
"""

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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

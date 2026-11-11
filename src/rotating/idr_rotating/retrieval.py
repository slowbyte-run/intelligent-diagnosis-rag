"""检索配置：阈值与召回规模的唯一事实源。

阈值这类参数是「猜出来的初值」，必须由金标准集校准后再覆写（#17）；因此它们
放在配置文件里而不是散在代码常量里——校准结果直接落文件，评估与线上读同一份。
配置写错（拼错键、越界、结果数大于候选数）在加载阶段就失败，不悄悄用默认值。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# 与本体文件同级的配置目录：源码运行时直接可读，和 ontology 一样不依赖安装位置。
DEFAULT_RETRIEVAL_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "retrieval.yaml"
)


class RetrievalSettingsError(RuntimeError):
    """检索配置缺失或非法；消息面向人阅读，调用方不应打印堆栈。"""


class RetrievalSettings(BaseModel):
    """一次检索用到的参数；默认值未经校准，见 ``config/retrieval.yaml``。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 向量召回候选数：重排只在前 k 段里做，k 太小会漏，太大会多花重排的 token。
    vector_top_k: int = Field(default=20, ge=1, le=200)
    # 重排分数的下限：低于它的候选没有依据价值，也是「文献里没有」的判据。
    vector_min_rerank_score: float = Field(default=0.35, ge=0.0, le=1.0)
    # 一次回答最多列出的证据段数：候选可以多，答案不能长到没人看。
    vector_max_results: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _results_fit_within_recall(self) -> "RetrievalSettings":
        if self.vector_max_results > self.vector_top_k:
            raise ValueError(
                f"vector_max_results（{self.vector_max_results}）不能大于 "
                f"vector_top_k（{self.vector_top_k}）：答案只能从召回的候选里挑"
            )
        return self


def load_retrieval_settings(path: Path | None = None) -> RetrievalSettings:
    """读取检索配置；不给路径时读包内默认文件。"""
    source = DEFAULT_RETRIEVAL_PATH if path is None else path
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise RetrievalSettingsError(
            f"检索配置读取失败：{source}（{error.strerror or error}）"
        ) from None

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise RetrievalSettingsError(f"检索配置不是合法 YAML：{source}（{error}）") from None
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise RetrievalSettingsError(f"检索配置的顶层必须是键值对：{source}")

    try:
        return RetrievalSettings.model_validate(payload)
    except ValidationError as error:
        details = "、".join(
            f"{'.'.join(str(part) for part in item['loc']) or '顶层'}（{item['msg']}）"
            for item in error.errors()
        )
        raise RetrievalSettingsError(f"检索配置取值非法：{details}") from None

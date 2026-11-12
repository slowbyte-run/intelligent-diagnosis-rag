"""检索配置：阈值、召回规模与查询改写的唯一事实源。

阈值这类参数最初是「猜出来的初值」，已经由金标准集校准并写回（#17）；它们放在
配置文件里而不是散在代码常量里——校准结果直接落文件，评估与线上读同一份。
查询改写开关（#20）同样如此：开哪个变体由金标准集的召回诊断决定，配置里写
开关名，代码里只保留「怎么改」。
配置写错（拼错键、越界、结果数大于候选数、未知指令预设）在加载阶段就失败，
不悄悄用默认值。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from idr_rotating.query_rewrite import INSTRUCTION_PRESETS

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
    # 0.35 是未经校准的初值；线上的值以 config/retrieval.yaml 为准（#17）。
    vector_min_rerank_score: float = Field(default=0.35, ge=0.0, le=1.0)
    # 该阈值是否已经用金标准集校准。与数值分开记录，拒答说明才能如实告诉
    # 用户「这个阈值可不可信」，校准命令也会同时写这两行。
    vector_min_rerank_score_calibrated: bool = False
    # 一次回答最多列出的证据段数：候选可以多，答案不能长到没人看。
    vector_max_results: int = Field(default=5, ge=1)
    # 查询侧的概念补全：把问句命中的本体概念名、别名与英文 ID 词追加到查询。
    # 实测只在候选集更大（k≥50）时才涨召回，k=20 无增益，故默认关闭（#20）。
    query_enrichment: bool = False
    # 查询侧的检索指令预设（Qwen3-Embedding 的 Instruct 格式）；空字符串表示
    # 不加指令。预设文本在 query_rewrite 里，配置只选名不复制正文。
    # 实测 crosslingual 在 k=20 最优，故打包配置启用它（#20）。
    query_instruction: str = ""
    # 词面召回（概念名/别名的子串命中）：与向量名次做 RRF 融合，补语义近邻
    # 召回不到的精确术语证据。实测 recall@20 从 27/48 提到 33/48（#20）。
    lexical_recall: bool = False

    @field_validator("query_instruction")
    @classmethod
    def _known_instruction_preset(cls, value: str) -> str:
        if value and value not in INSTRUCTION_PRESETS:
            choices = "、".join(sorted(INSTRUCTION_PRESETS))
            raise ValueError(
                f"未知的检索指令预设 {value!r}（可选：{choices} 或空字符串）"
            )
        return value

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

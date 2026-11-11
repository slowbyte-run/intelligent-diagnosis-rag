"""模型调用端口。

业务代码只依赖这个协议；具体供应商适配器放在 ``idr_core.llm`` 下（AGENTS.md 第 4 条）。

必需的成员只有 ``complete(prompt) -> str``。客户端还可以额外实现
``InstrumentedLLM``：把每次调用的用量与调用参数报告出来，批次报告与成本
核算依赖这些字段；没有实现的假实现仍能走完整条链路，只是用量记为未知。

嵌入同样是模型调用，形状与文本生成不同：``Embedder`` 是业务侧的最小端口，
``InstrumentedEmbedder`` 报告模型名与输入 token；``embed_texts`` 与
``call_model`` 一样负责计时并产出统一的成本记录。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from idr_core.embedding import Embedder

UNKNOWN_MODEL = "unknown"


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """一次模型调用的结果：文本加上供应商报告的用量。"""

    text: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """模型标识与影响输出的调用参数，用于复现一次运行。"""

    model: str
    parameters: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """一次嵌入调用的结果：向量加上供应商报告的模型与用量。

    嵌入没有输出 token，``embed_texts`` 产出的成本记录里 output_tokens 记为未知。
    """

    vectors: list[list[float]]
    model: str | None = None
    input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCall:
    """一次模型调用的事实记录：模型名、输入输出 token 数与耗时。"""

    model: str
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class InstrumentedLLM(Protocol):
    """能报告用量与调用参数的客户端；真实适配器实现它。"""

    def complete_with_usage(self, prompt: str) -> LLMResponse: ...

    def describe(self) -> ModelDescriptor: ...


@runtime_checkable
class InstrumentedEmbedder(Protocol):
    """能报告模型名与输入 token 的嵌入客户端；真实适配器实现它。"""

    def embed_with_usage(self, texts: Sequence[str]) -> EmbeddingResponse: ...


def call_model(llm: LLMClient, prompt: str) -> tuple[str, ModelCall]:
    """调用模型，并记录本次调用的模型名、token 用量与耗时。"""
    started = perf_counter()
    if isinstance(llm, InstrumentedLLM):
        response = llm.complete_with_usage(prompt)
        text = response.text
        model = response.model
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
    else:
        text = llm.complete(prompt)
        model = None
        input_tokens = None
        output_tokens = None
    return text, ModelCall(
        model=model or UNKNOWN_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=perf_counter() - started,
    )


def embed_texts(
    embedder: Embedder, texts: Sequence[str]
) -> tuple[list[list[float]], ModelCall]:
    """嵌入一批文本，并记录本次调用的模型名、输入 token 与耗时。"""
    started = perf_counter()
    if isinstance(embedder, InstrumentedEmbedder):
        response = embedder.embed_with_usage(texts)
        vectors = response.vectors
        model = response.model
        input_tokens = response.input_tokens
    else:
        vectors = embedder.embed(texts)
        model = None
        input_tokens = None
    return vectors, ModelCall(
        model=model or UNKNOWN_MODEL,
        input_tokens=input_tokens,
        output_tokens=None,
        elapsed_seconds=perf_counter() - started,
    )


def describe_model(llm: LLMClient) -> ModelDescriptor | None:
    """客户端的模型与调用参数；未实现该能力的客户端返回 None。"""
    if isinstance(llm, InstrumentedLLM):
        return llm.describe()
    return None

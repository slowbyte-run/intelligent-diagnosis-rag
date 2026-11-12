"""重排端口与 OpenAI 兼容适配器。

重排是模型调用，收敛在 ``idr_core.llm`` 层（AGENTS.md 第 4 条）。供应商的
``/rerank`` 不是 chat completions，但请求仍然走同一个 OpenAI 兼容客户端：适配
器用客户端的通用 ``post`` 发原始 JSON，不额外引入 HTTP 依赖。

端口的返回值是「输入文档的下标 + 分数」而不是 ``(文档, 分数)``：调用方持有
原文，适配器不复制它，错位也不可能发生。供应商省略用量时 token 记为未知，
不阻断链路；模型名同样在缺省时记为未知（见 ``UNKNOWN_MODEL``）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from idr_core.llm import UNKNOWN_MODEL, ModelCall


class RerankError(RuntimeError):
    """重排服务的响应与请求对不上：不是合法的 JSON 对象，或分数与文档错位。"""


@dataclass(frozen=True, slots=True)
class RerankScore:
    """一条候选的重排结果；``index`` 对应输入 ``documents`` 的下标。"""

    index: int
    score: float


@dataclass(frozen=True, slots=True)
class RerankResponse:
    """一次重排的结果：每条候选的分数，加上供应商报告的模型与用量。"""

    scores: tuple[RerankScore, ...]
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Reranker(Protocol):
    """重排端口：给查询与候选文档打分，不改变文档本身。"""

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse: ...


class OpenAICompatibleReranker:
    """SiliconFlow 一类的 OpenAI 兼容重排服务。

    请求体按供应商的 ``/rerank`` 契约组装：``model``、``query``、``documents``，
    并声明不需要把文档回传（原文在调用方手里）。响应里的 ``meta.tokens`` 是
    SiliconFlow 的用量位置；其他网关把用量放在顶层的 ``usage``，两种都读，
    都没有就记未知。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        client: Any = None,
    ) -> None:
        if client is not None:
            self._client: Any = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=base_url, api_key=api_key, timeout=timeout_seconds
            )
        self._model = model

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        if not documents:
            return RerankResponse(scores=(), model=self._model)

        payload = self._client.post(
            "/rerank",
            body={
                "model": self._model,
                "query": query,
                "documents": list(documents),
                "return_documents": False,
            },
            # 供应商的响应不在这版 SDK 的类型里；拿原始 JSON 自己校验。
            cast_to=object,
        )
        return _parse_rerank_response(payload, model=self._model, count=len(documents))


def rerank_documents(
    reranker: Reranker, query: str, documents: Sequence[str]
) -> tuple[RerankResponse, ModelCall]:
    """调用重排服务，并记录本次调用的模型名、token 用量与耗时。"""
    started = perf_counter()
    response = reranker.rerank(query, documents)
    return response, ModelCall(
        model=response.model or UNKNOWN_MODEL,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        elapsed_seconds=perf_counter() - started,
    )


def _parse_rerank_response(
    payload: object, *, model: str, count: int
) -> RerankResponse:
    """校验响应形状与下标：分数与文档必须一一对应，错位会静默把证据配错。"""
    if not isinstance(payload, Mapping):
        raise RerankError(f"重排服务返回的不是 JSON 对象：{type(payload).__name__}")

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RerankError("重排服务的响应缺少 results 数组")

    scores: list[RerankScore] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            raise RerankError("重排结果里的元素不是 JSON 对象")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RerankError(f"重排结果的下标不是整数：{index!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RerankError(f"重排结果的 relevance_score 不是数值：{score!r}")
        scores.append(RerankScore(index=index, score=float(score)))

    if {item.index for item in scores} != set(range(count)):
        raise RerankError(
            f"重排结果的下标与输入文档对不上：期望 0..{count - 1}，"
            f"收到 {sorted(item.index for item in scores)}"
        )

    input_tokens, output_tokens = _token_usage(payload)
    return RerankResponse(
        scores=tuple(scores),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _token_usage(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """读用量：优先 SiliconFlow 的 ``meta.tokens``，其次通用的 ``usage``。"""
    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        tokens = meta.get("tokens")
        if isinstance(tokens, Mapping):
            input_tokens = _as_int(tokens.get("input_tokens"))
            output_tokens = _as_int(tokens.get("output_tokens"))
            if input_tokens is not None or output_tokens is not None:
                return input_tokens, output_tokens

    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        input_tokens = _as_int(usage.get("prompt_tokens"))
        output_tokens = _as_int(usage.get("completion_tokens"))
        if input_tokens is not None or output_tokens is not None:
            return input_tokens, output_tokens

    return None, None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None

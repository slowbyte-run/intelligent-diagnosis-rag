"""重排适配器：把供应商返回的分数与用量映射成端口的结果对象。

测试用假的 SDK 客户端顶上真实客户端，不联网、不真调模型；验证的是适配器
对外的行为——请求形状、分数下标、token 用量与成本记录。
"""

from __future__ import annotations

from typing import Any

import pytest

from idr_core.llm.rerank import (
    OpenAICompatibleReranker,
    RerankError,
    RerankScore,
    rerank_documents,
)


class FakeRerankClient:
    """顶替 openai 客户端：记录通用 POST 的路径与请求体，返回预置响应。"""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.requests: list[tuple[str, dict[str, Any], type[Any]]] = []

    def post(
        self, path: str, *, body: dict[str, Any], cast_to: type[Any]
    ) -> Any:
        self.requests.append((path, body, cast_to))
        return self._payload


def reranker_with(
    payload: Any, *, model: str = "Qwen/Qwen3-Reranker-8B"
) -> tuple[OpenAICompatibleReranker, FakeRerankClient]:
    client = FakeRerankClient(payload)
    reranker = OpenAICompatibleReranker(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model=model,
        client=client,
    )
    return reranker, client


def test_rerank_maps_scores_and_token_usage() -> None:
    payload = {
        "id": "01a1cc9a2b6773ef9608dd0c1d1393bb",
        "results": [
            {"index": 1, "relevance_score": 3.6e-06},
            {"index": 0, "relevance_score": 0.9995},
        ],
        "meta": {"tokens": {"input_tokens": 193, "output_tokens": 0}},
    }
    reranker, client = reranker_with(payload)

    response = reranker.rerank(
        "查询文本",
        ["与查询最相关的候选。", "与查询无关的候选。"],
    )

    assert response.scores == (
        RerankScore(index=1, score=3.6e-06),
        RerankScore(index=0, score=0.9995),
    )
    assert response.model == "Qwen/Qwen3-Reranker-8B"
    assert (response.input_tokens, response.output_tokens) == (193, 0)
    (path, body, cast_to) = client.requests[0]
    assert path == "/rerank"
    assert body["model"] == "Qwen/Qwen3-Reranker-8B"
    assert body["query"] == "查询文本"
    assert body["documents"] == [
        "与查询最相关的候选。",
        "与查询无关的候选。",
    ]
    assert body["return_documents"] is False
    assert cast_to is object


def test_rerank_reports_unknown_tokens_when_the_service_omits_them() -> None:
    payload = {"results": [{"index": 0, "relevance_score": 0.5}]}
    reranker, _ = reranker_with(payload, model="local-reranker")

    response = reranker.rerank("问句", ["一段"])

    assert response.model == "local-reranker"
    assert (response.input_tokens, response.output_tokens) == (None, None)


def test_rerank_rejects_a_response_whose_indexes_do_not_match_the_documents() -> None:
    payload = {"results": [{"index": 1, "relevance_score": 0.5}]}
    reranker, _ = reranker_with(payload)

    with pytest.raises(RerankError, match="下标"):
        reranker.rerank("问句", ["第一段", "第二段"])


def test_rerank_rejects_a_score_that_is_not_a_number() -> None:
    payload = {"results": [{"index": 0, "relevance_score": "高"}]}
    reranker, _ = reranker_with(payload)

    with pytest.raises(RerankError, match="relevance_score"):
        reranker.rerank("问句", ["一段"])


def test_rerank_without_documents_does_not_call_the_service() -> None:
    reranker, client = reranker_with({"results": []})

    response = reranker.rerank("问句", [])

    assert response.scores == ()
    assert client.requests == []


def test_rerank_documents_records_model_tokens_and_elapsed_time() -> None:
    payload = {
        "results": [{"index": 0, "relevance_score": 0.75}],
        "meta": {"tokens": {"input_tokens": 42, "output_tokens": 0}},
    }
    reranker, _ = reranker_with(payload)

    response, call = rerank_documents(reranker, "问句", ["一段"])

    assert response.scores == (RerankScore(index=0, score=0.75),)
    assert call.model == "Qwen/Qwen3-Reranker-8B"
    assert call.input_tokens == 42
    assert call.output_tokens == 0
    assert call.elapsed_seconds >= 0.0

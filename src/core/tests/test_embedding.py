"""嵌入适配器：用假客户端验证请求与响应契约，不访问网络。"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from idr_core.embedding import EmbeddingError
from idr_core.llm import UNKNOWN_MODEL, EmbeddingResponse, embed_texts
from idr_core.llm.embedding import OpenAICompatibleEmbedder


class FakeEmbeddings:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.embeddings = FakeEmbeddings(response)


def embedder_with(
    response: Any, *, dimensions: int = 2
) -> tuple[OpenAICompatibleEmbedder, FakeClient]:
    client = FakeClient(response)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://example.invalid",
        api_key="sk-test",
        model="test-embedding",
        dimensions=dimensions,
        client=client,
    )
    return embedder, client


def test_embed_returns_one_vector_per_text_in_input_order() -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[1.0, 2.0]),
            SimpleNamespace(index=0, embedding=[0.25, 0.5]),
        ]
    )
    embedder, client = embedder_with(response)

    vectors = embedder.embed(["第二段", "第一段"])

    assert vectors == [[0.25, 0.5], [1.0, 2.0]]
    (request,) = client.embeddings.requests
    assert request["input"] == ["第二段", "第一段"]
    assert request["model"] == "test-embedding"
    assert request["dimensions"] == 2


def test_embed_rejects_a_vector_with_unexpected_dimension() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0])]
    )
    embedder, _ = embedder_with(response, dimensions=2)

    with pytest.raises(EmbeddingError, match="3"):
        embedder.embed(["一段"])


def test_embed_rejects_a_response_with_missing_vectors() -> None:
    response = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])])
    embedder, _ = embedder_with(response)

    with pytest.raises(EmbeddingError):
        embedder.embed(["一段", "两段"])


def test_embed_without_texts_does_not_call_the_service() -> None:
    embedder, client = embedder_with(SimpleNamespace(data=[]))

    assert embedder.embed([]) == []
    assert client.embeddings.requests == []


def test_embed_with_usage_reports_the_model_and_token_count() -> None:
    response = SimpleNamespace(
        model="test-embedding",
        data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])],
        usage=SimpleNamespace(prompt_tokens=17, total_tokens=17),
    )
    embedder, _ = embedder_with(response)

    result = embedder.embed_with_usage(["一段"])

    assert result == EmbeddingResponse(
        vectors=[[1.0, 2.0]], model="test-embedding", input_tokens=17
    )
    assert embedder.embed(["一段"]) == [[1.0, 2.0]]


def test_embed_with_usage_reports_unknown_usage_when_the_service_omits_it() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])]
    )
    embedder, _ = embedder_with(response)

    result = embedder.embed_with_usage(["一段"])

    assert (result.model, result.input_tokens) == (None, None)


def test_embed_texts_records_a_cost_record_for_a_plain_embedder() -> None:
    class PlainEmbedder:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts]

    vectors, call = embed_texts(PlainEmbedder(), ["一段"])

    assert vectors == [[0.0, 0.0]]
    assert call.model == UNKNOWN_MODEL
    assert (call.input_tokens, call.output_tokens) == (None, None)
    assert call.elapsed_seconds >= 0.0


def test_embed_texts_prefers_the_instrumented_client() -> None:
    response = SimpleNamespace(
        model="test-embedding",
        data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])],
        usage=SimpleNamespace(prompt_tokens=17, total_tokens=17),
    )
    embedder, _ = embedder_with(response)

    _, call = embed_texts(embedder, ["一段"])

    assert call.model == "test-embedding"
    assert call.input_tokens == 17
    assert call.output_tokens is None
    assert call.elapsed_seconds >= 0.0

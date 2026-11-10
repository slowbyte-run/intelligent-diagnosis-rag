"""嵌入适配器：用假客户端验证请求与响应契约，不访问网络。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from idr_core.embedding import EmbeddingError
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

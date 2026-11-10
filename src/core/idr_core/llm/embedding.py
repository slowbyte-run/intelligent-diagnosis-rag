"""OpenAI 兼容嵌入适配器。

模型调用收敛在 ``idr_core.llm`` 层（AGENTS.md 第 4 条）。ADR-0002 要求所有
嵌入模型产出 1024 维向量，因此向服务端传 MRL 截断参数；服务端若不遵守，
在适配器处明确失败，不静默裁切。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from idr_core.embedding import EMBEDDING_DIM, EmbeddingError


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = EMBEDDING_DIM,
        timeout_seconds: float = 60.0,
        client: Any = None,
    ) -> None:
        self._client: Any = (
            client
            if client is not None
            else OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        )
        self._model = model
        self._dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimensions,
        )
        vectors = [
            list(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"嵌入服务返回 {len(vectors)} 个向量，请求了 {len(texts)} 个文本"
            )
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"嵌入服务返回的向量维度为 {len(vector)}，"
                    f"请求的维度为 {self._dimensions}"
                )
        return vectors

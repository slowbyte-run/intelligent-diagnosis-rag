"""内存假实现：单元测试不碰 Neo4j、Postgres 与网络。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from idr_core.embedding import EMBEDDING_DIM
from idr_core.graph import GraphBatch
from idr_core.parsing import ParsedDocument, ParsedPage


class FakeParser:
    """按页码顺序返回固定页面，不读取真实 PDF。"""

    def __init__(self, pages: Sequence[ParsedPage]) -> None:
        self._pages = tuple(pages)

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(pages=self._pages)


class FakeLLM:
    """返回固定响应，并记录收到的提示词。"""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


class FakeEmbedder:
    """按文本长度生成确定性的定长向量，并记录被嵌入的文本。"""

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        self.dimensions = dimensions
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(text))] * self.dimensions for text in texts]


class FakeVectorStore:
    """按证据段标识覆盖写入，模拟真实适配器的 upsert 语义。"""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        for evidence_id, vector in zip(evidence_ids, vectors, strict=True):
            self.vectors[evidence_id] = list(vector)

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None:
        return self.vectors.get(evidence_id)


class FailingVectorStore:
    """写入必失败的向量存储，用于验证图谱不会被半途写入。"""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        raise self.error

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None:
        return None


class FakeGraphStore:
    """把写入批次留在内存里，供测试断言图谱内容。"""

    def __init__(self) -> None:
        self.batches: list[GraphBatch] = []

    def write_document(self, batch: GraphBatch) -> None:
        self.batches.append(batch)

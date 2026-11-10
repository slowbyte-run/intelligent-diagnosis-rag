"""内存假实现：单元测试不碰 Neo4j、Postgres 与网络。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
    """占位实现：本票不写向量。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class FakeVectorStore:
    """占位实现：本票不写向量。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, list[float]]] = []

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        self.rows.extend(
            (evidence_id, list(vector))
            for evidence_id, vector in zip(evidence_ids, vectors, strict=True)
        )


class FakeGraphStore:
    """把写入批次留在内存里，供测试断言图谱内容。"""

    def __init__(self) -> None:
        self.batches: list[GraphBatch] = []

    def write_document(self, batch: GraphBatch) -> None:
        self.batches.append(batch)

"""向量存储端口。

证据段向量以证据段标识关联，维度由 ADR-0002 固定为 1024。读写分成两个端口：
入库只需要写，检索只需要按向量取近邻；``PgVectorStore`` 同时实现两者。
"""

from collections.abc import Sequence
from typing import Protocol

from idr_core.vector_store.read import VectorMatch, VectorReader


class VectorDimensionError(ValueError):
    """向量维度与服务端约定不符；调用方应明确失败而不是截断或补零。"""


class VectorStore(Protocol):
    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None: ...


__all__ = [
    "VectorDimensionError",
    "VectorMatch",
    "VectorReader",
    "VectorStore",
]

"""向量存储端口。

证据段向量以证据段标识关联，维度由 ADR-0002 固定为 1024。
"""

from collections.abc import Sequence
from typing import Protocol


class VectorDimensionError(ValueError):
    """向量维度与服务端约定不符；调用方应明确失败而不是截断或补零。"""


class VectorStore(Protocol):
    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None: ...

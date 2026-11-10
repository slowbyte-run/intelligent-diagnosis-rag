"""向量存储端口。证据段向量写入在后续票接入，本票只保留注入点。"""

from collections.abc import Sequence
from typing import Protocol


class VectorStore(Protocol):
    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None: ...

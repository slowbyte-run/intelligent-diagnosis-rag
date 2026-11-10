"""图存储端口。"""

from typing import Protocol

from idr_core.graph.models import GraphBatch


class GraphStore(Protocol):
    def write_document(self, batch: GraphBatch) -> None: ...

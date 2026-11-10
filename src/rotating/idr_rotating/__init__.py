"""设备领域包：本体、抽取与入库门面。"""

from idr_rotating.ingest import (
    DocumentReport,
    IngestDeps,
    IngestError,
    IngestReport,
    ingest,
)

__all__ = [
    "DocumentReport",
    "IngestDeps",
    "IngestError",
    "IngestReport",
    "ingest",
]

"""设备领域包：本体、抽取、切分与入库门面。"""

from idr_rotating.document import DocumentType
from idr_rotating.ingest import (
    DocumentReport,
    IngestDeps,
    IngestError,
    IngestReport,
    ingest,
)

__all__ = [
    "DocumentReport",
    "DocumentType",
    "IngestDeps",
    "IngestError",
    "IngestReport",
    "ingest",
]

"""文档处理状态端口：让入库可重复执行。

同一个文档由内容哈希得到稳定的 ``doc_id``；处理状态按 ``doc_id`` 记录在
Postgres 里（不落数据目录），下一次运行据此跳过已完成的文献。中途中断的
文献只有进行中的记录，没有完成记录，因此重跑时只补做未完成的部分。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DocumentStatus(StrEnum):
    """一篇文献在入库流水线中的处理状态。"""

    PROCESSING = "processing"
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocumentProcessingState:
    """一篇文献最近一次处理的结果；``doc_id`` 由内容哈希给出。"""

    doc_id: str
    status: DocumentStatus
    evidence_segments: int = 0
    entities: int = 0
    relations: int = 0
    reason: str | None = None

    @property
    def settled(self) -> bool:
        """是否已处理到不需要再跑：已入库，或已确定要隔离。"""
        return self.status in (DocumentStatus.COMPLETED, DocumentStatus.QUARANTINED)


class IngestStateStore(Protocol):
    """按 ``doc_id`` 读写处理状态，让入库能跳过与续跑。"""

    def get(self, doc_id: str) -> DocumentProcessingState | None:
        """返回该文献最近一次处理状态；从未处理过返回 ``None``。"""
        ...

    def record(self, state: DocumentProcessingState) -> None:
        """覆盖写入该文献的处理状态；同一 ``doc_id`` 只保留一行。"""
        ...

"""批次报告：入库结果的值对象、状态归类与可留存的 JSON。

``DocumentReport``、``IngestReport``、``RunMetadata`` 与 ``ModelCallRecord``
是入库门面返回给调用方的值对象；``DocumentOutcome`` 是报告汇总与命令行共用
的状态归类，一篇文献属于哪一类只由 ``DocumentReport.status`` 推导一次。

报告落在语料目录表约定的 ``data/output/reports/``（见 AGENTS.md），具体目录由
调用方传入。文件名带运行开始时间与随机后缀，因此同一目录里的多次运行不会
互相覆盖，光看文件名就能区分批次。

报告除了各状态计数与明细，还记录每次模型调用的模型名、输入输出 token 数与
耗时，以及模型参数与本体指纹——足够复现一次运行的模型与参数组合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from idr_core.llm import ModelDescriptor
from idr_core.state import DocumentStatus
from idr_rotating.document import DocumentType
from idr_rotating.pending import (
    PendingObservation,
    PendingReview,
    aggregate_pending,
)

REPORT_SCHEMA_VERSION = 1


class DocumentOutcome(StrEnum):
    """一篇文献在一次入库运行里的归类，报告与命令行共用这一套词汇。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DocumentReport:
    path: Path
    doc_id: str
    evidence_segments: int
    entities: int
    relations: int
    rejected_entities: int = 0
    rejected_relations: int = 0
    error: str | None = None
    document_type: DocumentType | None = None
    quarantine_reason: str | None = None
    skipped: bool = False
    extraction_attempts: int = 0
    pending: tuple[PendingObservation, ...] = ()
    # 本次跳过的文献，此前一次处理的状态；用于在报告里保留原分类。
    prior_status: DocumentStatus | None = None
    # 解析或切分时的降级事实（丢行、bbox 回退整页等），不得静默。
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> DocumentOutcome:
        """本篇在报告里的归类；CLI 与汇总计数都只从这里读取。"""
        if self.skipped:
            # 此前已隔离的文献这次虽被跳过，分类仍是隔离，不能混进「跳过」。
            if self.prior_status is DocumentStatus.QUARANTINED:
                return DocumentOutcome.QUARANTINED
            return DocumentOutcome.SKIPPED
        if self.quarantine_reason is not None:
            return DocumentOutcome.QUARANTINED
        if self.error is not None:
            return DocumentOutcome.FAILED
        return DocumentOutcome.SUCCEEDED

    @property
    def pending_reviews(self) -> tuple[PendingReview, ...]:
        """本篇被拒绝抽取的待审条目（按类型与原因聚合）。"""
        return aggregate_pending(self.pending)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """一次入库运行的元数据，足以复现当次的模型与参数组合。"""

    source_dir: Path
    limit: int | None
    started_at: datetime
    finished_at: datetime
    model: ModelDescriptor | None
    ontology_version: str
    ontology_sha256: str
    # 抽取依据的「提示词模板 + 本体 + 分批参数」指纹，与处理状态里的
    # 跳过判定同源。
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    """一次模型调用的成本记录，挂在发起它的文献上。"""

    document_id: str
    purpose: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class IngestReport:
    documents: tuple[DocumentReport, ...]
    model_calls: tuple[ModelCallRecord, ...] = ()
    run: RunMetadata | None = None

    @property
    def evidence_segments(self) -> int:
        return sum(document.evidence_segments for document in self.documents)

    @property
    def entities(self) -> int:
        return sum(document.entities for document in self.documents)

    @property
    def relations(self) -> int:
        return sum(document.relations for document in self.documents)

    @property
    def failed(self) -> int:
        return self._count(DocumentOutcome.FAILED)

    @property
    def succeeded(self) -> int:
        return self._count(DocumentOutcome.SUCCEEDED)

    @property
    def quarantined(self) -> int:
        """隔离的篇数，含此前已隔离、本次被跳过的文献。"""
        return self._count(DocumentOutcome.QUARANTINED)

    @property
    def skipped(self) -> int:
        return self._count(DocumentOutcome.SKIPPED)

    @property
    def pending_reviews(self) -> tuple[PendingReview, ...]:
        """整批的待审队列；同一越界类型跨文献合并频次与例句。"""
        return aggregate_pending(
            observation
            for document in self.documents
            for observation in document.pending
        )

    @property
    def pending_review_count(self) -> int:
        """待审计条目的类别数，供命令行摘要使用。"""
        return len(self.pending_reviews)

    def _count(self, outcome: DocumentOutcome) -> int:
        return sum(1 for document in self.documents if document.status is outcome)


def write_report(report: IngestReport, *, output_dir: str | Path) -> Path:
    """把一次入库的结果写成 JSON 文件，返回文件路径。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(report)
    path = directory / f"ingest-{run_id}.json"
    path.write_text(
        json.dumps(_payload(report, run_id), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _run_id(report: IngestReport) -> str:
    started_at = report.run.started_at if report.run is not None else datetime.now(UTC)
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _payload(report: IngestReport, run_id: str) -> dict[str, object]:
    pending_review = _pending_review_payload(report)
    pending_occurrences = sum(
        review.frequency
        for document in report.documents
        for review in document.pending_reviews
    )
    usage_reported = [
        call
        for call in report.model_calls
        if call.input_tokens is not None and call.output_tokens is not None
    ]

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": _run_payload(report, run_id),
        "summary": {
            "documents": {
                "total": len(report.documents),
                "succeeded": report.succeeded,
                "failed": report.failed,
                "quarantined": report.quarantined,
                "skipped": report.skipped,
            },
            "pending_review": {
                "items": len(pending_review),
                "occurrences": pending_occurrences,
            },
            "evidence_segments": report.evidence_segments,
            "entities": report.entities,
            "relations": report.relations,
            "model_calls": {
                "count": len(report.model_calls),
                "usage_reported": len(usage_reported),
                "input_tokens": sum(call.input_tokens or 0 for call in usage_reported),
                "output_tokens": sum(
                    call.output_tokens or 0 for call in usage_reported
                ),
                "elapsed_seconds": round(
                    sum(call.elapsed_seconds for call in report.model_calls), 4
                ),
            },
        },
        "documents": [_document_payload(document) for document in report.documents],
        "pending_review": pending_review,
        "model_calls": [_model_call_payload(call) for call in report.model_calls],
    }


def _run_payload(report: IngestReport, run_id: str) -> dict[str, object]:
    run = report.run
    if run is None:
        return {
            "id": run_id,
            "started_at": None,
            "finished_at": None,
            "source_dir": None,
            "limit": None,
            "model": None,
            "ontology": None,
            "prompt": None,
        }

    model = None
    if run.model is not None:
        model = {
            "model": run.model.model,
            "parameters": dict(run.model.parameters),
        }

    return {
        "id": run_id,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat(),
        "source_dir": str(run.source_dir),
        "limit": run.limit,
        "model": model,
        "ontology": {
            "version": run.ontology_version,
            "sha256": run.ontology_sha256,
        },
        "prompt": {"sha256": run.prompt_sha256},
    }


def _document_payload(document: DocumentReport) -> dict[str, object]:
    return {
        "file_name": document.path.name,
        "path": str(document.path),
        "doc_id": document.doc_id,
        "status": document.status,
        "skipped": document.skipped,
        "prior_status": (
            None if document.prior_status is None else document.prior_status.value
        ),
        "document_type": (
            None if document.document_type is None else str(document.document_type)
        ),
        "error": document.error,
        "quarantine_reason": document.quarantine_reason,
        "evidence_segments": document.evidence_segments,
        "entities": document.entities,
        "relations": document.relations,
        "rejected_entities": document.rejected_entities,
        "rejected_relations": document.rejected_relations,
        "warnings": list(document.warnings),
    }


def _pending_review_payload(report: IngestReport) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for document in report.documents:
        for review in document.pending_reviews:
            entries.append(
                {
                    "document_id": document.doc_id,
                    "file_name": document.path.name,
                    "kind": review.kind.value,
                    "label": review.label,
                    "type": review.type,
                    "reason": review.reason.value,
                    "occurrences": review.occurrences,
                    "evidence_ids": list(review.evidence_ids),
                    "examples": [
                        {
                            "document_id": example.doc_id,
                            "evidence_id": example.evidence_id,
                            "text": example.text,
                        }
                        for example in review.examples
                    ],
                }
            )
    return entries


def _model_call_payload(call: ModelCallRecord) -> dict[str, object]:
    return {
        "document_id": call.document_id,
        "purpose": call.purpose,
        "model": call.model,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "elapsed_seconds": round(call.elapsed_seconds, 4),
    }

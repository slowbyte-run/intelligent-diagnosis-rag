"""批次报告文件：把一次入库的结果写成可留存、可核对的 JSON。

报告落在语料目录表约定的 ``data/output/reports/``（见 AGENTS.md），具体目录由
调用方传入。文件名带运行开始时间与随机后缀，因此同一目录里的多次运行不会
互相覆盖，光看文件名就能区分批次。

报告除了各状态计数与明细，还记录每次模型调用的模型名、输入输出 token 数与
耗时，以及模型参数与本体指纹——足够复现一次运行的模型与参数组合。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from idr_rotating.ingest import DocumentReport, IngestReport, ModelCallRecord

REPORT_SCHEMA_VERSION = 1


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
        item.occurrences
        for document in report.documents
        for item in document.pending_review
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
    }


def _document_payload(document: DocumentReport) -> dict[str, object]:
    return {
        "file_name": document.path.name,
        "path": str(document.path),
        "doc_id": document.doc_id,
        "status": document.status,
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
    }


def _pending_review_payload(report: IngestReport) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for document in report.documents:
        for item in document.pending_review:
            entries.append(
                {
                    "document_id": document.doc_id,
                    "file_name": document.path.name,
                    "kind": item.kind,
                    "label": item.label,
                    "type": item.type,
                    "reason": item.reason,
                    "occurrences": item.occurrences,
                    "evidence_ids": list(item.evidence_ids),
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

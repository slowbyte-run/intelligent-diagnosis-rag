"""领域包对外门面：一篇文献从 PDF 走到图谱。

调用方只依赖 ``ingest(source_dir, *, deps, limit)`` 与 ``IngestReport``；
解析器、模型、图存储、嵌入、向量存储与处理状态存储都由 ``deps`` 注入，
测试用内存假实现。

``_ingest_document`` 只按生命周期顺序编排具名步骤：跳过已完成文献 → 记录
处理中 → 准备（解析、扫描件隔离、类型判别、切分）→ 分批抽取 → 持久化
（先向量后图）→ 记录完成。四类结果（跳过、隔离、失败、完成）各由独立的
报告组装函数负责，异常只在这一个编排层分类。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from idr_core.documents import document_id
from idr_core.embedding import Embedder
from idr_core.graph import GraphStore
from idr_core.llm import (
    LLMClient,
    ModelDescriptor,
    call_model,
    describe_model,
)
from idr_core.models import EvidenceSegment
from idr_core.parsing import Parser
from idr_core.state import (
    DocumentProcessingState,
    DocumentStatus,
    IngestStateStore,
)
from idr_core.vector_store import VectorStore
from idr_rotating.document import (
    DocumentType,
    detect_document_type,
    scanned_document_reason,
)
from idr_rotating.errors import IngestError
from idr_rotating.extraction import (
    ExtractionError,
    ValidatedExtraction,
    batch_segments,
    extract_with_retry,
    merge_validated_extractions,
    validate_extraction,
)
from idr_rotating.graph_batch import build_graph_batch
from idr_rotating.ontology import (
    Ontology,
    load_ontology,
    ontology_fingerprint,
)
from idr_rotating.prompts import build_extraction_prompt, extraction_fingerprint
from idr_rotating.report import (
    DocumentReport,
    IngestReport,
    ModelCallRecord,
    RunMetadata,
)
from idr_rotating.segments import build_evidence_segments


@dataclass(frozen=True, slots=True)
class IngestDeps:
    """入库流程的全部外部依赖。

    ``state_store`` 可选：没有注入时按无状态运行（每次都处理），正式入库由
    CLI 注入 Postgres 实现，靠它跳过已完成的文献并断点续跑。
    """

    llm: LLMClient
    parser: Parser
    graph_store: GraphStore
    embedder: Embedder
    vector_store: VectorStore
    state_store: IngestStateStore | None = None


@dataclass(slots=True)
class _DocumentAttempt:
    """单篇入库的中间事实；异常时据此组装失败报告，不丢警告与尝试次数。"""

    doc_id: str = ""
    attempts: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    """准备步骤的结果；被隔离的文献没有证据段与类型。"""

    segments: tuple[EvidenceSegment, ...]
    document_type: DocumentType | None
    quarantine_reason: str | None = None


def ingest(
    source_dir: str | Path,
    *,
    deps: IngestDeps,
    limit: int | None = None,
) -> IngestReport:
    """把一个目录里的 PDF 入库，按文件名排序处理。

    ``limit`` 只处理前 N 篇，用于扩大语料前快速验证。注入了
    ``deps.state_store`` 时，按当前提示词与本体处理完成的文献直接跳过；
    提示词或本体变了就重做。被中断的文献没有完成记录，下次运行只补做这些。
    """
    started_at = datetime.now(UTC)
    root = Path(source_dir)
    if not root.is_dir():
        raise IngestError(f"语料目录不存在或不是目录：{root}")
    if limit is not None and limit < 1:
        raise IngestError(f"limit 必须是正整数，收到 {limit}")

    # 本体格式错误必须在这里失败，不能等处理到一半才暴露。
    ontology = load_ontology()
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name,
    )
    if limit is not None:
        paths = paths[:limit]

    fingerprint = extraction_fingerprint(ontology)
    model_calls: list[ModelCallRecord] = []
    documents = tuple(
        _ingest_document(path, deps, ontology, fingerprint, model_calls)
        for path in paths
    )
    return IngestReport(
        documents=documents,
        model_calls=tuple(model_calls),
        run=RunMetadata(
            source_dir=root.resolve(),
            limit=limit,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            model=describe_model(deps.llm),
            ontology_version=ontology.version,
            ontology_sha256=ontology_fingerprint(),
            prompt_sha256=fingerprint,
        ),
    )


def _ingest_document(
    path: Path,
    deps: IngestDeps,
    ontology: Ontology,
    fingerprint: str,
    model_calls: list[ModelCallRecord],
) -> DocumentReport:
    """编排一篇文献的生命周期；每一类结果都有独立的报告组装函数。"""
    attempt = _DocumentAttempt()
    try:
        attempt.doc_id = document_id(path.read_bytes())
        skipped = _skip_if_current(path, deps, attempt.doc_id, fingerprint)
        if skipped is not None:
            return skipped

        # 先记进行中再干活：进程被中断时这篇只剩「处理中」，下次会重做。
        _record_state(deps, _processing_state(attempt.doc_id, fingerprint))

        prepared = _prepare_document(deps, path, attempt)
        if prepared.quarantine_reason is not None:
            _record_state(
                deps,
                _quarantined_state(
                    attempt.doc_id, prepared.quarantine_reason, fingerprint
                ),
            )
            return _quarantined_report(
                path, attempt.doc_id, prepared.quarantine_reason, attempt.warnings
            )

        validated, attempt.attempts = _extract_document(
            deps, ontology, attempt.doc_id, prepared.segments, model_calls
        )
        _persist_document(deps, attempt.doc_id, prepared.segments, validated)
    except Exception as error:  # noqa: BLE001 - 单篇失败要记录原因，不能中断整批
        return _failed_report(path, deps, fingerprint, attempt, error)

    # 完成状态写在异常分支之外：状态存储在这里出错应直接暴露，而不是被归成
    # 该篇的抽取失败。
    _record_state(
        deps,
        _completed_state(attempt.doc_id, prepared, validated, fingerprint),
    )
    return _completed_report(path, prepared, validated, attempt)


def _skip_if_current(
    path: Path, deps: IngestDeps, doc_id: str, fingerprint: str
) -> DocumentReport | None:
    """已处理且抽取依据未变时直接给出跳过报告；否则返回 None。"""
    previous = _previous_state(deps, doc_id)
    if previous is None or not previous.is_current(fingerprint):
        return None
    return _skipped_report(path, doc_id, previous)


def _prepare_document(
    deps: IngestDeps, path: Path, attempt: _DocumentAttempt
) -> _PreparedDocument:
    """解析 → 扫描件隔离 → 类型判别 → 切分；警告随过程累积到 attempt 上。"""
    parsed = deps.parser.parse(path)
    attempt.warnings = parsed.warnings

    quarantine_reason = scanned_document_reason(parsed)
    if quarantine_reason is not None:
        # 扫描件隔离在抽取之前：不调模型、不写图，也不产出空证据段。
        return _PreparedDocument(
            segments=(), document_type=None, quarantine_reason=quarantine_reason
        )

    document_type = detect_document_type(parsed)
    segments = build_evidence_segments(attempt.doc_id, parsed, document_type)
    attempt.warnings += tuple(
        warning for segment in segments for warning in segment.warnings
    )
    if not segments:
        raise IngestError("未解析出可切分的文字，无法产出证据段")
    return _PreparedDocument(segments=segments, document_type=document_type)


def _extract_document(
    deps: IngestDeps,
    ontology: Ontology,
    doc_id: str,
    segments: tuple[EvidenceSegment, ...],
    model_calls: list[ModelCallRecord],
) -> tuple[ValidatedExtraction, int]:
    """按批抽取并合并；每次模型调用的成本都记到本篇文献上。"""
    recording_llm = _CallRecordingLLM(deps.llm, doc_id, model_calls)
    return _extract_batches(recording_llm, ontology, doc_id=doc_id, segments=segments)


def _persist_document(
    deps: IngestDeps,
    doc_id: str,
    segments: tuple[EvidenceSegment, ...],
    validated: ValidatedExtraction,
) -> None:
    """先写向量、再写图：向量失败时这篇文献的图谱数据一行都不会落库。

    两个存储跨不了服务事务，图存储自身按文档事务写入，因此不会半完成。
    """
    deps.vector_store.upsert_evidence_vectors(
        evidence_ids=[segment.id for segment in segments],
        vectors=deps.embedder.embed([segment.text for segment in segments]),
    )
    deps.graph_store.write_document(
        build_graph_batch(
            doc_id, segments, validated.entities, validated.relations
        )
    )


def _processing_state(doc_id: str, fingerprint: str) -> DocumentProcessingState:
    return DocumentProcessingState(
        doc_id=doc_id,
        status=DocumentStatus.PROCESSING,
        fingerprint=fingerprint,
    )


def _quarantined_state(
    doc_id: str, reason: str, fingerprint: str
) -> DocumentProcessingState:
    return DocumentProcessingState(
        doc_id=doc_id,
        status=DocumentStatus.QUARANTINED,
        reason=reason,
        fingerprint=fingerprint,
    )


def _completed_state(
    doc_id: str,
    prepared: _PreparedDocument,
    validated: ValidatedExtraction,
    fingerprint: str,
) -> DocumentProcessingState:
    return DocumentProcessingState(
        doc_id=doc_id,
        status=DocumentStatus.COMPLETED,
        evidence_segments=len(prepared.segments),
        entities=len(validated.entities),
        relations=len(validated.relations),
        fingerprint=fingerprint,
    )


def _skipped_report(
    path: Path, doc_id: str, state: DocumentProcessingState
) -> DocumentReport:
    """已处理完的文献不再重做，但把此前的结论带回报告里。"""
    return DocumentReport(
        path=path,
        doc_id=doc_id,
        evidence_segments=state.evidence_segments,
        entities=state.entities,
        relations=state.relations,
        quarantine_reason=(
            state.reason if state.status == DocumentStatus.QUARANTINED else None
        ),
        skipped=True,
        prior_status=state.status,
    )


def _quarantined_report(
    path: Path, doc_id: str, reason: str, warnings: tuple[str, ...]
) -> DocumentReport:
    return DocumentReport(
        path=path,
        doc_id=doc_id,
        evidence_segments=0,
        entities=0,
        relations=0,
        quarantine_reason=reason,
        warnings=warnings,
    )


def _failed_report(
    path: Path,
    deps: IngestDeps,
    fingerprint: str,
    attempt: _DocumentAttempt,
    error: Exception,
) -> DocumentReport:
    """记录失败原因并组装报告；失败不是终态，下次运行会重试这篇文献。"""
    if isinstance(error, ExtractionError):
        attempt.attempts = error.attempts
    message = f"{type(error).__name__}: {error}"
    if attempt.doc_id:
        _record_state(
            deps,
            DocumentProcessingState(
                doc_id=attempt.doc_id,
                status=DocumentStatus.FAILED,
                reason=message,
                fingerprint=fingerprint,
            ),
        )
    return DocumentReport(
        path=path,
        doc_id=attempt.doc_id,
        evidence_segments=0,
        entities=0,
        relations=0,
        error=message,
        extraction_attempts=attempt.attempts,
        warnings=attempt.warnings,
    )


def _completed_report(
    path: Path,
    prepared: _PreparedDocument,
    validated: ValidatedExtraction,
    attempt: _DocumentAttempt,
) -> DocumentReport:
    return DocumentReport(
        path=path,
        doc_id=attempt.doc_id,
        evidence_segments=len(prepared.segments),
        entities=len(validated.entities),
        relations=len(validated.relations),
        rejected_entities=validated.rejected_entities,
        rejected_relations=validated.rejected_relations,
        document_type=prepared.document_type,
        extraction_attempts=attempt.attempts,
        pending=validated.pending,
        warnings=attempt.warnings,
    )


def _extract_batches(
    llm: LLMClient,
    ontology: Ontology,
    *,
    doc_id: str,
    segments: tuple[EvidenceSegment, ...],
) -> tuple[ValidatedExtraction, int]:
    """逐批抽取并合并；任何一批重试后仍失败，整篇失败并指明是哪一批。

    每批单独校验证据引用：引用了本批之外证据段的抽取照旧判失败，不会因为
    同一篇文献的其他批次引用过就放行。
    """
    batches = batch_segments(segments)
    parts: list[ValidatedExtraction] = []
    attempts = 0
    for position, batch in enumerate(batches, start=1):
        try:
            extraction, batch_attempts = extract_with_retry(
                llm, build_extraction_prompt(ontology, batch)
            )
        except ExtractionError as error:
            attempts += error.attempts
            # 失败批次之前的抽取结果不落库：一篇文献要么完整入库，要么整篇
            # 记为失败并在下次运行重试，避免图谱里留下没有报告的半份知识。
            raise ExtractionError(
                f"第 {position}/{len(batches)} 批抽取失败"
                f"（累计尝试 {attempts} 次）：{error}",
                attempts=attempts,
            ) from None
        attempts += batch_attempts
        parts.append(
            validate_extraction(
                extraction,
                ontology,
                doc_id=doc_id,
                evidence_text={segment.id: segment.text for segment in batch},
            )
        )
    return (
        merge_validated_extractions(
            parts,
            doc_id=doc_id,
            evidence_text={segment.id: segment.text for segment in segments},
        ),
        attempts,
    )


class _CallRecordingLLM:
    """包一层模型客户端，把每次调用（含重试）的成本记到当前文献上。"""

    def __init__(
        self,
        llm: LLMClient,
        document_id: str,
        model_calls: list[ModelCallRecord],
    ) -> None:
        self._llm = llm
        self._document_id = document_id
        self._model_calls = model_calls

    def complete(self, prompt: str) -> str:
        text, call = call_model(self._llm, prompt)
        self._model_calls.append(
            ModelCallRecord(
                document_id=self._document_id,
                purpose="extraction",
                model=call.model,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                elapsed_seconds=call.elapsed_seconds,
            )
        )
        return text


def _previous_state(
    deps: IngestDeps, doc_id: str
) -> DocumentProcessingState | None:
    if deps.state_store is None:
        return None
    return deps.state_store.get(doc_id)


def _record_state(deps: IngestDeps, state: DocumentProcessingState) -> None:
    if deps.state_store is not None:
        deps.state_store.record(state)

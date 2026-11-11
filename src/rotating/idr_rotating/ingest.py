"""领域包对外门面：一篇文献从 PDF 走到图谱。

调用方只依赖 ``ingest(source_dir, *, deps, limit)`` 与 ``IngestReport``；
解析器、模型、图存储、嵌入、向量存储与处理状态存储都由 ``deps`` 注入，
测试用内存假实现。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from idr_core.embedding import Embedder
from idr_core.graph import (
    ConceptEvidenceLink,
    ConceptNode,
    ConceptRelation,
    GraphBatch,
    GraphStore,
    Provenance,
)
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
from idr_rotating.extraction import (
    ExtractedRelation,
    ExtractionError,
    ValidatedEntity,
    ValidatedExtraction,
    batch_segments,
    extract_with_retry,
    merge_validated_extractions,
    validate_extraction,
)
from idr_rotating.ontology import (
    Ontology,
    load_ontology,
    ontology_fingerprint,
)
from idr_rotating.pending import (
    PendingObservation,
    PendingReview,
    aggregate_pending,
)
from idr_rotating.prompts import build_extraction_prompt, extraction_fingerprint
from idr_rotating.segments import build_evidence_segments


class IngestError(RuntimeError):
    """入库过程中可以直接展示给人的错误。"""


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
    def status(
        self,
    ) -> Literal["succeeded", "failed", "quarantined", "skipped"]:
        if self.skipped:
            # 此前已隔离的文献这次虽被跳过，分类仍是隔离，不能混进「跳过」。
            if self.prior_status is DocumentStatus.QUARANTINED:
                return "quarantined"
            return "skipped"
        if self.quarantine_reason is not None:
            return "quarantined"
        if self.error is not None:
            return "failed"
        return "succeeded"

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
        return sum(1 for document in self.documents if document.status == "failed")

    @property
    def succeeded(self) -> int:
        return sum(1 for document in self.documents if document.status == "succeeded")

    @property
    def quarantined(self) -> int:
        """隔离的篇数，含此前已隔离、本次被跳过的文献。"""
        return sum(
            1 for document in self.documents if document.status == "quarantined"
        )

    @property
    def skipped(self) -> int:
        return sum(1 for document in self.documents if document.status == "skipped")

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
    doc_id = ""
    attempts = 0
    warnings: tuple[str, ...] = ()
    try:
        doc_id = _doc_id(path.read_bytes())
        previous = _previous_state(deps, doc_id)
        if previous is not None and previous.is_current(fingerprint):
            return _skipped_report(path, doc_id, previous)

        # 先记进行中再干活：进程被中断时这篇只剩「处理中」，下次会重做。
        _record_state(
            deps,
            DocumentProcessingState(
                doc_id=doc_id,
                status=DocumentStatus.PROCESSING,
                fingerprint=fingerprint,
            ),
        )
        parsed = deps.parser.parse(path)
        warnings = parsed.warnings
        quarantine_reason = scanned_document_reason(parsed)
        if quarantine_reason is not None:
            # 扫描件隔离在抽取之前：不调模型、不写图，也不产出空证据段。
            _record_state(
                deps,
                DocumentProcessingState(
                    doc_id=doc_id,
                    status=DocumentStatus.QUARANTINED,
                    reason=quarantine_reason,
                    fingerprint=fingerprint,
                ),
            )
            return DocumentReport(
                path=path,
                doc_id=doc_id,
                evidence_segments=0,
                entities=0,
                relations=0,
                quarantine_reason=quarantine_reason,
                warnings=warnings,
            )

        document_type = detect_document_type(parsed)
        segments = build_evidence_segments(doc_id, parsed, document_type)
        warnings = warnings + tuple(
            warning for segment in segments for warning in segment.warnings
        )
        if not segments:
            raise IngestError("未解析出可切分的文字，无法产出证据段")

        recording_llm = _CallRecordingLLM(deps.llm, doc_id, model_calls)
        validated, attempts = _extract_batches(
            recording_llm,
            ontology,
            doc_id=doc_id,
            segments=segments,
        )
        # 先写向量、再写图：向量失败时这篇文献的图谱数据一行都不会落库。
        # 两个存储跨不了服务事务，图存储自身按文档事务写入，因此不会半完成。
        deps.vector_store.upsert_evidence_vectors(
            evidence_ids=[segment.id for segment in segments],
            vectors=deps.embedder.embed([segment.text for segment in segments]),
        )
        deps.graph_store.write_document(
            _build_batch(doc_id, segments, validated.entities, validated.relations)
        )
    except Exception as error:  # noqa: BLE001 - 单篇失败要记录原因，不能中断整批
        if isinstance(error, ExtractionError):
            attempts = error.attempts
        message = f"{type(error).__name__}: {error}"
        if doc_id:
            # 失败不是终态：下次运行会重试这篇文献。
            _record_state(
                deps,
                DocumentProcessingState(
                    doc_id=doc_id,
                    status=DocumentStatus.FAILED,
                    reason=message,
                    fingerprint=fingerprint,
                ),
            )
        return DocumentReport(
            path=path,
            doc_id=doc_id,
            evidence_segments=0,
            entities=0,
            relations=0,
            error=message,
            extraction_attempts=attempts,
            warnings=warnings,
        )

    _record_state(
        deps,
        DocumentProcessingState(
            doc_id=doc_id,
            status=DocumentStatus.COMPLETED,
            evidence_segments=len(segments),
            entities=len(validated.entities),
            relations=len(validated.relations),
            fingerprint=fingerprint,
        ),
    )
    return DocumentReport(
        path=path,
        doc_id=doc_id,
        evidence_segments=len(segments),
        entities=len(validated.entities),
        relations=len(validated.relations),
        rejected_entities=validated.rejected_entities,
        rejected_relations=validated.rejected_relations,
        document_type=document_type,
        extraction_attempts=attempts,
        pending=validated.pending,
        warnings=warnings,
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


def _doc_id(content: bytes) -> str:
    """文献标识由内容哈希推导，同一文件重复入库得到同一标识。"""
    return hashlib.sha256(content).hexdigest()[:16]


def _build_batch(
    doc_id: str,
    segments: tuple[EvidenceSegment, ...],
    entities: tuple[ValidatedEntity, ...],
    relations: tuple[ExtractedRelation, ...],
) -> GraphBatch:
    concepts: list[ConceptNode] = []
    links: list[ConceptEvidenceLink] = []
    concept_by_name: dict[str, ConceptNode] = {}

    for entity in entities:
        concept = ConceptNode(
            id=f"{doc_id}:concept:{entity.key}",
            name=entity.name,
            type=entity.type,
            doc_id=doc_id,
            attributes=dict(entity.attributes),
        )
        concepts.append(concept)
        concept_by_name.setdefault(entity.name, concept)
        for evidence_id in entity.evidence_ids:
            links.append(
                ConceptEvidenceLink(
                    concept_id=concept.id,
                    provenance=Provenance(
                        evidence_ids=(evidence_id,),
                        confidence=entity.confidence,
                        source="extracted",
                    ),
                )
            )

    graph_relations: list[ConceptRelation] = []
    for relation in relations:
        head = concept_by_name.get(relation.head)
        tail = concept_by_name.get(relation.tail)
        if head is None or tail is None:
            # 端点缺失是数据错误（AGENTS.md 第 5 条），不得静默丢边；
            # 抛出让本篇记为失败，报告里能看见是哪条关系出的问题。
            raise IngestError(
                f"关系 {relation.type} 的端点未写入概念节点："
                f"{relation.head} → {relation.tail}"
            )
        graph_relations.append(
            ConceptRelation(
                head_id=head.id,
                type=relation.type,
                tail_id=tail.id,
                provenance=Provenance(
                    evidence_ids=tuple(relation.evidence_ids),
                    confidence=relation.confidence,
                    source="extracted",
                ),
            )
        )

    return GraphBatch(
        doc_id=doc_id,
        concepts=tuple(concepts),
        evidence=segments,
        relations=tuple(graph_relations),
        evidence_links=tuple(links),
    )

"""领域包对外门面：一篇文献从 PDF 走到图谱。

调用方只依赖 ``ingest(source_dir, *, deps, limit)`` 与 ``IngestReport``；
解析器、模型、图存储、嵌入、向量存储与处理状态存储都由 ``deps`` 注入，
测试用内存假实现。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from idr_core.embedding import Embedder
from idr_core.graph import (
    ConceptEvidenceLink,
    ConceptNode,
    ConceptRelation,
    GraphBatch,
    GraphStore,
)
from idr_core.llm import LLMClient
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
    ExtractedEntity,
    ExtractedRelation,
    build_extraction_prompt,
    parse_extraction,
    validate_extraction,
)
from idr_rotating.ontology import Ontology, load_ontology
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


@dataclass(frozen=True, slots=True)
class IngestReport:
    documents: tuple[DocumentReport, ...]

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
        return sum(1 for document in self.documents if document.error is not None)

    @property
    def quarantined(self) -> int:
        """本次运行新判定为隔离的篇数；跳过的文献不重复计数。"""
        return sum(
            1
            for document in self.documents
            if document.quarantine_reason is not None and not document.skipped
        )

    @property
    def skipped(self) -> int:
        return sum(1 for document in self.documents if document.skipped)


def ingest(
    source_dir: str | Path,
    *,
    deps: IngestDeps,
    limit: int | None = None,
) -> IngestReport:
    """把一个目录里的 PDF 入库，按文件名排序处理。

    ``limit`` 只处理前 N 篇，用于扩大语料前快速验证。注入了
    ``deps.state_store`` 时，已完成的文献直接跳过；被中断的文献没有完成
    记录，下次运行只补做这些。
    """
    root = Path(source_dir)
    if not root.is_dir():
        raise IngestError(f"语料目录不存在或不是目录：{root}")
    if limit is not None and limit < 1:
        raise IngestError(f"limit 必须是正整数，收到 {limit}")

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

    return IngestReport(
        documents=tuple(_ingest_document(path, deps, ontology) for path in paths)
    )


def _ingest_document(
    path: Path, deps: IngestDeps, ontology: Ontology
) -> DocumentReport:
    doc_id = ""
    try:
        doc_id = _doc_id(path.read_bytes())
        previous = _previous_state(deps, doc_id)
        if previous is not None and previous.settled:
            return _skipped_report(path, doc_id, previous)

        # 先记进行中再干活：进程被中断时这篇只剩「处理中」，下次会重做。
        _record_state(
            deps,
            DocumentProcessingState(
                doc_id=doc_id, status=DocumentStatus.PROCESSING
            ),
        )
        parsed = deps.parser.parse(path)
        quarantine_reason = scanned_document_reason(parsed)
        if quarantine_reason is not None:
            # 扫描件隔离在抽取之前：不调模型、不写图，也不产出空证据段。
            _record_state(
                deps,
                DocumentProcessingState(
                    doc_id=doc_id,
                    status=DocumentStatus.QUARANTINED,
                    reason=quarantine_reason,
                ),
            )
            return DocumentReport(
                path=path,
                doc_id=doc_id,
                evidence_segments=0,
                entities=0,
                relations=0,
                quarantine_reason=quarantine_reason,
            )

        document_type = detect_document_type(parsed)
        segments = build_evidence_segments(doc_id, parsed, document_type)
        if not segments:
            raise IngestError("未解析出可切分的文字，无法产出证据段")

        extraction = parse_extraction(
            deps.llm.complete(build_extraction_prompt(ontology, segments))
        )
        validated = validate_extraction(
            extraction,
            ontology,
            available_evidence_ids={segment.id for segment in segments},
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
        message = f"{type(error).__name__}: {error}"
        if doc_id:
            # 失败不是终态：下次运行会重试这篇文献。
            _record_state(
                deps,
                DocumentProcessingState(
                    doc_id=doc_id,
                    status=DocumentStatus.FAILED,
                    reason=message,
                ),
            )
        return DocumentReport(
            path=path,
            doc_id=doc_id,
            evidence_segments=0,
            entities=0,
            relations=0,
            error=message,
        )

    _record_state(
        deps,
        DocumentProcessingState(
            doc_id=doc_id,
            status=DocumentStatus.COMPLETED,
            evidence_segments=len(segments),
            entities=len(validated.entities),
            relations=len(validated.relations),
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
    )


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
        evidence_segments=0,
        entities=0,
        relations=0,
        quarantine_reason=(
            state.reason if state.status == DocumentStatus.QUARANTINED else None
        ),
        skipped=True,
    )


def _doc_id(content: bytes) -> str:
    """文献标识由内容哈希推导，同一文件重复入库得到同一标识。"""
    return hashlib.sha256(content).hexdigest()[:16]


def _build_batch(
    doc_id: str,
    segments: tuple[EvidenceSegment, ...],
    entities: tuple[ExtractedEntity, ...],
    relations: tuple[ExtractedRelation, ...],
) -> GraphBatch:
    concepts: list[ConceptNode] = []
    links: list[ConceptEvidenceLink] = []
    concept_by_name: dict[str, ConceptNode] = {}

    for index, entity in enumerate(entities):
        concept = ConceptNode(
            id=f"{doc_id}:concept:{index}",
            name=entity.name,
            type=entity.type,
            doc_id=doc_id,
        )
        concepts.append(concept)
        concept_by_name.setdefault(entity.name, concept)
        for evidence_id in entity.evidence_ids:
            links.append(
                ConceptEvidenceLink(
                    concept_id=concept.id,
                    evidence_id=evidence_id,
                    evidence_ids=(evidence_id,),
                    confidence=entity.confidence,
                    source="extracted",
                )
            )

    graph_relations: list[ConceptRelation] = []
    for relation in relations:
        head = concept_by_name.get(relation.head)
        tail = concept_by_name.get(relation.tail)
        if head is None or tail is None:
            continue
        graph_relations.append(
            ConceptRelation(
                head_id=head.id,
                type=relation.type,
                tail_id=tail.id,
                evidence_ids=tuple(relation.evidence_ids),
                confidence=relation.confidence,
                source="extracted",
            )
        )

    return GraphBatch(
        doc_id=doc_id,
        concepts=tuple(concepts),
        evidence=segments,
        relations=tuple(graph_relations),
        evidence_links=tuple(links),
    )

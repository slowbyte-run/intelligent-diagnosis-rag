"""领域包对外门面：一篇文献从 PDF 走到图谱。

调用方只依赖 ``ingest(source_dir, *, deps, limit)`` 与 ``IngestReport``；
解析器、模型、图存储、嵌入与向量存储都由 ``deps`` 注入，测试用内存假实现。
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
)
from idr_core.llm import LLMClient, ModelDescriptor, call_model, describe_model
from idr_core.models import EvidenceSegment
from idr_core.parsing import Parser
from idr_core.vector_store import VectorStore
from idr_rotating.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    RejectedExtraction,
    build_extraction_prompt,
    parse_extraction,
    validate_extraction,
)
from idr_rotating.ontology import (
    Ontology,
    load_ontology,
    ontology_fingerprint,
)
from idr_rotating.segments import build_evidence_segments


class IngestError(RuntimeError):
    """入库过程中可以直接展示给人的错误。"""


class QuarantineError(IngestError):
    """文档被隔离：不进图谱，但仍要在报告里记下原因。"""


@dataclass(frozen=True, slots=True)
class IngestDeps:
    """入库流程的全部外部依赖。"""

    llm: LLMClient
    parser: Parser
    graph_store: GraphStore
    embedder: Embedder
    vector_store: VectorStore


@dataclass(frozen=True, slots=True)
class PendingReviewItem:
    """一条等待人工审计的越界抽取；同名同因合并计频次。"""

    kind: str
    label: str
    type: str
    reason: str
    occurrences: int
    evidence_ids: tuple[str, ...]


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
    quarantined: bool = False
    quarantine_reason: str | None = None
    pending_review: tuple[PendingReviewItem, ...] = ()

    @property
    def status(self) -> Literal["succeeded", "failed", "quarantined"]:
        if self.quarantined:
            return "quarantined"
        if self.error is not None:
            return "failed"
        return "succeeded"


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
        return sum(
            1 for document in self.documents if document.status == "quarantined"
        )

    @property
    def pending_review(self) -> int:
        return sum(len(document.pending_review) for document in self.documents)


def ingest(
    source_dir: str | Path,
    *,
    deps: IngestDeps,
    limit: int | None = None,
) -> IngestReport:
    started_at = datetime.now(UTC)
    root = Path(source_dir)
    if not root.is_dir():
        raise IngestError(f"语料目录不存在或不是目录：{root}")

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

    model_calls: list[ModelCallRecord] = []
    documents = tuple(
        _ingest_document(path, deps, ontology, model_calls) for path in paths
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
        ),
    )


def _ingest_document(
    path: Path,
    deps: IngestDeps,
    ontology: Ontology,
    model_calls: list[ModelCallRecord],
) -> DocumentReport:
    doc_id = ""
    try:
        doc_id = _doc_id(path.read_bytes())
        parsed = deps.parser.parse(path)
        segments = build_evidence_segments(doc_id, parsed)
        if not segments:
            raise QuarantineError("未解析出文字层，可能是扫描件或空文档，未产出证据段")

        prompt = build_extraction_prompt(ontology, segments)
        text, call = call_model(deps.llm, prompt)
        model_calls.append(
            ModelCallRecord(
                document_id=doc_id,
                purpose="extraction",
                model=call.model,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                elapsed_seconds=call.elapsed_seconds,
            )
        )
        extraction = parse_extraction(text)
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
    except QuarantineError as error:
        return DocumentReport(
            path=path,
            doc_id=doc_id,
            evidence_segments=0,
            entities=0,
            relations=0,
            quarantined=True,
            quarantine_reason=str(error),
        )
    except Exception as error:  # noqa: BLE001 - 单篇失败要记录原因，不能中断整批
        return DocumentReport(
            path=path,
            doc_id=doc_id,
            evidence_segments=0,
            entities=0,
            relations=0,
            error=f"{type(error).__name__}: {error}",
        )

    return DocumentReport(
        path=path,
        doc_id=doc_id,
        evidence_segments=len(segments),
        entities=len(validated.entities),
        relations=len(validated.relations),
        rejected_entities=validated.rejected_entities,
        rejected_relations=validated.rejected_relations,
        pending_review=_pending_review(validated.rejected),
    )


def _pending_review(
    rejected: tuple[RejectedExtraction, ...],
) -> tuple[PendingReviewItem, ...]:
    """按「类型 + 名称 + 原因」合并被拒抽取，保留首次出现顺序。"""
    occurrences: dict[tuple[str, str, str, str], int] = {}
    evidence_ids: dict[tuple[str, str, str, str], list[str]] = {}
    for item in rejected:
        key = (item.kind, item.label, item.type, item.reason)
        occurrences[key] = occurrences.get(key, 0) + 1
        known = evidence_ids.setdefault(key, [])
        for evidence_id in item.evidence_ids:
            if evidence_id not in known:
                known.append(evidence_id)

    return tuple(
        PendingReviewItem(
            kind=kind,
            label=label,
            type=type_id,
            reason=reason,
            occurrences=count,
            evidence_ids=tuple(evidence_ids[(kind, label, type_id, reason)]),
        )
        for (kind, label, type_id, reason), count in occurrences.items()
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

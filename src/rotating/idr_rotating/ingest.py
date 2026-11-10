"""领域包对外门面：一篇文献从 PDF 走到图谱。

调用方只依赖 ``ingest(source_dir, *, deps, limit)`` 与 ``IngestReport``；
解析器、模型、图存储、嵌入与向量存储都由 ``deps`` 注入，测试用内存假实现。
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
from idr_core.vector_store import VectorStore
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
    """入库流程的全部外部依赖。"""

    llm: LLMClient
    parser: Parser
    graph_store: GraphStore
    embedder: Embedder
    vector_store: VectorStore


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


def ingest(
    source_dir: str | Path,
    *,
    deps: IngestDeps,
    limit: int | None = None,
) -> IngestReport:
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

    return IngestReport(
        documents=tuple(_ingest_document(path, deps, ontology) for path in paths)
    )


def _ingest_document(
    path: Path, deps: IngestDeps, ontology: Ontology
) -> DocumentReport:
    doc_id = ""
    try:
        doc_id = _doc_id(path.read_bytes())
        parsed = deps.parser.parse(path)
        segments = build_evidence_segments(doc_id, parsed)
        if not segments:
            raise IngestError("未解析出文字层，无法产出证据段")

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

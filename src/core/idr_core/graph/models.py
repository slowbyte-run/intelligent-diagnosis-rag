"""图谱写入模型：概念节点、证据节点、关系边。

概念到证据的边给每条概念挂上出处；概念之间的关系边直接携带证据属性
（见 ADR-0004：论断的证据挂在边属性上，不建断言节点）。
"""

from dataclasses import dataclass

from idr_core.models import EvidenceSegment


@dataclass(frozen=True, slots=True)
class ConceptNode:
    id: str
    name: str
    type: str
    doc_id: str


@dataclass(frozen=True, slots=True)
class ConceptRelation:
    head_id: str
    type: str
    tail_id: str
    evidence_ids: tuple[str, ...]
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class ConceptEvidenceLink:
    concept_id: str
    evidence_id: str
    evidence_ids: tuple[str, ...]
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class GraphBatch:
    doc_id: str
    concepts: tuple[ConceptNode, ...]
    evidence: tuple[EvidenceSegment, ...]
    relations: tuple[ConceptRelation, ...]
    evidence_links: tuple[ConceptEvidenceLink, ...]

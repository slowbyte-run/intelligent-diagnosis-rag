"""图谱写入模型：概念节点、证据节点、关系边。

概念到证据的边给每条概念挂上出处；概念之间的关系边直接携带证据属性
（见 ADR-0004：论断的证据挂在边属性上，不建断言节点）。
"""

from dataclasses import dataclass, field

from idr_core.models import EvidenceSegment


@dataclass(frozen=True, slots=True)
class Provenance:
    """一条断言/链接的出处：证据段、抽取置信度与来源。

    关系边与概念—证据边共用这一份形状，避免两处各写一遍三个字段。
    """

    evidence_ids: tuple[str, ...]
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class ConceptNode:
    id: str
    name: str
    type: str
    doc_id: str
    # 本体为该实体类型声明的属性（规范标识），与具体领域无关。
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConceptRelation:
    head_id: str
    type: str
    tail_id: str
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ConceptEvidenceLink:
    concept_id: str
    provenance: Provenance

    @property
    def evidence_id(self) -> str:
        """链接指向的证据段：概念—证据边永远是单条证据。"""
        return self.provenance.evidence_ids[0]


@dataclass(frozen=True, slots=True)
class GraphBatch:
    doc_id: str
    concepts: tuple[ConceptNode, ...]
    evidence: tuple[EvidenceSegment, ...]
    relations: tuple[ConceptRelation, ...]
    evidence_links: tuple[ConceptEvidenceLink, ...]

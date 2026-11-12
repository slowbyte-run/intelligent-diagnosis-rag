"""图谱写入模型：概念节点、证据节点、关系边。

概念到证据的边给每条概念挂上出处；概念之间的关系边直接携带证据属性
（见 ADR-0004：论断的证据挂在边属性上，不建断言节点）。
"""

from dataclasses import dataclass, field

from idr_core.anchors import EvidenceAnchor
from idr_core.models import EvidenceSegment


@dataclass(frozen=True, slots=True)
class Provenance:
    """一条断言/链接的出处：来自哪篇文献、哪段证据、置信度与来源。

    关系边与概念—证据边共用这一份形状，避免两处各写一遍这几个字段。
    ``doc_id`` 是每条边的归属：同一对概念在不同文献里的断言各留一条并行边，
    重跑某篇文献时只替换该文献贡献的边。

    ``anchors`` 与 ``evidence_ids`` 平行：每个锚点指明它属于哪一段证据，
    指向的区间是入库时固化的逐字出处。某个证据段没有锚点是合法状态（旧数据
    或模型没给出片段），该段仍按整段引用；但锚点不得指向没被引用的证据段。
    """

    doc_id: str
    evidence_ids: tuple[str, ...]
    confidence: float
    source: str
    anchors: tuple[EvidenceAnchor, ...] = ()

    def __post_init__(self) -> None:
        anchor_evidence_ids = [anchor.evidence_id for anchor in self.anchors]
        for anchor in self.anchors:
            if anchor.evidence_id not in self.evidence_ids:
                raise ValueError(
                    f"锚点指向未被引用的证据段 {anchor.evidence_id}："
                    f"证据段清单为 {self.evidence_ids}"
                )
        if len(anchor_evidence_ids) != len(set(anchor_evidence_ids)):
            raise ValueError(
                f"同一条证据段出现了多个锚点：{self.evidence_ids}"
            )


@dataclass(frozen=True, slots=True)
class ConceptNode:
    id: str
    name: str
    type: str
    # ``None`` 表示概念身份由本体规范实例确定，节点跨文献共享，
    # 不带任何单篇文献的归属；非 None 时节点限定在该文献内。
    doc_id: str | None
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
    """一篇文献向图谱提交的全部贡献。

    边与文档作用域节点的 ``doc_id`` 必须与批次一致：图存储按它删除上一篇
    文献的贡献，写错归属会让别的文献的数据被误删或漏删。
    """

    doc_id: str
    concepts: tuple[ConceptNode, ...]
    evidence: tuple[EvidenceSegment, ...]
    relations: tuple[ConceptRelation, ...]
    evidence_links: tuple[ConceptEvidenceLink, ...]

    def __post_init__(self) -> None:
        for concept in self.concepts:
            if concept.doc_id is not None and concept.doc_id != self.doc_id:
                raise ValueError(
                    f"概念节点 {concept.id} 的 doc_id={concept.doc_id} "
                    f"与批次 {self.doc_id} 不一致"
                )
        for relation in self.relations:
            if relation.provenance.doc_id != self.doc_id:
                raise ValueError(
                    f"关系 {relation.type} 的 doc_id="
                    f"{relation.provenance.doc_id} 与批次 {self.doc_id} 不一致"
                )
        for link in self.evidence_links:
            if link.provenance.doc_id != self.doc_id:
                raise ValueError(
                    f"概念 {link.concept_id} 的证据边 doc_id="
                    f"{link.provenance.doc_id} 与批次 {self.doc_id} 不一致"
                )

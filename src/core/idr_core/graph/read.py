"""图检索的读模型与端口。

图谱里同一对概念在每篇文献各留一条并行边（ADR-0004 的修订记录）。检索不能
把并行边当成多条结论，必须先按 ``(源, 类型, 目标)`` 聚合：证据取并集，置信度
取各边最大值。没有 ``evidence_ids`` 的边不进结果——没有出处的结论不允许出现在
答案里。

端口只返回图谱里真实存在的边与节点，不做领域判断；哪些关系类型可以走、走几跳
由领域包决定，聚合与无证据边的排除在这里统一做，两个领域共用。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from idr_core.anchors import EvidenceAnchor
from idr_core.models import CharSpan


@dataclass(frozen=True, slots=True)
class Concept:
    """图谱里的概念节点：标识、名称与本体类型。"""

    id: str
    name: str
    type: str


@dataclass(frozen=True, slots=True)
class FactEdge:
    """一条未经聚合的事实边；同一 ``(head, relation, tail)`` 可有多条。

    ``anchors`` 是这条断言在证据段内的逐字出处（``(evidence_id, start, end)``），
    与 ``evidence_ids`` 平行但可以更少：锚点上线前写入的边没有锚点，读取侧
    必须把「没有锚点」当成整段引用，而不是把边丢掉。锚点实现见
    :mod:`idr_core.anchors`；图里以四条平行数组属性持久化。
    """

    head_id: str
    relation: str
    tail_id: str
    evidence_ids: tuple[str, ...]
    confidence: float
    doc_id: str
    anchors: tuple[EvidenceAnchor, ...] = ()


@dataclass(frozen=True, slots=True)
class AggregatedFact:
    """聚合后的一条事实：同一概念对的全部并行边合成一条。

    ``anchors`` 与 ``evidence_ids`` 平行（每个证据段至多一个锚点），按证据段
    首次出现的顺序排列；没有锚点的证据段不在其中——引用层据此退回整段。
    """

    head_id: str
    relation: str
    tail_id: str
    evidence_ids: tuple[str, ...]
    confidence: float
    anchors: tuple[EvidenceAnchor, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """证据段在图谱里的定位信息；文献名由调用方按 ``doc_id`` 补上。

    ``charspan`` 是证据段在所属页面文字中的字符区间。引用只要展示片段
    （锚点）时它用不上；要把相邻证据段并进展示上下文，就必须知道每段在
    页面上的确切位置，才能判断两段是否连续、拼接结果是否仍是页面文字的
    连续切片。
    """

    id: str
    doc_id: str
    page: int
    text: str
    charspan: CharSpan


class GraphReader(Protocol):
    """图检索端口：按概念取一跳事实，再按标识批量取概念与证据。

    实现方返回**未聚合**的并行边，由 ``aggregate_facts`` 统一聚合；关系标识
    统一用小写本体标识，不暴露图数据库自己的类型名。
    """

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        """从该概念出发、类型在清单内的一跳事实边；不存在时返回空。"""
        ...

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        """按标识批量取概念节点；不存在的标识不出现在结果里。"""
        ...

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        """按标识批量取证据段；不存在的标识不出现在结果里。"""
        ...

    def evidence_by_page(self, doc_id: str, page: int) -> tuple[EvidenceRecord, ...]:
        """取一篇文献某一页的全部证据段，按页面字符区间升序。

        引用层只有需要并入上下文时才调用它（短锚点），且只用来找同一页上
        与锚点相邻的证据段；跨页的段不在这里返回，合并也就不会跨页。
        """
        ...


class EvidenceTextSearch(Protocol):
    """证据段原文的词面检索端口；与图遍历分开，实现方可以是同一个存储。

    只返回按相关度排好序的证据段标识，原文仍由调用方按标识从图里取——
    词面检索不是第二个事实源。没有实现这个端口的图存储只能走纯向量召回。
    """

    def search_evidence(
        self, terms: Sequence[str], *, limit: int
    ) -> tuple[str, ...]:
        """返回文本包含任一 ``terms`` 的证据段标识，命中词多者靠前。"""
        ...


def aggregate_facts(edges: Iterable[FactEdge]) -> tuple[AggregatedFact, ...]:
    """按 ``(源, 类型, 目标)`` 聚合并行边，丢掉没有证据的边。

    结果保持边首次出现的顺序：关系类型的先后由检索编排决定（例如先观测、
    后原因），聚合层不重排。证据并集同样按首次出现的顺序去重。
    """
    merged: dict[
        tuple[str, str, str],
        tuple[list[str], dict[str, EvidenceAnchor], float],
    ] = {}
    for edge in edges:
        if not edge.evidence_ids:
            continue
        key = (edge.head_id, edge.relation, edge.tail_id)
        if key not in merged:
            merged[key] = ([], {}, edge.confidence)
        evidence_ids, anchors, confidence = merged[key]
        for evidence_id in edge.evidence_ids:
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        for anchor in edge.anchors:
            # 同一证据段只会由同一篇文献的一条边引用；万一聚合时重复出现，
            # 保留先到的那个锚点，顺序因此与证据并集一致。
            anchors.setdefault(anchor.evidence_id, anchor)
        merged[key] = (
            evidence_ids,
            anchors,
            max(confidence, edge.confidence),
        )

    return tuple(
        AggregatedFact(
            head_id=head_id,
            relation=relation,
            tail_id=tail_id,
            evidence_ids=tuple(evidence_ids),
            confidence=confidence,
            anchors=tuple(
                anchor
                for evidence_id in evidence_ids
                if (anchor := anchors.get(evidence_id)) is not None
            ),
        )
        for (head_id, relation, tail_id), (
            evidence_ids,
            anchors,
            confidence,
        ) in merged.items()
    )

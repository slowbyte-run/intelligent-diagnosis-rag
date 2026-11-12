"""快照读端口：把一份已加载的图谱快照适配成 ``GraphReader``。

快照（:mod:`idr_core.graph.snapshot`）是数据；本模块是它的读侧。适配器只做
按标识与关系类型的过滤，规则与在线图存储的读端口一致：

- ``facts_from`` 按起点精确匹配、按关系类型过滤，返回**未聚合**的并行边，
  聚合仍由 :func:`idr_core.graph.read.aggregate_facts` 统一完成。
- ``concepts_by_ids`` / ``evidence_by_ids`` 只返回快照里真实存在的记录；
  查不到的标识不出现在结果里，不为缺失项编造记录。去重与按标识排序与在线
  端口（``WHERE … IN … ORDER BY id``）同口径。
- ``evidence_by_page`` 只返回快照闭包内该页的证据段，按页面字符区间升序；
  闭包外的同页邻居不在文件里，因此取不到——这是快照的边界，不是丢数据。

**快照不是图数据库备份。** 闭包之外的节点、边与证据段查不到就是查不到；
需要闭包外数据时应当重新导出快照，而不是让读侧猜。读端口也不知道任何领域
语义（白名单、跳数由领域包决定），``GraphReader`` 的调用方无需修改。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from idr_core.graph.read import Concept, EvidenceRecord, FactEdge
from idr_core.graph.snapshot import GraphSnapshot, load_snapshot


class SnapshotGraphReader:
    """以一份冻结快照为全部数据源的 ``GraphReader`` 实现。

    构造时建立按起点、按标识与按页的索引；之后只读，不写文件、不联网。
    同一个读端口两次调用返回同一批记录，评估因此可以在同一张图上重跑。
    """

    def __init__(self, snapshot: GraphSnapshot) -> None:
        self._snapshot = snapshot
        self._concepts = {concept.id: concept for concept in snapshot.concepts}
        self._evidence = {record.id: record for record in snapshot.evidence}

        edges: dict[str, list[FactEdge]] = {}
        for edge in snapshot.edges:
            edges.setdefault(edge.head_id, []).append(edge)
        self._edges_from = {
            head_id: tuple(records) for head_id, records in edges.items()
        }

        pages: dict[tuple[str, int], list[EvidenceRecord]] = {}
        for record in snapshot.evidence:
            pages.setdefault((record.doc_id, record.page), []).append(record)
        self._pages = {
            key: tuple(
                sorted(records, key=lambda item: (item.charspan.start, item.id))
            )
            for key, records in pages.items()
        }

    @classmethod
    def load(cls, path: Path) -> SnapshotGraphReader:
        """读取并校验快照文件后构造读端口；校验失败抛 ``SnapshotError``。"""
        return cls(load_snapshot(path))

    @property
    def snapshot(self) -> GraphSnapshot:
        """被包装的快照；调用方据此读取校验和与闭包范围。"""
        return self._snapshot

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        """从该概念出发、类型在清单内的一跳边；按快照内的确定顺序返回。"""
        wanted = set(relation_types)
        if not wanted:
            return ()
        return tuple(
            edge
            for edge in self._edges_from.get(concept_id, ())
            if edge.relation in wanted
        )

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        """按标识取概念；缺失的标识被跳过，结果按标识排序、每个标识一条。"""
        return tuple(
            self._concepts[concept_id]
            for concept_id in sorted(set(concept_ids) & self._concepts.keys())
        )

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        """按标识取证据段；缺失的标识被跳过，结果按标识排序、每个标识一条。"""
        return tuple(
            self._evidence[evidence_id]
            for evidence_id in sorted(set(evidence_ids) & self._evidence.keys())
        )

    def evidence_by_page(self, doc_id: str, page: int) -> tuple[EvidenceRecord, ...]:
        """该文献该页在快照闭包内的证据段，按页面字符区间升序。

        闭包外的同页邻居不在快照里，这里不会返回它们；引用层把返回结果当作
        完整同页集合时，只会少并上下文，不会拼出页面里不存在的句子。
        """
        return self._pages.get((doc_id, page), ())

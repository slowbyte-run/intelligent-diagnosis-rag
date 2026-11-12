"""图上遍历：从检索起点沿白名单关系走至多两跳，并把每一步留下来。

白名单固定六类边，三跳及以上不执行。这是有意的检索边界：两跳以外结果集
爆炸，「为什么是这个结论」的解释链也随之崩溃——设备→部件→故障模式已经是
两跳，再往下取征兆就是三跳。

两跳只允许沿结构性关系（``has_component`` / ``has_fault_mode``）中转。其余
关系本身就是结论（征兆、检测手段、原因、易混淆模式），拿它们当中转会把
「A 的易混淆模式 B 的征兆」算成 A 的征兆，答案看似更全，其实在误导。

遍历返回路径而不是终点集合：每一步带关系、两端概念、证据段与置信度，答案层
据此既能组装结论，也能如实回答「怎么走到的」。没有证据的边不参与遍历——
结论缺失一条依据时整条路径不存在，而不是悄悄缩短。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.graph import (
    AggregatedFact,
    Concept,
    EvidenceAnchor,
    GraphReader,
    aggregate_facts,
)

# 边白名单（#15）：只有这六类关系允许出现在检索路径上。
EDGE_WHITELIST = (
    "has_component",
    "has_fault_mode",
    "has_cause",
    "indicated_by",
    "detected_by",
    "confused_with",
)

# 描述结构的关系才能作为两跳的中转边；它是白名单的子集，不是另一张表。
STRUCTURAL_RELATIONS = ("has_component", "has_fault_mode")

# 三跳及以上不执行：设备→部件→故障模式是两跳，再取征兆就是三跳。
MAX_HOPS = 2

# 结果集上界：两跳的扇出可以很大，超出时按固定顺序截断并在答案里说明。
MAX_PATHS = 20

# 结论顺序由关系类型决定（图路径不重排）：先征兆，再检测手段、原因、
# 故障模式、部件构成，最后是易混淆模式。
RELATION_ORDER = (
    "indicated_by",
    "detected_by",
    "has_cause",
    "has_fault_mode",
    "has_component",
    "confused_with",
)


@dataclass(frozen=True, slots=True)
class PathStep:
    """路径上的一步：从一个概念沿一条有证据的边走到下一个概念。

    ``anchors`` 是这一步各证据段内的逐字出处，与 ``evidence_ids`` 平行但可以
    更少；某段没有锚点时该段整段引用。一步就是一条断言，所以锚点挂在步上，
    不挂在整条路径上——两跳结论的不同跳可以引用同一段证据的不同句子。
    """

    head: Concept
    relation: str
    tail: Concept
    evidence_ids: tuple[str, ...]
    confidence: float
    anchors: tuple[EvidenceAnchor, ...] = ()

    def anchor_for(self, evidence_id: str) -> EvidenceAnchor | None:
        """该证据段在这一步里的锚点；没有锚点时返回 ``None``（整段引用）。"""
        for anchor in self.anchors:
            if anchor.evidence_id == evidence_id:
                return anchor
        return None


@dataclass(frozen=True, slots=True)
class TraversalPath:
    """从检索起点出发的一条路径；长度即跳数（1 或 2）。"""

    steps: tuple[PathStep, ...]

    @property
    def start(self) -> Concept:
        return self.steps[0].head

    @property
    def end(self) -> Concept:
        return self.steps[-1].tail

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def route(self) -> tuple[Concept, ...]:
        """路径上的概念链，从起点到终点。"""
        return (self.start, *(step.tail for step in self.steps))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """全部步骤的证据段，按首次出现顺序去重。"""
        return tuple(
            dict.fromkeys(
                evidence_id
                for step in self.steps
                for evidence_id in step.evidence_ids
            )
        )


@dataclass(frozen=True, slots=True)
class Traversal:
    """一次遍历的结果：截断后的路径、截断前的命中数与上界。"""

    paths: tuple[TraversalPath, ...]
    found: int
    limit: int

    @property
    def truncated(self) -> bool:
        return self.found > len(self.paths)


def traverse(
    graph: GraphReader,
    focus: Concept,
    *,
    limit: int = MAX_PATHS,
) -> Traversal:
    """从 ``focus`` 出发做至多 ``MAX_HOPS`` 跳遍历。

    每一跳只走白名单里有证据的边；无证据的边由 ``aggregate_facts`` 在聚合时
    丢弃，因此需要它的路径根本不会出现。结果按关系类型排序后截断到 ``limit``。
    """
    if limit < 1:
        raise ValueError("结果上界必须是正整数")

    candidates: list[TraversalPath] = []
    frontier: tuple[tuple[Concept, tuple[PathStep, ...]], ...] = ((focus, ()),)

    for hop in range(1, MAX_HOPS + 1):
        edges: dict[str, tuple[AggregatedFact, ...]] = {
            node.id: aggregate_facts(graph.facts_from(node.id, EDGE_WHITELIST))
            for node, _ in frontier
        }
        tail_ids = list(
            dict.fromkeys(
                fact.tail_id for facts in edges.values() for fact in facts
            )
        )
        concepts = {
            concept.id: concept
            for concept in graph.concepts_by_ids(tail_ids)
        }
        reached: list[tuple[Concept, tuple[PathStep, ...]]] = []
        for node, path in frontier:
            visited = {focus.id, *(step.tail.id for step in path)}
            for fact in edges[node.id]:
                tail = concepts.get(fact.tail_id)
                if (
                    fact.relation not in EDGE_WHITELIST
                    or not fact.evidence_ids
                    or tail is None
                    or tail.id in visited
                ):
                    continue
                steps = (
                    *path,
                    PathStep(
                        head=node,
                        relation=fact.relation,
                        tail=tail,
                        evidence_ids=fact.evidence_ids,
                        confidence=fact.confidence,
                        anchors=fact.anchors,
                    ),
                )
                candidates.append(TraversalPath(steps=steps))
                if hop < MAX_HOPS and fact.relation in STRUCTURAL_RELATIONS:
                    reached.append((tail, steps))
        frontier = tuple(reached)
        if not frontier:
            break

    ordered = tuple(sorted(candidates, key=_sort_key))
    return Traversal(paths=ordered[:limit], found=len(ordered), limit=limit)


def _sort_key(
    path: TraversalPath,
) -> tuple[int, tuple[int, ...], tuple[str, ...], str]:
    """先按末跳关系排序，再按整条关系链、概念名与终点标识定序。

    末跳关系决定结论的类型（征兆优先于检测手段），整条关系链让同一类型下
    直接结论排在多跳结论之前，概念名则让同一故障模式的观察聚在一起。
    """
    return (
        _relation_rank(path.steps[-1].relation),
        tuple(_relation_rank(step.relation) for step in path.steps),
        tuple(step.tail.name for step in path.steps),
        path.end.id,
    )


def _relation_rank(relation: str) -> int:
    try:
        return RELATION_ORDER.index(relation)
    except ValueError:
        return len(RELATION_ORDER)

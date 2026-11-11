"""领域包第二个公开门面：一个问题从问句走到带引用的结构化答案。

调用方只依赖 ``ask(question, *, deps) -> Answer``；图存储与文献名目录由
``deps`` 注入，测试用内存假实现。这一版只走图：本体别名表定位问句里的
概念，沿关系白名单取一跳事实，每条事实的证据段回填文献名、页码与原文。

线索留在这里，不要在本版展开：

- 两跳（设备→部件→故障模式，#15）：在 ``ONE_HOP_RELATIONS`` 之外增加关系，
  并按跳数扩展事实收集；
- 向量兜底与重排（#16）：图没有事实时改走向量检索，答案路径标
  ``AnswerPath.VECTOR``；
- 拒答策略与阈值（#17）：``_refusal`` 里的 ``detail`` 目前只是事实描述，
  阈值判定接在同一个出口。

答案必须带引用（AGENTS.md 第 6 条）：没有证据段的事实在组装阶段就被排除，
不进入 ``facts``。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.documents import DocumentNames
from idr_core.graph import (
    AggregatedFact,
    Concept,
    EvidenceRecord,
    GraphReader,
    aggregate_facts,
)
from idr_rotating.answer import Answer, AnswerPath, Citation, Fact
from idr_rotating.linking import ConceptMatch, link_question
from idr_rotating.ontology import Ontology, load_ontology

# 本票只取这三类一跳事实。has_component / has_fault_mode 需要两跳（#15）；
# confused_with 属于鉴别型问答，尚未开票，暂不纳入。
ONE_HOP_RELATIONS = ("indicated_by", "detected_by", "has_cause")


@dataclass(frozen=True, slots=True)
class AskDeps:
    """问答流程的全部外部依赖。

    只注入当前路径真正用到的东西：图检索端口与（可选的）文献名目录。向量
    兜底需要的嵌入与向量存储、以及自然语言生成需要的模型，等对应票据落地时
    再加，避免为未实现的路径预备空实现。
    """

    graph_store: GraphReader
    document_names: DocumentNames | None = None


def ask(question: str, *, deps: AskDeps) -> Answer:
    """回答一个问题；没有可用依据时返回显式的空答案而不是猜测。"""
    ontology = load_ontology()
    matches = link_question(question, ontology)
    if not matches:
        return _refusal(
            question,
            matches=matches,
            focus=None,
            detail="本体别名表没有在问题中匹配到概念，无法定位检索起点。",
        )

    candidates = _focus_candidates(matches, ontology)
    if not candidates:
        return _refusal(
            question,
            matches=matches,
            focus=None,
            detail="命中的概念不在本次一跳检索的关系范围内，无法取事实。",
        )

    focus, tied = _most_specific(candidates)
    if focus is None:
        names = "、".join(f"「{match.concept.name}」" for match in tied)
        return _refusal(
            question,
            matches=matches,
            focus=None,
            detail=f"问题同时命中了多个同等具体的概念（{names}），"
            "不确定以哪个作为检索起点。",
        )

    return _answer_from_graph(question, deps, matches, focus)


def _answer_from_graph(
    question: str,
    deps: AskDeps,
    matches: tuple[ConceptMatch, ...],
    focus: ConceptMatch,
) -> Answer:
    """图命中路径：一跳事实 + 证据回填；任何一步没有依据都退成显式空答案。"""
    merged = aggregate_facts(
        deps.graph_store.facts_from(focus.concept.id, ONE_HOP_RELATIONS)
    )
    concept_ids = list(
        dict.fromkeys([focus.concept.id, *(fact.tail_id for fact in merged)])
    )
    concepts = {
        node.id: node for node in deps.graph_store.concepts_by_ids(concept_ids)
    }
    if focus.concept.id not in concepts:
        return _refusal(
            question,
            matches=matches,
            focus=focus,
            detail=f"概念「{focus.concept.name}」不在图谱中。",
        )

    evidence_ids = list(
        dict.fromkeys(
            evidence_id for fact in merged for evidence_id in fact.evidence_ids
        )
    )
    segments = {
        segment.id: segment for segment in deps.graph_store.evidence_by_ids(evidence_ids)
    }
    facts = _facts_with_evidence(merged, concepts, segments)
    if not facts:
        # #16 在图没有事实时接向量兜底；#17 在同一出口判定拒答。
        return _refusal(
            question,
            matches=matches,
            focus=focus,
            detail=f"概念「{focus.concept.name}」没有带证据的一跳事实。",
        )

    return Answer(
        question=question,
        path=AnswerPath.GRAPH,
        matches=matches,
        focus=focus,
        facts=facts,
        citations=_citations_for(facts, segments, deps.document_names),
    )


def _facts_with_evidence(
    merged: tuple[AggregatedFact, ...],
    concepts: dict[str, Concept],
    segments: dict[str, EvidenceRecord],
) -> tuple[Fact, ...]:
    """只保留两端概念与至少一段证据都能落地的结论，并按关系白名单排序。"""
    relation_order = {
        relation: index for index, relation in enumerate(ONE_HOP_RELATIONS)
    }
    facts: list[Fact] = []
    for fact in merged:
        head = concepts.get(fact.head_id)
        tail = concepts.get(fact.tail_id)
        usable = tuple(
            evidence_id
            for evidence_id in fact.evidence_ids
            if evidence_id in segments
        )
        if head is None or tail is None or not usable:
            continue
        facts.append(
            Fact(
                relation=fact.relation,
                head=head,
                tail=tail,
                confidence=fact.confidence,
                evidence_ids=usable,
            )
        )
    # 关系白名单的顺序就是结论顺序：征兆优先于检测手段，检测手段优先于原因。
    return tuple(
        sorted(
            facts,
            key=lambda item: (
                relation_order.get(item.relation, len(relation_order)),
                item.tail.name,
                item.tail.id,
            ),
        )
    )


def _citations_for(
    facts: tuple[Fact, ...],
    segments: dict[str, EvidenceRecord],
    document_names: DocumentNames | None,
) -> tuple[Citation, ...]:
    """按结论顺序展开引用，同一证据段只列一次。"""
    citations: dict[str, Citation] = {}
    for fact in facts:
        for evidence_id in fact.evidence_ids:
            if evidence_id in citations:
                continue
            segment = segments[evidence_id]
            citations[evidence_id] = Citation(
                evidence_id=segment.id,
                doc_id=segment.doc_id,
                document=(
                    document_names.name_for(segment.doc_id)
                    if document_names is not None
                    else None
                ),
                page=segment.page,
                text=segment.text,
            )
    return tuple(citations.values())


def _focus_candidates(
    matches: tuple[ConceptMatch, ...], ontology: Ontology
) -> tuple[ConceptMatch, ...]:
    """关系定义域内的命中才可能成为检索起点。"""
    head_types = frozenset(
        entity_type
        for relation in ONE_HOP_RELATIONS
        for entity_type in ontology.relation_types[relation].domain
    )
    return tuple(match for match in matches if match.concept.type in head_types)


def _most_specific(
    candidates: tuple[ConceptMatch, ...],
) -> tuple[ConceptMatch | None, tuple[ConceptMatch, ...]]:
    """取最具体的候选；同样具体的不猜，返回全部并列项供拒答说明。"""
    best = max(match.specificity for match in candidates)
    tied = tuple(
        match for match in candidates if match.specificity == best
    )
    if len(tied) > 1:
        return None, tied
    return tied[0], tied


def _refusal(
    question: str,
    *,
    matches: tuple[ConceptMatch, ...],
    focus: ConceptMatch | None,
    detail: str,
) -> Answer:
    """显式的空答案：把「缺在哪一步」写在 detail 里，不返回像模像样的猜测。"""
    return Answer(
        question=question,
        path=AnswerPath.REFUSAL,
        matches=matches,
        focus=focus,
        facts=(),
        citations=(),
        detail=detail,
    )

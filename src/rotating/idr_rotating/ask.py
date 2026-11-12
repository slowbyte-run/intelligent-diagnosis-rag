"""领域包第二个公开门面：一个问题从问句走到带引用的结构化答案。

调用方只依赖 ``ask(question, *, deps) -> Answer``；图存储、文献名目录与向量
兜底的三件依赖（嵌入、向量检索、重排）都由 ``deps`` 注入，测试用内存假实现。

检索有先后，不并行：

1. **图路径先走**：本体别名表定位问句里的概念，从最具体的命中出发沿白名单
   关系走至多两跳（见 ``traversal``），每条结论的证据段回填文献名、页码与
   原文。结论本身带完整路径，答案因此能说明「怎么走到的」。图拿出结论时
   向量检索一次都不发生——结构化结论不该被一堆语义相近的段落盖住。
2. **图没有依据时落向量**：``vector_recall`` 按检索配置改写查询（概念补全与
   检索指令只影响嵌入，见 ``query_rewrite``），顺序扫描 pgvector 取候选、过
   重排、过阈值，答案标 ``AnswerPath.VECTOR``，``facts`` 为空，证据段直接进
   ``citations`` 并带上重排分数。
3. **两边都没有依据才拒答**：``detail`` 分别写清图路径缺在哪、向量兜底差多少；
   ``refusal_reason`` 区分 ``no_concept``（概念没识别出来）与 ``no_evidence``
   （识别了但没有依据），用户据此知道该换问法还是该补语料。

问句里出现多个同级概念时按**对比**回答，不做意图分类：两个类型相同、具体
程度相差不超过 ``MAX_ANCHOR_SPECIFICITY_GAP`` 的独立提及就是两个锚点。对比
答案的两侧各自遍历，事实按 ``(关系, 尾概念)`` 分成共同项、仅左侧、仅右侧；
两侧之间的 ``confused_with`` 单独成一组，它是文献自己说「这两个容易混」，不
混进「仅左侧」。三个及以上锚点暂不支持，显式拒答而不是悄悄只比两个。

歧义问题不兜底：同一个别名指向多个规范实例（如「磨粒磨损」既可能是轴承的也
可能是齿面的）仍是一处歧义，不是两个锚点；向量召回分不清该答哪一个，兜底
只会把歧义藏进一个像模像样的答案里。

答案必须带引用（AGENTS.md 第 6 条）：没有证据段的事实、或路径上有一步
拿不到证据的结论，在组装阶段就被排除，不进入答案。对比答案的结论在
``comparison`` 分区里，``facts`` 为空；两个分区里的每条结论都带证据。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.documents import DocumentNames
from idr_core.embedding import Embedder
from idr_core.graph import (
    EvidenceAnchor,
    EvidenceRecord,
    EvidenceTextSearch,
    GraphReader,
)
from idr_core.llm import ModelCall
from idr_core.llm.rerank import Reranker
from idr_core.vector_store import VectorReader
from idr_rotating.answer import (
    Answer,
    AnswerPath,
    Citation,
    Comparison,
    Fact,
    RefusalReason,
    SharedFact,
)
from idr_rotating.citation_context import display_text
from idr_rotating.citation_selection import select_citation_window
from idr_rotating.linking import ConceptMatch, link_question
from idr_rotating.ontology import Ontology, load_ontology
from idr_rotating.retrieval import RetrievalSettings, load_retrieval_settings
from idr_rotating.traversal import (
    EDGE_WHITELIST,
    Traversal,
    TraversalPath,
    traverse,
)
from idr_rotating.vector_recall import EvidenceRecall, recall_evidence

# 两个锚点的具体程度相差不超过这么多字，就视为「同级」：问句对一侧用简写、
# 对另一侧用全称是常见写法（「内圈剥落」与「滚动体疲劳剥落」），字面长度
# 本身说明不了哪个概念更具体。差得更多时仍按单焦点处理，避免把泛称与
# 专称强行配对。
MAX_ANCHOR_SPECIFICITY_GAP = 3


@dataclass(frozen=True, slots=True)
class AskDeps:
    """问答流程的全部外部依赖。

    图检索端口是必需的；向量兜底的三件依赖（嵌入、向量检索、重排）可缺省：
    三者都给齐才会在图路径落空时兜底，缺任何一个都保持「图路径失败即拒答」的
    旧行为，命令行因此可以先只接图。

    召回条数与重排阈值来自 ``retrieval_settings``；不给时读
    ``config/retrieval.yaml``，线上与校准读同一份配置。
    """

    graph_store: GraphReader
    document_names: DocumentNames | None = None
    embedder: Embedder | None = None
    vector_store: VectorReader | None = None
    reranker: Reranker | None = None
    # 词面召回端口（通常由同一个图存储实现）；缺省时向量兜底只走语义通道。
    text_search: EvidenceTextSearch | None = None
    retrieval_settings: RetrievalSettings | None = None


def ask(question: str, *, deps: AskDeps) -> Answer:
    """回答一个问题；图没有依据时降级到向量兜底，两边都没有就显式拒答。"""
    ontology = load_ontology()
    matches = link_question(question, ontology)
    if not matches:
        return _fall_back_or_refuse(
            question,
            deps,
            _refusal(
                question,
                matches=matches,
                focus=None,
                detail="本体别名表没有在问题中匹配到概念，无法定位检索起点。",
                reason=RefusalReason.NO_CONCEPT,
            ),
            ontology,
        )

    candidates = _focus_candidates(matches, ontology)
    if not candidates:
        return _fall_back_or_refuse(
            question,
            deps,
            _refusal(
                question,
                matches=matches,
                focus=None,
                detail="命中的概念不在本次图检索的关系范围内，无法取事实。",
                reason=RefusalReason.NO_EVIDENCE,
            ),
            ontology,
        )

    anchors = _comparison_anchors(candidates)
    if anchors is not None:
        if len(anchors) > 2:
            names = "、".join(f"「{match.concept.name}」" for match in anchors)
            return _refusal(
                question,
                matches=matches,
                focus=None,
                detail=f"问题同时提到了 {len(anchors)} 个同级概念（{names}），"
                "当前只支持两两对比，请一次比较两个。",
                reason=RefusalReason.NO_EVIDENCE,
            )
        graph_answer = _answer_from_comparison(
            question, deps, matches, anchors[0], anchors[1], ontology
        )
    else:
        focus, tied = _most_specific(candidates)
        if focus is None:
            names = "、".join(f"「{match.concept.name}」" for match in tied)
            # 歧义不兜底：向量召回分不清两个同样具体的概念，只会把歧义藏起来。
            return _refusal(
                question,
                matches=matches,
                focus=None,
                detail=f"问题同时命中了多个同等具体的概念（{names}），"
                "不确定以哪个作为检索起点。",
                reason=RefusalReason.NO_EVIDENCE,
            )
        graph_answer = _answer_from_graph(question, deps, matches, focus, ontology)

    if graph_answer.path is AnswerPath.GRAPH:
        return graph_answer
    return _fall_back_or_refuse(question, deps, graph_answer, ontology)


def _fall_back_or_refuse(
    question: str, deps: AskDeps, graph_answer: Answer, ontology: Ontology
) -> Answer:
    """图路径没拿出依据时改走向量；没有注入向量依赖就原样返回图的空答案。"""
    if (
        deps.embedder is None
        or deps.vector_store is None
        or deps.reranker is None
    ):
        return graph_answer

    settings = deps.retrieval_settings or load_retrieval_settings()
    recall = recall_evidence(
        question,
        embedder=deps.embedder,
        vector_store=deps.vector_store,
        reranker=deps.reranker,
        graph_store=deps.graph_store,
        settings=settings,
        ontology=ontology,
        text_search=deps.text_search,
    )
    if not recall.segments:
        return _refusal(
            question,
            matches=graph_answer.matches,
            focus=graph_answer.focus,
            detail=f"{graph_answer.detail or ''}{_vector_miss_detail(recall, settings)}",
            model_calls=recall.model_calls,
            reason=(
                RefusalReason.NO_CONCEPT
                if not graph_answer.matches
                else RefusalReason.NO_EVIDENCE
            ),
        )

    return Answer(
        question=question,
        path=AnswerPath.VECTOR,
        matches=graph_answer.matches,
        focus=graph_answer.focus,
        # 向量路径没有图上的结论；证据段本身就是要给用户看的东西。
        facts=(),
        citations=tuple(
            Citation(
                evidence_id=item.segment.id,
                doc_id=item.segment.doc_id,
                document=(
                    deps.document_names.name_for(item.segment.doc_id)
                    if deps.document_names is not None
                    else None
                ),
                page=item.segment.page,
                text=item.segment.text,
                score=item.score,
            )
            for item in recall.segments
        ),
        model_calls=recall.model_calls,
    )


def _vector_miss_detail(
    recall: EvidenceRecall, settings: RetrievalSettings
) -> str:
    """向量兜底为什么没给出依据：写成可核对的数字，而不是「没找到」。"""
    if recall.recalled == 0:
        return "向量兜底也没有召回任何证据段。"
    if recall.top_score is None:
        return "向量兜底召回的候选在图上找不到原文，不能作为依据。"
    provenance = (
        "已用金标准集校准"
        if settings.vector_min_rerank_score_calibrated
        else "未经校准的初值"
    )
    return (
        f"向量兜底最高重排分 {recall.top_score:.2f}，低于阈值 "
        f"{settings.vector_min_rerank_score:.2f}（{provenance}），不作为依据。"
    )


def _answer_from_graph(
    question: str,
    deps: AskDeps,
    matches: tuple[ConceptMatch, ...],
    focus: ConceptMatch,
    ontology: Ontology,
) -> Answer:
    """图命中路径：至多两跳的带证据事实 + 证据回填。

    路径上任一步缺证据就丢弃该条结论；全部路径都不可用时退成显式空答案。
    结果集撞到上界时照常作答，但把截断写进 ``detail`` 并置 ``truncated``。
    """
    anchored = _facts_from_anchor(deps.graph_store, focus)
    if anchored is None:
        return _refusal(
            question,
            matches=matches,
            focus=focus,
            detail=f"概念「{focus.concept.name}」不在图谱中。",
            reason=RefusalReason.NO_EVIDENCE,
        )

    if not anchored.facts:
        # 图没有带证据的事实；向量兜底与最终拒答由 ask() 在同一出口判定。
        return _refusal(
            question,
            matches=matches,
            focus=focus,
            detail=f"概念「{focus.concept.name}」没有带证据的一跳或两跳事实。",
            reason=RefusalReason.NO_EVIDENCE,
        )

    return Answer(
        question=question,
        path=AnswerPath.GRAPH,
        matches=matches,
        focus=focus,
        facts=anchored.facts,
        citations=_citations_for(
            anchored.facts,
            anchored.segments,
            deps.document_names,
            ontology,
            deps.graph_store,
        ),
        detail=_truncated_detail([(focus.concept.name, anchored.traversal)]),
        truncated=anchored.traversal.truncated,
    )


def _answer_from_comparison(
    question: str,
    deps: AskDeps,
    matches: tuple[ConceptMatch, ...],
    left: ConceptMatch,
    right: ConceptMatch,
    ontology: Ontology,
) -> Answer:
    """对比路径：两侧各自取事实，按 ``(关系, 尾概念)`` 分区后作答。

    两侧的事实都必须是带证据的；一侧取不到事实时照常给出另一侧的分区，并在
    ``detail`` 里写明缺的是哪一侧——只答一侧是事实，不说是隐瞒。两侧都没有
    事实才退成显式空答案，交给向量兜底与最终拒答判定。
    """
    left_facts = _facts_from_anchor(deps.graph_store, left)
    right_facts = _facts_from_anchor(deps.graph_store, right)
    if left_facts is None or right_facts is None:
        missing = [
            anchor
            for anchor, anchored in ((left, left_facts), (right, right_facts))
            if anchored is None
        ]
        names = "、".join(f"「{match.concept.name}」" for match in missing)
        return _refusal(
            question,
            matches=matches,
            focus=None,
            detail=f"概念{names}不在图谱中，无法对比。",
            reason=RefusalReason.NO_EVIDENCE,
        )

    comparison = _comparison(left, right, left_facts.facts, right_facts.facts)
    facts = _comparison_facts(comparison)
    if not facts:
        return _refusal(
            question,
            matches=matches,
            focus=None,
            detail=f"「{left.concept.name}」与「{right.concept.name}」"
            "都没有带证据的一跳或两跳事实，无法对比。",
            reason=RefusalReason.NO_EVIDENCE,
        )

    details = [
        _truncated_detail(
            [
                (left.concept.name, left_facts.traversal),
                (right.concept.name, right_facts.traversal),
            ]
        )
    ]
    if not left_facts.facts:
        details.append(f"「{left.concept.name}」没有带证据的一跳或两跳事实，只列出右侧。")
    if not right_facts.facts:
        details.append(f"「{right.concept.name}」没有带证据的一跳或两跳事实，只列出左侧。")
    segments = {**left_facts.segments, **right_facts.segments}
    return Answer(
        question=question,
        path=AnswerPath.GRAPH,
        matches=matches,
        focus=None,
        facts=(),
        citations=_citations_for(
            facts, segments, deps.document_names, ontology, deps.graph_store
        ),
        detail="".join(detail for detail in details if detail) or None,
        truncated=left_facts.traversal.truncated or right_facts.traversal.truncated,
        comparison=comparison,
    )


@dataclass(frozen=True, slots=True)
class _AnchorFacts:
    """一个锚点在图上取到的事实、证据段与遍历结果。"""

    facts: tuple[Fact, ...]
    segments: dict[str, EvidenceRecord]
    traversal: Traversal


def _facts_from_anchor(graph: GraphReader, anchor: ConceptMatch) -> _AnchorFacts | None:
    """从一个锚点取至多两跳的带证据事实；锚点不在图谱时返回 ``None``。"""
    focus_node = next(iter(graph.concepts_by_ids([anchor.concept.id])), None)
    if focus_node is None:
        return None
    traversal = traverse(graph, focus_node)
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for path in traversal.paths
            for step in path.steps
            for evidence_id in step.evidence_ids
        )
    )
    segments = {
        segment.id: segment for segment in graph.evidence_by_ids(evidence_ids)
    }
    return _AnchorFacts(
        facts=_facts_with_evidence(traversal.paths, segments),
        segments=segments,
        traversal=traversal,
    )


def _truncated_detail(
    sides: list[tuple[str, Traversal]],
) -> str | None:
    """哪些一侧撞到了结果上界；截断必须让调用方看得见。"""
    truncated = [(name, traversal) for name, traversal in sides if traversal.truncated]
    if not truncated:
        return None
    if len(sides) == 1:
        _, traversal = truncated[0]
        return (
            f"结果超过 {traversal.limit} 条上限（命中 {traversal.found} 条），"
            "已按关系类型与概念名称排序截断。"
        )
    return "；".join(
        f"「{name}」一侧结果超过 {traversal.limit} 条上限"
        f"（命中 {traversal.found} 条），已截断"
        for name, traversal in truncated
    ) + "。"


def _comparison(
    left: ConceptMatch,
    right: ConceptMatch,
    left_facts: tuple[Fact, ...],
    right_facts: tuple[Fact, ...],
) -> Comparison:
    """按 ``(关系, 尾概念)`` 把两侧事实分成共同项、各自独有项与易混淆声明。"""
    left_keys = _facts_by_key(left_facts)
    right_keys = _facts_by_key(right_facts)
    shared = tuple(
        SharedFact(
            relation=key[0],
            tail=facts[0].tail,
            left=facts,
            right=right_keys[key],
        )
        for key, facts in left_keys.items()
        if key in right_keys
    )
    only_left = tuple(
        fact
        for fact in left_facts
        if _fact_key(fact) not in right_keys
        and not _confused_with(fact, right.concept.id)
    )
    only_right = tuple(
        fact
        for fact in right_facts
        if _fact_key(fact) not in left_keys
        and not _confused_with(fact, left.concept.id)
    )
    discriminating = tuple(
        fact
        for fact, other_id in (
            *((fact, right.concept.id) for fact in left_facts),
            *((fact, left.concept.id) for fact in right_facts),
        )
        if _confused_with(fact, other_id)
    )
    return Comparison(
        left=left,
        right=right,
        shared=shared,
        only_left=only_left,
        only_right=only_right,
        discriminating=discriminating,
    )


def _facts_by_key(
    facts: tuple[Fact, ...],
) -> dict[tuple[str, str], tuple[Fact, ...]]:
    """按 ``(关系, 尾概念)`` 归拢一侧的事实，保持遍历顺序。

    同一对 ``(关系, 尾概念)`` 可能经一跳与两跳各出现一次（例如「轴承 → 故障
    模式 → 征兆」与「轴承 → 征兆」）：它们是不同的路径，不能压成一条。
    """
    grouped: dict[tuple[str, str], list[Fact]] = {}
    for fact in facts:
        grouped.setdefault(_fact_key(fact), []).append(fact)
    return {key: tuple(items) for key, items in grouped.items()}


def _fact_key(fact: Fact) -> tuple[str, str]:
    return (fact.relation, fact.tail.id)


def _confused_with(fact: Fact, other_id: str) -> bool:
    """这条结论是不是文献在说「两个概念容易混淆」。"""
    return fact.relation == "confused_with" and fact.tail.id == other_id


def _comparison_facts(comparison: Comparison) -> tuple[Fact, ...]:
    """对比答案里要进引用的全部结论：共同项（两侧）、各自独有项、易混淆声明。"""
    return (
        *(fact for shared in comparison.shared for fact in shared.left),
        *(fact for shared in comparison.shared for fact in shared.right),
        *comparison.only_left,
        *comparison.only_right,
        *comparison.discriminating,
    )


def _facts_with_evidence(
    paths: tuple[TraversalPath, ...],
    segments: dict[str, EvidenceRecord],
) -> tuple[Fact, ...]:
    """把路径组装成结论；任何一步缺证据就丢弃整条结论。

    路径保持遍历给出的顺序（关系类型决定），这里不重排。多跳结论的置信度取
    各步最小值：链条不会比最弱的一环更可信。
    """
    facts: list[Fact] = []
    for path in paths:
        if any(
            not step.evidence_ids
            or any(evidence_id not in segments for evidence_id in step.evidence_ids)
            for step in path.steps
        ):
            # 缩短成部分路径是静默降级：读者会以为系统查过那一步。
            continue
        facts.append(
            Fact(
                relation=path.steps[-1].relation,
                head=path.start,
                tail=path.end,
                confidence=min(step.confidence for step in path.steps),
                evidence_ids=path.evidence_ids,
                path=path.steps,
            )
        )
    return tuple(facts)


def _citations_for(
    facts: tuple[Fact, ...],
    segments: dict[str, EvidenceRecord],
    document_names: DocumentNames | None,
    ontology: Ontology,
    graph: GraphReader,
) -> tuple[Citation, ...]:
    """按结论顺序展开引用；同一 (证据段, 锚点) 只列一次。

    锚点挂在路径的每一步上（一步就是一条断言）：同一证据段被两条断言引用时，
    两条断言各自的片段都要出现，因此去重键是 (证据段, 锚点) 而不是证据段。
    某段没有锚点、或锚点区间在图上对不上原文时退回整段，绝不因缺锚点少给。

    没有锚点的旧数据先按断言选句，再进出去重：同一段证据被两条断言引用时，
    两条断言词法重合的句子不同，因此各自成一条引用，不会被并成一条整段引用；
    选不出句子时锚点仍是 ``None``，与以前一样整段引用。

    入库时固化的短锚点，展示文本会并入同页相邻证据段（见 ``citation_context``）：
    锚点不变，只有 ``Citation.text`` 变长。同页证据段按页缓存，一次提问里同一页
    最多取一次。查询期选句得到的窗口已经落在句子边界上，不再二次并入上下文——
    那会把刚收窄的引用又摊回整段。
    """
    pages: dict[tuple[str, int], tuple[EvidenceRecord, ...]] = {}
    citations: dict[tuple[str, EvidenceAnchor | None], Citation] = {}
    for fact in facts:
        spans = fact.cited_spans()
        assertions = _assertion_terms(fact, len(spans))
        for (evidence_id, anchor), (head, relation, tail) in zip(
            spans, assertions, strict=True
        ):
            segment = segments[evidence_id]
            if anchor is not None:
                if _anchor_fits(segment.text, anchor):
                    usable = anchor
                    # 入库固化的锚点：可能是公式区的碎片，短则并入同页上下文。
                    text = _displayed_text(segment, usable, graph, pages)
                else:
                    # 越界锚点是坏数据：不据此推断句子，退回整段，键也归一化为
                    # ``None``，否则消费方按原锚点找不到这条引用。
                    usable = None
                    text = segment.text
            else:
                usable = _selected_anchor(
                    segment, head=head, relation=relation, tail=tail, ontology=ontology
                )
                # 选句窗口已在句子边界上，不再并入上下文（理由见函数说明）。
                text = (
                    segment.text
                    if usable is None
                    else segment.text[usable.start : usable.end]
                )
            key = (evidence_id, usable)
            if key in citations:
                continue
            citations[key] = Citation(
                evidence_id=segment.id,
                doc_id=segment.doc_id,
                document=(
                    document_names.name_for(segment.doc_id)
                    if document_names is not None
                    else None
                ),
                page=segment.page,
                text=text,
                anchor=usable,
            )
    return tuple(citations.values())


def _displayed_text(
    segment: EvidenceRecord,
    anchor: EvidenceAnchor,
    graph: GraphReader,
    pages: dict[tuple[str, int], tuple[EvidenceRecord, ...]],
) -> str:
    """入库锚点的展示原文：长锚点取片段，短锚点再并入同页上下文。"""

    def page_segments() -> tuple[EvidenceRecord, ...]:
        """同页证据段；只有短锚点会真的调用它，长锚点不查图。"""
        key = (segment.doc_id, segment.page)
        known = pages.get(key)
        if known is None:
            known = graph.evidence_by_page(*key)
            pages[key] = known
        return known

    return display_text(segment, anchor, page_segments)


def _assertion_terms(
    fact: Fact, count: int
) -> tuple[tuple[str, str, str], ...]:
    """每条引用所属断言的三元组 ``(头概念名, 关系, 尾概念名)``，与引用平行。

    ``Fact.cited_spans`` 只给证据段与锚点；选句还要知道这条证据是哪一步的
    断言。两跳结论的不同跳断言不同，不能拿整条结论的头尾去套每一跳。
    """
    if fact.path:
        return tuple(
            (step.head.name, step.relation, step.tail.name)
            for step in fact.path
            for _ in step.evidence_ids
        )
    return ((fact.head.name, fact.relation, fact.tail.name),) * count


def _selected_anchor(
    segment: EvidenceRecord,
    *,
    head: str,
    relation: str,
    tail: str,
    ontology: Ontology,
) -> EvidenceAnchor | None:
    """按断言词法重合选出引用窗口；选不出时返回 ``None``（整段引用）。"""
    relation_type = ontology.relation_types.get(relation)
    label = relation_type.label if relation_type is not None else relation
    window = select_citation_window(
        segment.text, head=head, relation=label, tail=tail
    )
    if window is None:
        return None
    return EvidenceAnchor(
        evidence_id=segment.id, start=window.start, end=window.end
    )


def _anchor_fits(text: str, anchor: EvidenceAnchor) -> bool:
    """锚点区间是否落在证据段原文之内。

    锚点由入库时固化，切分规则冻结，正常情况下一定成立。读取侧仍不假设数据
    完好：越界的锚点当作没有锚点、整段引用，而不是让整次提问失败。
    """
    return 0 <= anchor.start < anchor.end <= len(text)


def _focus_candidates(
    matches: tuple[ConceptMatch, ...], ontology: Ontology
) -> tuple[ConceptMatch, ...]:
    """白名单关系的定义域内的命中才可能成为检索起点。

    设备、部件、故障模式都是白名单关系的定义域，因此设备与部件问题现在也
    有起点；症状、检测手段、原因只是关系的值域，不能当起点。
    """
    head_types = frozenset(
        entity_type
        for relation in EDGE_WHITELIST
        for entity_type in ontology.relation_types[relation].domain
    )
    return tuple(match for match in matches if match.concept.type in head_types)


def _comparison_anchors(
    candidates: tuple[ConceptMatch, ...],
) -> tuple[ConceptMatch, ...] | None:
    """同级命中的多个概念：它们是问句里的多个锚点，不是需要猜的歧义。

    锚点必须①与最具体的命中同类型；②具体程度在 ``MAX_ANCHOR_SPECIFICITY_GAP``
    以内；③是问句里各自独立的提及。落在同一跨度上的多个概念（同一个别名指向
    多个规范实例，如「磨粒磨损」）仍是一处歧义，不是两个锚点，交给调用方拒答。
    """
    best = max(candidates, key=lambda match: match.specificity)
    anchors = tuple(
        match
        for match in candidates
        if match.concept.type == best.concept.type
        and match.kind is best.kind
        and best.specificity[1] - len(match.alias) <= MAX_ANCHOR_SPECIFICITY_GAP
    )
    if len(anchors) < 2:
        return None
    if len({match.alias for match in anchors}) < len(anchors):
        return None
    return anchors


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
    reason: RefusalReason,
    model_calls: tuple[ModelCall, ...] = (),
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
        model_calls=model_calls,
        refusal_reason=reason,
    )

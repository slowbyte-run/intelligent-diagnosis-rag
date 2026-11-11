"""领域包第二个公开门面：一个问题从问句走到带引用的结构化答案。

调用方只依赖 ``ask(question, *, deps) -> Answer``；图存储、文献名目录与向量
兜底的三件依赖（嵌入、向量检索、重排）都由 ``deps`` 注入，测试用内存假实现。

检索有先后，不并行：

1. **图路径先走**：本体别名表定位问句里的概念，从最具体的命中出发沿白名单
   关系走至多两跳（见 ``traversal``），每条结论的证据段回填文献名、页码与
   原文。结论本身带完整路径，答案因此能说明「怎么走到的」。图拿出结论时
   向量检索一次都不发生——结构化结论不该被一堆语义相近的段落盖住。
2. **图没有依据时落向量**：``vector_recall`` 顺序扫描 pgvector 取候选、过重排、
   过阈值，答案标 ``AnswerPath.VECTOR``，``facts`` 为空，证据段直接进
   ``citations`` 并带上重排分数。
3. **两边都没有依据才拒答**：``detail`` 分别写清图路径缺在哪、向量兜底差多少；
   ``refusal_reason`` 区分 ``no_concept``（概念没识别出来）与 ``no_evidence``
   （识别了但没有依据），用户据此知道该换问法还是该补语料。

拒答是一个决定，不是降级：阈值来自配置（``vector_min_rerank_score``，校准方法
见 ``docs/retrieval-calibration.md``），低于阈值不答题；拒答答案的 ``facts`` 与
``citations`` 恒为空，由 ``Answer`` 的值对象不变量守住。

歧义问题不兜底：问句同时指向多个同样具体的概念时，向量召回分不清该答哪一个，
兜底只会把歧义藏进一个像模像样的答案里。

答案必须带引用（AGENTS.md 第 6 条）：没有证据段的事实、或路径上有一步
拿不到证据的结论，在组装阶段就被排除，不进入 ``facts``。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.documents import DocumentNames
from idr_core.embedding import Embedder
from idr_core.graph import (
    EvidenceRecord,
    GraphReader,
)
from idr_core.llm import ModelCall
from idr_core.llm.rerank import Reranker
from idr_core.vector_store import VectorReader
from idr_rotating.answer import (
    Answer,
    AnswerPath,
    Citation,
    Fact,
    RefusalReason,
)
from idr_rotating.linking import ConceptMatch, link_question
from idr_rotating.ontology import Ontology, load_ontology
from idr_rotating.retrieval import RetrievalSettings, load_retrieval_settings
from idr_rotating.traversal import (
    EDGE_WHITELIST,
    TraversalPath,
    traverse,
)
from idr_rotating.vector_recall import EvidenceRecall, recall_evidence


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
        )

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

    graph_answer = _answer_from_graph(question, deps, matches, focus)
    if graph_answer.path is AnswerPath.GRAPH:
        return graph_answer
    return _fall_back_or_refuse(question, deps, graph_answer)


def _fall_back_or_refuse(
    question: str, deps: AskDeps, graph_answer: Answer
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
) -> Answer:
    """图命中路径：至多两跳的带证据事实 + 证据回填。

    路径上任一步缺证据就丢弃该条结论；全部路径都不可用时退成显式空答案。
    结果集撞到上界时照常作答，但把截断写进 ``detail`` 并置 ``truncated``。
    """
    focus_node = next(
        iter(deps.graph_store.concepts_by_ids([focus.concept.id])), None
    )
    if focus_node is None:
        return _refusal(
            question,
            matches=matches,
            focus=focus,
            detail=f"概念「{focus.concept.name}」不在图谱中。",
            reason=RefusalReason.NO_EVIDENCE,
        )

    traversal = traverse(deps.graph_store, focus_node)
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for path in traversal.paths
            for step in path.steps
            for evidence_id in step.evidence_ids
        )
    )
    segments = {
        segment.id: segment
        for segment in deps.graph_store.evidence_by_ids(evidence_ids)
    }
    facts = _facts_with_evidence(traversal.paths, segments)
    if not facts:
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
        facts=facts,
        citations=_citations_for(facts, segments, deps.document_names),
        detail=(
            f"结果超过 {traversal.limit} 条上限（命中 {traversal.found} 条），"
            "已按关系类型与概念名称排序截断。"
            if traversal.truncated
            else None
        ),
        truncated=traversal.truncated,
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

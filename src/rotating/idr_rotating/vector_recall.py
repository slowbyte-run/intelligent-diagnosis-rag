"""图路径无命中时的向量兜底：查询改写 → 双通道召回 → 重排 → 阈值。

五件事，各有唯一职责：

0. **查询改写**只在向量路径发生：按 ``retrieval.yaml`` 的开关决定是否补全本体
   概念词、是否加 Qwen3-Embedding 的检索指令（见 ``query_rewrite``）。库里的
   向量不变，因此改的是查询侧，不需要重新嵌入语料。
1. **召回**有两条通道：向量通道问「哪些证据段的向量离问题最近」，pgvector
   顺序扫描取前 ``vector_top_k`` 条；词面通道拿本体命中的概念词去证据原文里
   做子串匹配，精确术语命中是语义近邻的盲区。两路名次用 RRF 融合，取前
   ``vector_top_k`` 条送重排；同一候选不会因为两路都命中而重复出现。
2. **重排**是第二个模型调用：原问句与候选原文一起过重排服务，分数取代向量距离
   与融合名次成为唯一的排序依据。查询改写不进入重排，避免把补全的概念词当成
   用户真的问过的内容。
3. **阈值**只在这里生效：分数低于 ``vector_min_rerank_score`` 的候选没有依据
   价值，进不了答案；图路径的结果不经过本模块，因此不会被重排打乱。

召回不到、或召回的候选在图上找不到原文时，本模块返回空的 ``segments``，
由调用方决定怎么说「没有依据」——这里不做拒答措辞。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from idr_core.embedding import Embedder
from idr_core.graph import EvidenceRecord, EvidenceTextSearch, GraphReader
from idr_core.llm import ModelCall, embed_texts
from idr_core.llm.rerank import Reranker, rerank_documents
from idr_core.vector_store import VectorReader
from idr_rotating.ontology import Ontology
from idr_rotating.query_rewrite import build_retrieval_query, lexical_terms
from idr_rotating.retrieval import RetrievalSettings

# Reciprocal Rank Fusion 的平滑常数：名次从 1 开始，常数越大越不看重头部名次。
# 60 是 RRF 的常用取值；两路同权，因为向量距离与词面命中数不可比，按名次融合
# 是唯一不需要调分数尺度的做法。金标准集实测（#20）见 retrieval.yaml 注释。
RRF_CONSTANT = 60


@dataclass(frozen=True, slots=True)
class RecalledSegment:
    """一条过完阈值的候选：证据段原文与它的重排分数。"""

    segment: EvidenceRecord
    score: float


@dataclass(frozen=True, slots=True)
class EvidenceRecall:
    """一次向量兜底的全部结果与诊断信息。

    ``recalled`` 是送进重排的候选数（向量与词面融合后），``top_score`` 是重排
    最高分（没有候选时为 ``None``）；两者一起让「为什么没答」可以写成可核对的
    数字。
    """

    segments: tuple[RecalledSegment, ...]
    model_calls: tuple[ModelCall, ...]
    recalled: int
    top_score: float | None


def fuse_rankings(
    vector_ids: Sequence[str],
    lexical_ids: Sequence[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    """按名次融合两路召回（RRF）：``1/(60+rank)`` 相加，同分按标识排序。

    名次而不是分数：余弦距离与词面命中数的尺度不可比，直接相加等于让其中一个
    通道吞掉另一个。没有词面命中时返回的集合与纯向量完全一致（按向量名次），
    因此关掉词面召回不会改变已有行为。
    """
    scores: dict[str, float] = {}
    for ids in (vector_ids, lexical_ids):
        for rank, evidence_id in enumerate(ids, start=1):
            scores[evidence_id] = scores.get(evidence_id, 0.0) + 1.0 / (
                RRF_CONSTANT + rank
            )
    return tuple(
        sorted(scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id))[
            :limit
        ]
    )


def recall_evidence(
    question: str,
    *,
    embedder: Embedder,
    vector_store: VectorReader,
    reranker: Reranker,
    graph_store: GraphReader,
    settings: RetrievalSettings,
    ontology: Ontology,
    text_search: EvidenceTextSearch | None = None,
) -> EvidenceRecall:
    """问题 → 改写后的查询 → 双通道候选 → 重排分数；返回过阈值的前几条。"""
    query = build_retrieval_query(question, settings=settings, ontology=ontology)
    lexical_ids = _lexical_candidates(
        question, settings=settings, ontology=ontology, text_search=text_search
    )
    query_vectors, embed_call = embed_texts(embedder, [query])
    matches = vector_store.search_similar(
        query_vectors[0], limit=settings.vector_top_k
    )
    vector_ids = tuple(match.evidence_id for match in matches)
    candidate_ids = (
        fuse_rankings(vector_ids, lexical_ids, limit=settings.vector_top_k)
        if lexical_ids
        else vector_ids
    )
    if not candidate_ids:
        return EvidenceRecall(
            segments=(), model_calls=(embed_call,), recalled=0, top_score=None
        )

    segments = {
        record.id: record
        for record in graph_store.evidence_by_ids(list(candidate_ids))
    }
    # 融合名次只决定候选集与送进重排的顺序；找不到原文的候选直接丢掉。
    candidates = [
        segments[evidence_id]
        for evidence_id in candidate_ids
        if evidence_id in segments
    ]
    if not candidates:
        return EvidenceRecall(
            segments=(),
            model_calls=(embed_call,),
            recalled=len(candidate_ids),
            top_score=None,
        )

    response, rerank_call = rerank_documents(
        reranker, question, [candidate.text for candidate in candidates]
    )
    scores = {item.index: item.score for item in response.scores}
    ranked = sorted(
        (
            RecalledSegment(segment=candidate, score=scores[index])
            for index, candidate in enumerate(candidates)
        ),
        key=lambda item: (-item.score, item.segment.id),
    )
    kept = tuple(
        item for item in ranked if item.score >= settings.vector_min_rerank_score
    )[: settings.vector_max_results]
    return EvidenceRecall(
        segments=kept,
        model_calls=(embed_call, rerank_call),
        recalled=len(candidate_ids),
        top_score=ranked[0].score if ranked else None,
    )


def _lexical_candidates(
    question: str,
    *,
    settings: RetrievalSettings,
    ontology: Ontology,
    text_search: EvidenceTextSearch | None,
) -> tuple[str, ...]:
    """词面通道：没有开关、没有词或没有实现端口时返回空（纯向量行为）。"""
    if not settings.lexical_recall or text_search is None:
        return ()
    terms = lexical_terms(question, ontology)
    if not terms:
        return ()
    return text_search.search_evidence(terms, limit=settings.vector_top_k)

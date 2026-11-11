"""图路径无命中时的向量兜底：顺序扫描召回 → 重排 → 阈值。

三件事，各有唯一职责：

1. **召回**只问「哪些证据段的向量离问题最近」：问题过一个嵌入，pgvector 顺序
   扫描取前 ``vector_top_k`` 条。证据段的文献名、页码与原文不在向量表里，按
   证据段标识回图存储取——向量表只存向量，引用只有图这一个事实源。
2. **重排**是第二个模型调用：查询与候选原文一起过重排服务，分数取代向量距离
   成为唯一的排序依据（向量距离只负责把候选带进来）。
3. **阈值**只在这里生效：分数低于 ``vector_min_rerank_score`` 的候选没有依据
   价值，进不了答案；图路径的结果不经过本模块，因此不会被重排打乱。

召回不到、或召回的候选在图上找不到原文时，本模块返回空的 ``segments``，
由调用方决定怎么说「没有依据」——这里不做拒答措辞。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.embedding import Embedder
from idr_core.graph import EvidenceRecord, GraphReader
from idr_core.llm import ModelCall, embed_texts
from idr_core.llm.rerank import Reranker, rerank_documents
from idr_core.vector_store import VectorReader
from idr_rotating.retrieval import RetrievalSettings


@dataclass(frozen=True, slots=True)
class RecalledSegment:
    """一条过完阈值的候选：证据段原文与它的重排分数。"""

    segment: EvidenceRecord
    score: float


@dataclass(frozen=True, slots=True)
class EvidenceRecall:
    """一次向量兜底的全部结果与诊断信息。

    ``recalled`` 是向量召回的候选数，``top_score`` 是重排最高分（没有候选时
    为 ``None``）；两者一起让「为什么没答」可以写成可核对的数字。
    """

    segments: tuple[RecalledSegment, ...]
    model_calls: tuple[ModelCall, ...]
    recalled: int
    top_score: float | None


def recall_evidence(
    question: str,
    *,
    embedder: Embedder,
    vector_store: VectorReader,
    reranker: Reranker,
    graph_store: GraphReader,
    settings: RetrievalSettings,
) -> EvidenceRecall:
    """问题 → 候选证据段 → 重排分数；返回过阈值的前几条与两次模型调用记录。"""
    query_vectors, embed_call = embed_texts(embedder, [question])
    matches = vector_store.search_similar(
        query_vectors[0], limit=settings.vector_top_k
    )
    if not matches:
        return EvidenceRecall(
            segments=(), model_calls=(embed_call,), recalled=0, top_score=None
        )

    segments = {
        record.id: record
        for record in graph_store.evidence_by_ids(
            [match.evidence_id for match in matches]
        )
    }
    # 召回顺序由向量距离给出；找不到原文的候选直接丢掉，不送进重排浪费 token。
    candidates = [
        segments[match.evidence_id]
        for match in matches
        if match.evidence_id in segments
    ]
    if not candidates:
        return EvidenceRecall(
            segments=(),
            model_calls=(embed_call,),
            recalled=len(matches),
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
        recalled=len(matches),
        top_score=ranked[0].score if ranked else None,
    )

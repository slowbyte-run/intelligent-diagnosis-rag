"""ask 门面：图路径无命中时的向量兜底。

全部在 ``ask`` 这一层测外部行为：图命中时向量检索一次都不发生；图没命中时
问题向量召回证据段、重排、过阈值，答案与图路径同形（同样的文献名、页码、
原文与引用标识），只是 ``path`` 标为 ``vector``、``facts`` 为空。

单元测试只依赖内存假存储与假模型，不连 Neo4j、Postgres 或网络。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fakes import FakeGraphStore
from idr_core.graph import ConceptNode, ConceptRelation, GraphBatch, Provenance
from idr_core.llm import EmbeddingResponse
from idr_core.llm.rerank import RerankResponse, RerankScore
from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_core.vector_store import VectorMatch
from idr_rotating import AnswerPath, AskDeps, ask
from idr_rotating.retrieval import RetrievalSettings

OUTER_RACE = "concept:bearing_outer_race_fatigue_spalling"
BPFO = "concept:bpfo_harmonics"
ENVELOPE = "concept:envelope_demodulation"

OUT_OF_GRAPH_QUESTION = "振动信号里出现周期性冲击应该检查什么"
GRAPH_QUESTION = "轴承外圈疲劳剥落有什么征兆"

SYMPTOM_TEXT = "外圈剥落会在频谱上出现 BPFO 及其倍频。"
DETECTION_TEXT = "包络解调可以确认外圈剥落的冲击。"
SYMPTOM_ID = f"ev:doc-a:4:0-{len(SYMPTOM_TEXT)}"
DETECTION_ID = f"ev:doc-a:5:100-{100 + len(DETECTION_TEXT)}"


class FakeDocumentNames:
    def __init__(self, names: Mapping[str, str]) -> None:
        self._names = dict(names)

    def name_for(self, doc_id: str) -> str | None:
        return self._names.get(doc_id)


class FakeVectorReader:
    """按构造顺序返回候选，并记录查询向量与召回条数。"""

    def __init__(self, matches: Sequence[VectorMatch] = ()) -> None:
        self._matches = tuple(matches)
        self.calls: list[tuple[list[float], int]] = []

    def search_similar(
        self, vector: Sequence[float], *, limit: int
    ) -> tuple[VectorMatch, ...]:
        self.calls.append((list(vector), limit))
        return self._matches[:limit]


class FakeReranker:
    """按原文查表给分，记录送进重排的原文；未登记的原文给 0 分。"""

    def __init__(
        self,
        scores: Mapping[str, float],
        *,
        model: str = "fake-rerank",
        input_tokens: int = 11,
        output_tokens: int = 0,
    ) -> None:
        self._scores = dict(scores)
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResponse:
        self.calls.append((query, tuple(documents)))
        return RerankResponse(
            scores=tuple(
                RerankScore(index=index, score=self._scores.get(text, 0.0))
                for index, text in enumerate(documents)
            ),
            model=self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


class FakeEmbedder:
    """返回按文本查表的定长向量，并记录被嵌入的文本。"""

    def __init__(self, vectors: Mapping[str, Sequence[float]] | None = None) -> None:
        self._vectors = dict(vectors or {})
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [
            list(self._vectors.get(text, [float(len(text))] * 4)) for text in texts
        ]


class FakeUsageEmbedder(FakeEmbedder):
    """同时报告模型名与输入 token 的嵌入客户端，让成本记录有数可查。"""

    def embed_with_usage(self, texts: Sequence[str]) -> EmbeddingResponse:
        return EmbeddingResponse(
            vectors=self.embed(texts), model="fake-embedding", input_tokens=7
        )


def concept(instance_id: str, name: str, type_: str) -> ConceptNode:
    return ConceptNode(id=f"concept:{instance_id}", name=name, type=type_, doc_id=None)


def evidence(
    doc_id: str, *, page: int, offset: int, text: str
) -> EvidenceSegment:
    return EvidenceSegment(
        id=f"ev:{doc_id}:{page}:{offset}-{offset + len(text)}",
        doc_id=doc_id,
        page=page,
        bbox=BBox(0.0, 0.0, 595.0, 842.0),
        charspan=CharSpan(offset, offset + len(text)),
        text=text,
    )


def relation(
    head: ConceptNode,
    type_: str,
    tail: ConceptNode,
    *,
    doc_id: str,
    evidence_ids: tuple[str, ...],
    confidence: float = 0.8,
) -> ConceptRelation:
    return ConceptRelation(
        head_id=head.id,
        type=type_,
        tail_id=tail.id,
        provenance=Provenance(
            doc_id=doc_id,
            evidence_ids=evidence_ids,
            confidence=confidence,
            source="extracted",
        ),
    )


def write_one_hop_graph(graph: FakeGraphStore) -> None:
    """一篇文献：外圈剥落有征兆与检测手段各一条，证据段落在 4/5 页。"""
    doc_id = "doc-a"
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    envelope = concept(ENVELOPE.removeprefix("concept:"), "包络解调", "detection_method")
    symptom = evidence(doc_id, page=4, offset=0, text=SYMPTOM_TEXT)
    detection = evidence(doc_id, page=5, offset=100, text=DETECTION_TEXT)
    assert symptom.id == SYMPTOM_ID
    assert detection.id == DETECTION_ID
    graph.write_document(
        GraphBatch(
            doc_id=doc_id,
            concepts=(outer, bpfo, envelope),
            evidence=(symptom, detection),
            relations=(
                relation(
                    outer,
                    "detected_by",
                    envelope,
                    doc_id=doc_id,
                    evidence_ids=(detection.id,),
                    confidence=0.7,
                ),
                relation(
                    outer,
                    "indicated_by",
                    bpfo,
                    doc_id=doc_id,
                    evidence_ids=(symptom.id,),
                    confidence=0.8,
                ),
            ),
            evidence_links=(),
        )
    )


@dataclass
class VectorDeps:
    """内存依赖 + 每类假实现，便于断言「谁被调用了几次」。"""

    deps: AskDeps
    graph: FakeGraphStore
    embedder: FakeEmbedder
    vector_store: FakeVectorReader
    reranker: FakeReranker


def build_vector_deps(
    *,
    matches: Sequence[tuple[str, float]] = (),
    rerank_scores: Mapping[str, float] | None = None,
    top_k: int = 3,
    min_score: float = 0.35,
    max_results: int | None = None,
    usage_embedder: bool = False,
) -> VectorDeps:
    """候选来自向量召回，分数来自重排；证据段原文与页码挂在图存储上。"""
    graph = FakeGraphStore()
    write_one_hop_graph(graph)
    embedder = FakeUsageEmbedder() if usage_embedder else FakeEmbedder()
    vector_store = FakeVectorReader(
        tuple(
            VectorMatch(evidence_id=evidence_id, distance=distance)
            for evidence_id, distance in matches
        )
    )
    reranker = FakeReranker(rerank_scores or {})
    deps = AskDeps(
        graph_store=graph,
        document_names=FakeDocumentNames({"doc-a": "论文A.pdf"}),
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        retrieval_settings=RetrievalSettings(
            vector_top_k=top_k,
            vector_min_rerank_score=min_score,
            vector_max_results=max_results or top_k,
        ),
    )
    return VectorDeps(
        deps=deps,
        graph=graph,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
    )


def test_graph_hit_does_not_trigger_vector_retrieval() -> None:
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.1),),
        rerank_scores={SYMPTOM_TEXT: 0.99},
    )

    answer = ask(GRAPH_QUESTION, deps=built.deps)

    assert answer.path is AnswerPath.GRAPH
    assert built.embedder.texts == []
    assert built.vector_store.calls == []
    assert built.reranker.calls == []
    assert answer.model_calls == ()


def test_graph_miss_falls_back_to_reranked_evidence() -> None:
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.05), (DETECTION_ID, 0.15)),
        rerank_scores={SYMPTOM_TEXT: 0.91, DETECTION_TEXT: 0.62},
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert answer.path is AnswerPath.VECTOR
    assert answer.matches == ()
    assert answer.focus is None
    assert answer.facts == ()
    assert answer.detail is None
    assert [citation.evidence_id for citation in answer.citations] == [
        SYMPTOM_ID,
        DETECTION_ID,
    ]


def test_vector_citations_carry_the_same_attribution_as_graph_citations() -> None:
    """向量路径的每条证据都带证据段标识、文献名、页码与原文。"""
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.05),),
        rerank_scores={SYMPTOM_TEXT: 0.8},
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    (citation,) = answer.citations
    assert citation.evidence_id == SYMPTOM_ID
    assert citation.doc_id == "doc-a"
    assert citation.document == "论文A.pdf"
    assert citation.page == 4
    assert citation.text == SYMPTOM_TEXT
    assert citation.score == 0.8


def test_vector_recall_embeds_the_question_and_asks_for_top_k_candidates() -> None:
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.05), (DETECTION_ID, 0.06)),
        rerank_scores={SYMPTOM_TEXT: 0.8, DETECTION_TEXT: 0.7},
        top_k=2,
    )

    ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert built.embedder.texts == [OUT_OF_GRAPH_QUESTION]
    (query_vector, limit) = built.vector_store.calls[0]
    assert query_vector == [float(len(OUT_OF_GRAPH_QUESTION))] * 4
    assert limit == 2


def test_rerank_scores_decide_the_citation_order_not_vector_distance() -> None:
    """向量距离只负责把候选带进来；答案顺序由重排分数决定。"""
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.01), (DETECTION_ID, 0.02)),
        rerank_scores={SYMPTOM_TEXT: 0.51, DETECTION_TEXT: 0.93},
        top_k=2,
        min_score=0.35,
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert [citation.evidence_id for citation in answer.citations] == [
        DETECTION_ID,
        SYMPTOM_ID,
    ]
    assert [citation.score for citation in answer.citations] == [0.93, 0.51]


def test_rerank_sees_every_recalled_candidate_but_only_keeps_the_passing_ones() -> None:
    """阈值只筛答案，不筛送进重排的候选：否则高分候选的位置无从判断。"""
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.01), (DETECTION_ID, 0.02)),
        rerank_scores={SYMPTOM_TEXT: 0.55, DETECTION_TEXT: 0.2},
        top_k=2,
        min_score=0.35,
        max_results=2,
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert built.reranker.calls == [
        (OUT_OF_GRAPH_QUESTION, (SYMPTOM_TEXT, DETECTION_TEXT))
    ]
    assert [citation.evidence_id for citation in answer.citations] == [SYMPTOM_ID]


def test_vector_recall_skips_candidates_whose_text_cannot_be_traced() -> None:
    """向量表里可能有图里不存在的证据段；找不到原文的不进重排也不进答案。"""
    built = build_vector_deps(
        matches=(("ev:ghost:1:0-9", 0.0), (SYMPTOM_ID, 0.1)),
        rerank_scores={SYMPTOM_TEXT: 0.8},
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert built.reranker.calls == [(OUT_OF_GRAPH_QUESTION, (SYMPTOM_TEXT,))]
    assert [citation.evidence_id for citation in answer.citations] == [SYMPTOM_ID]


def test_vector_path_records_both_model_calls_with_tokens_and_elapsed_time() -> None:
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.05),),
        rerank_scores={SYMPTOM_TEXT: 0.8},
        usage_embedder=True,
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    embedding_call, rerank_call = answer.model_calls
    assert embedding_call.model == "fake-embedding"
    assert embedding_call.input_tokens == 7
    assert embedding_call.output_tokens is None
    assert embedding_call.elapsed_seconds >= 0.0
    assert rerank_call.model == "fake-rerank"
    assert rerank_call.input_tokens == 11
    assert rerank_call.output_tokens == 0
    assert rerank_call.elapsed_seconds >= 0.0


def test_low_rerank_scores_refuse_and_report_the_threshold() -> None:
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.01),),
        rerank_scores={SYMPTOM_TEXT: 0.2},
        min_score=0.5,
    )

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert answer.path is AnswerPath.REFUSAL
    assert answer.facts == ()
    assert answer.citations == ()
    assert answer.detail is not None
    assert "0.20" in answer.detail
    assert "0.50" in answer.detail
    assert "阈值" in answer.detail
    # 拒绝也花了两次模型调用，成本记录不能因为没答出来就丢掉。
    assert len(answer.model_calls) == 2


def test_empty_vector_table_refuses_without_a_rerank_call() -> None:
    built = build_vector_deps(matches=())

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=built.deps)

    assert answer.path is AnswerPath.REFUSAL
    assert built.reranker.calls == []
    assert answer.detail is not None and "没有召回" in answer.detail
    assert [call.model for call in answer.model_calls] == ["unknown"]


def test_without_vector_dependencies_the_graph_refusal_is_unchanged() -> None:
    built = build_vector_deps()

    answer = ask(OUT_OF_GRAPH_QUESTION, deps=AskDeps(graph_store=built.graph))

    assert answer.path is AnswerPath.REFUSAL
    assert answer.detail is not None and "无法定位检索起点" in answer.detail
    assert answer.model_calls == ()


def test_ambiguous_concepts_are_not_papered_over_by_vector_recall() -> None:
    built = build_vector_deps(
        matches=((SYMPTOM_ID, 0.01),),
        rerank_scores={SYMPTOM_TEXT: 0.99},
    )

    answer = ask("磨粒磨损有什么征兆", deps=built.deps)

    assert answer.path is AnswerPath.REFUSAL
    assert "多个" in (answer.detail or "")
    assert built.embedder.texts == []
    assert built.reranker.calls == []


def test_graph_path_fact_order_is_untouched_when_a_reranker_is_available() -> None:
    built = build_vector_deps(
        matches=((DETECTION_ID, 0.01), (SYMPTOM_ID, 0.02)),
        rerank_scores={DETECTION_TEXT: 0.99, SYMPTOM_TEXT: 0.1},
    )

    answer = ask(GRAPH_QUESTION, deps=built.deps)

    assert answer.path is AnswerPath.GRAPH
    assert [fact.relation for fact in answer.facts] == [
        "indicated_by",
        "detected_by",
    ]
    assert [citation.score for citation in answer.citations] == [None, None]

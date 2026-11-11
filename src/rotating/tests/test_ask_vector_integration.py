"""ask 向量兜底在真实 Neo4j、pgvector 与 SiliconFlow 上的契约测试。

这条测试**会真实调用计费的嵌入与重排模型**（一个问题、二十段候选），默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration

它验证的是线上那张知识库能被「图里没有的问题」问到：向量召回走真实的顺序
扫描，重排分数来自真实服务，引用回到真实证据段的文献标识与页码，成本记录
里两次模型调用都有 token 与耗时。也验证校准后的阈值确实把「语料里没有」的
问题挡在生成之外，并给出正确的拒答原因。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from idr_core.config import load_config
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.llm.embedding import OpenAICompatibleEmbedder
from idr_core.llm.rerank import OpenAICompatibleReranker
from idr_core.vector_store.pgvector import PgVectorStore
from idr_rotating import AnswerPath, AskDeps, RefusalReason, ask
from idr_rotating.retrieval import load_retrieval_settings

# 图外问题：本体别名表一个概念都匹配不到，图路径必然落空；语料里的中文专利
# 正好写了这个测量步骤，向量兜底要能把它捞回来并给出页码。
QUESTION = "现场测振传感器应该测哪几个方向"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            os.environ.get("NEO4J_PASSWORD")
            and os.environ.get("DATABASE_URL")
            and os.environ.get("SILICONFLOW_API_KEY")
        ),
        reason="需要 NEO4J_PASSWORD、DATABASE_URL 与 SILICONFLOW_API_KEY（先 source .env）",
    ),
]


@pytest.fixture
def deps() -> Iterator[AskDeps]:
    config = load_config()
    graph_store = Neo4jGraphStore(
        config.neo4j_uri, config.neo4j_user, config.neo4j_password
    )
    try:
        yield AskDeps(
            graph_store=graph_store,
            embedder=OpenAICompatibleEmbedder(
                base_url=config.siliconflow_base_url,
                api_key=config.siliconflow_api_key,
                model=config.embedding_model,
            ),
            vector_store=PgVectorStore(config.database_url),
            reranker=OpenAICompatibleReranker(
                base_url=config.siliconflow_base_url,
                api_key=config.siliconflow_api_key,
                model=config.rerank_model,
            ),
        )
    finally:
        graph_store.close()


def test_a_question_outside_the_ontology_is_answered_from_reranked_evidence(
    deps: AskDeps,
) -> None:
    settings = load_retrieval_settings()

    answer = ask(QUESTION, deps=deps)

    # 图里没有这个概念：图路径必然落空，答案应来自向量兜底而不是拒答。
    assert answer.matches == ()
    assert answer.focus is None
    assert answer.path is AnswerPath.VECTOR, answer.detail
    assert answer.facts == ()
    assert answer.detail is None
    assert answer.citations

    scores = [citation.score for citation in answer.citations]
    assert all(score is not None for score in scores)
    numeric_scores = [score for score in scores if score is not None]
    assert len(numeric_scores) <= settings.vector_max_results
    assert all(
        score >= settings.vector_min_rerank_score for score in numeric_scores
    )
    assert numeric_scores == sorted(numeric_scores, reverse=True)

    # 每条引用都要能回到具体证据段、文献标识与页码；原文不能是空的。
    for citation in answer.citations:
        assert citation.evidence_id.startswith("ev:")
        assert citation.doc_id
        assert citation.page >= 1
        assert citation.text.strip()

    # 两次模型调用（嵌入 + 重排）都要留下 token 与耗时，成本核算才有依据。
    embedding_call, rerank_call = answer.model_calls
    assert embedding_call.model
    assert embedding_call.input_tokens is not None and embedding_call.input_tokens > 0
    assert embedding_call.elapsed_seconds > 0.0
    assert rerank_call.model
    assert rerank_call.input_tokens is not None and rerank_call.input_tokens > 0
    assert rerank_call.elapsed_seconds > 0.0


NO_EVIDENCE_QUESTION = "汽轮机有什么故障模式？"
NO_CONCEPT_QUESTION = "冷却塔的布水器堵塞如何判断？"


def test_a_linked_concept_without_corpus_evidence_refuses_as_no_evidence(
    deps: AskDeps,
) -> None:
    """汽轮机在本体里、语料里没有：识别了概念，但没有依据。"""
    answer = ask(NO_EVIDENCE_QUESTION, deps=deps)

    assert answer.path is AnswerPath.REFUSAL
    assert answer.refusal_reason is RefusalReason.NO_EVIDENCE
    assert answer.matches
    assert answer.facts == ()
    assert answer.citations == ()
    assert answer.detail is not None and "阈值" in answer.detail


def test_a_question_outside_the_ontology_refuses_as_no_concept(
    deps: AskDeps,
) -> None:
    answer = ask(NO_CONCEPT_QUESTION, deps=deps)

    assert answer.path is AnswerPath.REFUSAL
    assert answer.refusal_reason is RefusalReason.NO_CONCEPT
    assert answer.matches == ()
    assert answer.facts == ()
    assert answer.citations == ()

"""检索查询构造：把问句变成送给嵌入模型的文本。

问句是中文、证据多为英文；本模块只做确定性改写——本体别名表里命中的概念
名称、别名与概念 ID 的英文词，以及可选的检索指令。不调模型，同一句话永远
得到同一个查询文本，因此诊断里哪个变体变好或变坏可以逐条核对。
"""

from __future__ import annotations

from idr_rotating.ontology import load_ontology
from idr_rotating.query_rewrite import (
    CROSS_LINGUAL_INSTRUCTION,
    apply_instruction,
    build_retrieval_query,
    enrich_question,
    lexical_terms,
)
from idr_rotating.retrieval import RetrievalSettings

QUESTION = "轴承外圈剥落有什么征兆？"


def test_enrichment_appends_canonical_names_aliases_and_english_id_words() -> None:
    ontology = load_ontology()

    enriched = enrich_question(QUESTION, ontology)

    assert enriched.startswith(f"{QUESTION}\n")
    # 规范名与别名来自本体，英文词来自概念 ID 的 snake_case 分段。
    assert "轴承外圈疲劳剥落" in enriched
    assert "外圈剥落" in enriched
    for word in ("bearing", "outer", "race", "fatigue", "spalling"):
        assert word in enriched.split()
    # 同一个词只出现一次：别名之间、别名与 ID 之间可能重复。
    terms = enriched.split("\n", 1)[1].split()
    assert len(terms) == len(set(terms))


def test_enrichment_is_deterministic() -> None:
    ontology = load_ontology()

    assert enrich_question(QUESTION, ontology) == enrich_question(
        QUESTION, ontology
    )


def test_a_question_without_ontology_matches_is_left_untouched() -> None:
    ontology = load_ontology()

    question = "冷却塔的布水器堵塞如何判断？"

    assert enrich_question(question, ontology) == question


def test_instruction_uses_the_official_qwen_query_format() -> None:
    instructed = apply_instruction(QUESTION, "crosslingual")

    assert instructed == (
        f"Instruct: {CROSS_LINGUAL_INSTRUCTION}\nQuery:{QUESTION}"
    )


def test_an_empty_instruction_preset_keeps_the_raw_question() -> None:
    assert apply_instruction(QUESTION, "") == QUESTION


def test_configured_settings_enrich_then_instruct() -> None:
    ontology = load_ontology()
    settings = RetrievalSettings(
        query_enrichment=True, query_instruction="crosslingual"
    )

    query = build_retrieval_query(QUESTION, settings=settings, ontology=ontology)

    assert query.startswith(
        f"Instruct: {CROSS_LINGUAL_INSTRUCTION}\nQuery:{QUESTION}\n"
    )
    assert "轴承外圈疲劳剥落" in query


def test_default_settings_embed_the_raw_question() -> None:
    ontology = load_ontology()

    query = build_retrieval_query(
        QUESTION, settings=RetrievalSettings(), ontology=ontology
    )

    assert query == QUESTION


def test_lexical_terms_keep_chinese_and_drop_short_english_words() -> None:
    """``one``、``x`` 这类词在英文正文里处处都是子串，做 CONTAINS 只会引入噪声。"""
    terms = lexical_terms(QUESTION, load_ontology())

    assert "外圈剥落" in terms
    assert "bearing" in terms
    assert all(not term.isascii() or len(term) >= 4 for term in terms)

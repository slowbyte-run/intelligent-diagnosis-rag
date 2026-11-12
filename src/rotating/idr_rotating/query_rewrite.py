"""检索查询构造：确定性改写，不调模型。

问句是中文，证据却以英文为主。这里只做两件可解释的事：

1. **概念补全**：用本体别名表找出问句里命中的概念，把规范名、全部别名与概念
   ID 的英文词（本体实例 ID 是 ``bearing_outer_race_fatigue_spalling`` 这样的
   snake_case）追加到问句后面。依据是 ``link_question`` 的同一张别名表，因此
   「为什么查询里出现这个词」可以逐条核对，不需要额外模型调用。
2. **检索指令**：Qwen3-Embedding 是 instruction-aware 的。按模型卡的要求，指令
   只加在查询侧（``Instruct: <task>\\nQuery:<query>``），库里的向量保持不变。
   指令文本按预设名选择，配置里只存预设名，避免同一句话在代码与配置中各写一遍。

改写不改变重排输入：重排服务仍然拿原问句与候选原文比对，改写只影响嵌入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from idr_rotating.linking import CONCEPT_ID_PREFIX, link_question
from idr_rotating.ontology import Ontology

if TYPE_CHECKING:
    from idr_rotating.retrieval import RetrievalSettings


# 模型卡给出的通用检索指令。它是公开基线，不是本项目调优的结果。
GENERIC_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# 本项目实测的中英跨语言检索指令（金标准集 recall@k 对比见
# config/retrieval.yaml 的注释）。中文问句、英文证据是主要障碍，指令里把
# 语言方向与领域说清楚；模型卡也建议指令用英文写、按场景定制。
CROSS_LINGUAL_INSTRUCTION = (
    "Given a Chinese question about rotating machinery faults, retrieve relevant "
    "English or Chinese passages that answer the question"
)

# 可选的指令预设：空字符串表示不加指令。
INSTRUCTION_PRESETS: dict[str, str] = {
    "generic": GENERIC_RETRIEVAL_INSTRUCTION,
    "crosslingual": CROSS_LINGUAL_INSTRUCTION,
}


def concept_terms(question: str, ontology: Ontology) -> tuple[str, ...]:
    """问句命中概念的中文名、别名与英文 ID 词，按命中顺序去重。

    同一个词可能既是别名又是 ID 分段（或出现在多个别名里），只保留第一次
    出现的位置；结果只用于拼接查询，不做展示。
    """
    instances = {
        instance.id: instance
        for group in ontology.instances.values()
        for instance in group
    }
    terms: list[str] = []
    for match in link_question(question, ontology):
        instance_id = match.concept.id.removeprefix(CONCEPT_ID_PREFIX)
        instance = instances.get(instance_id)
        candidates = (
            (
                (instance.label, *instance.aliases)
                if instance is not None
                else (match.concept.name,)
            )
            + tuple(instance_id.split("_"))
        )
        for term in candidates:
            if term and term not in terms:
                terms.append(term)
    return tuple(terms)


def lexical_terms(question: str, ontology: Ontology) -> tuple[str, ...]:
    """词面检索用的词：中文概念词 + 长度至少 4 的英文 ID 词。

    短英文词（``one``、``x``）在英文正文里几乎处处都是子串，直接做
    ``CONTAINS`` 只会把噪声塞进候选集；中文别名没有这个问题。阈值取 4 是
    实测出来的：3 字词（``fft``）带来的额外命中不如同长度噪声多（#20）。
    """
    return tuple(
        term
        for term in concept_terms(question, ontology)
        if not term.isascii() or len(term) >= 4
    )


def enrich_question(question: str, ontology: Ontology) -> str:
    """问句后追加概念词；没有任何命中时原样返回。"""
    terms = concept_terms(question, ontology)
    if not terms:
        return question
    return f"{question}\n{' '.join(terms)}"


def apply_instruction(question: str, preset: str) -> str:
    """按 Qwen3-Embedding 的官方格式包装查询；空预设表示不加指令。"""
    if not preset:
        return question
    try:
        instruction = INSTRUCTION_PRESETS[preset]
    except KeyError:
        choices = "、".join(sorted(INSTRUCTION_PRESETS))
        raise ValueError(
            f"未知的检索指令预设：{preset!r}（可选：{choices} 或空字符串）"
        ) from None
    return f"Instruct: {instruction}\nQuery:{question}"


def build_retrieval_query(
    question: str, *, settings: RetrievalSettings, ontology: Ontology
) -> str:
    """线上检索真正嵌入的文本：先按配置补概念，再加检索指令。"""
    query = (
        enrich_question(question, ontology)
        if settings.query_enrichment
        else question
    )
    return apply_instruction(query, settings.query_instruction)

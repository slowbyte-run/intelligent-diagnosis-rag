"""答案的值对象：检索路径、命中概念、结论与引用。

答案分两层：``facts`` 是图谱上的结论（带关系、置信度与证据段标识），
``citations`` 是这些证据段的出处（文献名、页码、原文）。一条事实的证据用
``evidence_ids`` 指向引用层，同一段证据被多条结论引用时只出现一次。

``path`` 明确回答「这条答案是怎么来的」——图命中、向量兜底还是拒答。本层
只表达事实，不设阈值：判定什么时候拒答是检索编排的事，答案只把结果如实标注。

向量兜底没有图上的结论，``facts`` 为空，证据段直接进 ``citations``；这类引用
带重排分数 ``score``（图路径为 ``None``，顺序由关系类型决定）。每条答案还带着
本次产生的模型调用记录，成本核算不需要另开一条统计通道。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from idr_core.graph import Concept
from idr_core.llm import ModelCall
from idr_rotating.linking import ConceptMatch


class AnswerPath(StrEnum):
    """本次回答走过的检索路径。"""

    # 图命中：沿本体关系取到带证据的事实。
    GRAPH = "graph"
    # 向量兜底：图没有命中时按语义召回证据段并重排（#16）。
    VECTOR = "vector"
    # 拒答：图与向量两条路都没有给出可用依据；判据与阈值校准在 #17。
    REFUSAL = "refusal"


@dataclass(frozen=True, slots=True)
class Citation:
    """一条结论的出处；``document`` 缺省表示按标识查不到文献名。

    ``score`` 只有向量兜底路径有值：它是这条证据段的重排分数，也是它排在这个
    位置的原因；图路径的引用没有分数，顺序由关系类型决定。
    """

    evidence_id: str
    doc_id: str
    document: str | None
    page: int
    text: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class Fact:
    """图上的一条结论：头概念沿某类关系指向尾概念，证据可回溯。"""

    relation: str
    head: Concept
    tail: Concept
    confidence: float
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    """一次提问的结果。空答案也是显式的：``path`` 说明为什么没答。"""

    question: str
    path: AnswerPath
    matches: tuple[ConceptMatch, ...]
    focus: ConceptMatch | None
    facts: tuple[Fact, ...]
    citations: tuple[Citation, ...]
    # 空答案时说明缺在哪一步（未命中概念、概念不在图谱、没有带证据的事实）。
    detail: str | None = None
    # 本次回答产生的模型调用（向量兜底的嵌入与重排）；图路径为空。
    model_calls: tuple[ModelCall, ...] = ()


# 2：引用增加重排分数，答案增加模型调用记录；字段只增不改，评估按版本号读。
ANSWER_SCHEMA_VERSION = 2


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    """答案的 JSON 形态；字段名即评估与接口层的契约，改动要升 schema 版本。"""
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "question": answer.question,
        "path": answer.path.value,
        "detail": answer.detail,
        "focus": _match_to_dict(answer.focus),
        "matches": [_match_to_dict(match) for match in answer.matches],
        "facts": [
            {
                "relation": fact.relation,
                "head": _concept_to_dict(fact.head),
                "tail": _concept_to_dict(fact.tail),
                "confidence": fact.confidence,
                "evidence_ids": list(fact.evidence_ids),
            }
            for fact in answer.facts
        ],
        "citations": [
            {
                "evidence_id": citation.evidence_id,
                "doc_id": citation.doc_id,
                "document": citation.document,
                "page": citation.page,
                "text": citation.text,
                "score": citation.score,
            }
            for citation in answer.citations
        ],
        "model_calls": [
            {
                "model": call.model,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "elapsed_seconds": call.elapsed_seconds,
            }
            for call in answer.model_calls
        ],
    }


def _match_to_dict(match: ConceptMatch | None) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        "concept": _concept_to_dict(match.concept),
        "alias": match.alias,
        "kind": match.kind.value,
    }


def _concept_to_dict(concept: Concept) -> dict[str, str]:
    return {"id": concept.id, "name": concept.name, "type": concept.type}

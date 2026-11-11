"""答案的值对象：检索路径、命中概念、结论与引用。

答案分两层：``facts`` 是图谱上的结论（带关系、置信度与证据段标识），
``citations`` 是这些证据段的出处（文献名、页码、原文）。一条事实的证据用
``evidence_ids`` 指向引用层，同一段证据被多条结论引用时只出现一次。

``path`` 明确回答「这条答案是怎么来的」——图命中、向量兜底还是拒答。本层
只表达事实，不设阈值：判定什么时候拒答是检索编排的事，答案只把结果如实标注。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from idr_core.graph import Concept
from idr_rotating.linking import ConceptMatch


class AnswerPath(StrEnum):
    """本次回答走过的检索路径。"""

    # 图命中：沿本体关系取到带证据的事实。
    GRAPH = "graph"
    # 向量兜底：图没有命中时按语义召回证据段（#16 接进来，本票不产生）。
    VECTOR = "vector"
    # 拒答：没有可用的带证据结论；本票只表示「空答案」，阈值策略在 #17。
    REFUSAL = "refusal"


@dataclass(frozen=True, slots=True)
class Citation:
    """一条结论的出处；``document`` 缺省表示按标识查不到文献名。"""

    evidence_id: str
    doc_id: str
    document: str | None
    page: int
    text: str


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


ANSWER_SCHEMA_VERSION = 1


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
            }
            for citation in answer.citations
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

"""答案的值对象：检索路径、命中概念、结论与引用。

答案分两层：``facts`` 是图谱上的结论（带关系、置信度与证据段标识），
``citations`` 是这些证据段的出处（文献名、页码、引文）。一条事实的证据用
``evidence_ids`` 指向引用层。引用带锚点时引文是证据段内的支撑片段，同一段
证据支撑多条断言时每条断言各有一条引用；短锚点的引文会并入同页相邻证据段
作为展示上下文（``citation_context``），锚点区间本身不变。没有锚点时整段引用，
同一证据段只出现一次。

``path`` 明确回答「这条答案是怎么来的」——图命中、向量兜底还是拒答。图命中的
结论还带 ``Fact.path``：从结论起点走到终点的每一步及其证据，答案因此能说明
「怎么走到的」，而不只是给出终点。向量兜底没有图上的结论，``facts`` 为空，
证据段直接进 ``citations``；这类引用带重排分数 ``score``（图路径为 ``None``，
顺序由关系类型决定）。每条答案还带着本次产生的模型调用记录，成本核算不需要
另开一条统计通道。

本层只表达事实，不设阈值：判定什么时候拒答是检索编排的事，答案只把结果如实
标注。``truncated`` 表示结果集撞到上界被截断，``detail`` 里写明原因。

对比型回答用 :class:`Comparison` 表达：两个锚点的事实按 ``(关系, 尾概念)``
分成共同项、各自独有项，以及文献明确写下的易混淆声明。对比答案的 ``facts``
为空、``focus`` 为空，结论只出现在分区里；单焦点答案的 ``facts`` / ``focus``
形态不变。

拒答有两种原因，由 :class:`RefusalReason` 区分：``no_concept`` 是问题里的概念
没被识别出来，``no_evidence`` 是识别了概念却没有证据支撑。这个区分是给用户的
行动指引——前者该换个问法，后者该补语料。两种拒答都不允许携带 ``facts``、
``citations`` 或 ``comparison``：值对象在构造时就拒绝这种自相矛盾的答案，而不是
靠调用点自觉。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from idr_core.graph import Concept, EvidenceAnchor
from idr_core.llm import ModelCall
from idr_rotating.linking import ConceptMatch
from idr_rotating.traversal import PathStep


class AnswerPath(StrEnum):
    """本次回答走过的检索路径。"""

    # 图命中：沿本体关系取到带证据的事实。
    GRAPH = "graph"
    # 向量兜底：图没有命中时按语义召回证据段并重排（#16）。
    VECTOR = "vector"
    # 拒答：图与向量两条路都没有给出可用依据。
    REFUSAL = "refusal"


class RefusalReason(StrEnum):
    """拒答原因：问题没被理解，还是理解了但没有依据。

    两者的行动含义不同：``NO_CONCEPT`` 说明问题里的说法没有落到本体概念上，
    用户换一种问法（或维护者补别名）就可能答得出来；``NO_EVIDENCE`` 说明概念
    已经识别，是图谱与文献里没有可用的证据，需要补语料或补抽取。
    """

    # 问题里没有任何片段链接到本体概念。
    NO_CONCEPT = "no_concept"
    # 链接到了概念，但没有带证据的事实，向量兜底也没有候选过阈值。
    NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True, slots=True)
class Citation:
    """一条结论的出处；``document`` 缺省表示按标识查不到文献名。

    ``text`` 是要展示给用户的引文：有锚点时以锚点区间对应的原文为基础，短于
    阈值时并入同页相邻证据段作为上下文（文本仍是页面文字的连续切片）；没有
    锚点时是整段证据原文。``anchor`` 记录支撑片段在 ``evidence_id`` 所指证据
    段里的字符区间（``None`` 表示整段）——它始终是原碎片，不随展示文本扩大：
    同一段证据支撑多条断言时，每条引用各有自己的锚点与引文，因此引用列表里
    同一个 ``evidence_id`` 可以出现多次。页码与证据段标识不因锚点改变，回源
    校验仍按「所标页逐字比对」进行。

    ``score`` 只有向量兜底路径有值：它是这条证据段的重排分数，也是它排在这个
    位置的原因；图路径的引用没有分数，顺序由关系类型决定。
    """

    evidence_id: str
    doc_id: str
    document: str | None
    page: int
    text: str
    score: float | None = None
    anchor: EvidenceAnchor | None = None

    def __post_init__(self) -> None:
        """锚点必须属于本引用指向的证据段，不能张冠李戴。"""
        if self.anchor is not None and self.anchor.evidence_id != self.evidence_id:
            raise ValueError(
                f"引用的锚点属于 {self.anchor.evidence_id}，"
                f"但引用指向 {self.evidence_id}"
            )


@dataclass(frozen=True, slots=True)
class Fact:
    """图上的一条结论：头概念沿某类关系指向尾概念，证据可回溯。

    ``path`` 是完整的图路径（至少一步；多跳时不止一步），``relation`` 是末跳的
    关系，``head`` / ``tail`` 是路径的起点与终点。``evidence_ids`` 汇总路径上
    每一步的证据，因此每条结论的每一跳都能回到引用层核对。
    """

    relation: str
    head: Concept
    tail: Concept
    confidence: float
    evidence_ids: tuple[str, ...]
    # 向量兜底的事实没有图路径，因此允许为空；图命中路径的结论一定非空。
    path: tuple[PathStep, ...] = ()

    @property
    def trail(self) -> tuple[Concept, ...]:
        """结论走过的概念链；没有图路径时退化为两端概念。"""
        if not self.path:
            return (self.head, self.tail)
        return (self.head, *(step.tail for step in self.path))

    def cited_spans(self) -> tuple[tuple[str, EvidenceAnchor | None], ...]:
        """这条结论要引用的 ``(证据段, 锚点)`` 序列，按路径逐条展开。

        一步就是一条断言，锚点挂在步上：两跳结论的不同跳可以引用同一段证据的
        不同句子，因此同一证据段可能出现多次。没有图路径的结论没有锚点，逐段
        整段引用。引用层的去重键是 ``(证据段, 锚点)``，不是证据段。
        """
        if not self.path:
            return tuple((evidence_id, None) for evidence_id in self.evidence_ids)
        return tuple(
            (evidence_id, step.anchor_for(evidence_id))
            for step in self.path
            for evidence_id in step.evidence_ids
        )


@dataclass(frozen=True, slots=True)
class SharedFact:
    """两侧都成立的同一结论：关系与尾概念相同，证据分别来自两侧的路径。

    两侧的路径各自独立（起点不同），因此不合并成一条 ``Fact``：把右侧的证据
    塞进左侧的路径会破坏「每条证据都在它所属的边上」这一可核对性。
    """

    relation: str
    tail: Concept
    left: tuple[Fact, ...]
    right: tuple[Fact, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """两侧路径的证据并集，按首次出现顺序去重。"""
        return tuple(
            dict.fromkeys(
                evidence_id
                for fact in (*self.left, *self.right)
                for evidence_id in fact.evidence_ids
            )
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """对比型回答：两个同级锚点，事实按 ``(关系, 尾概念)`` 分区。

    ``shared`` 是两侧都成立的结论，``only_left`` / ``only_right`` 只有一侧成立；
    用户问「怎么区分」要的是差异，不是两份各自的完整清单。``discriminating``
    单独放两侧之间的 ``confused_with`` 结论：它是文献自己说「这两个容易混」，
    与「我们据图谱归纳出的差异」不是一回事，不能混进 ``only_left``。
    """

    left: ConceptMatch
    right: ConceptMatch
    shared: tuple[SharedFact, ...]
    only_left: tuple[Fact, ...]
    only_right: tuple[Fact, ...]
    discriminating: tuple[Fact, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    """一次提问的结果。空答案也是显式的：``path`` 说明为什么没答。"""

    question: str
    path: AnswerPath
    matches: tuple[ConceptMatch, ...]
    focus: ConceptMatch | None
    facts: tuple[Fact, ...]
    citations: tuple[Citation, ...]
    # 空答案时说明缺在哪一步（未命中概念、概念不在图谱、没有带证据的事实）；
    # 截断时说明结果集撞到了哪个上界。
    detail: str | None = None
    # 结果集超过上界被截断；截断不是错误，但必须让调用方看得见。
    truncated: bool = False
    # 本次回答产生的模型调用（向量兜底的嵌入与重排）；图路径为空。
    model_calls: tuple[ModelCall, ...] = ()
    # 拒答原因；非拒答路径必须为 None。拒绝与原因一一对应，不靠调用方记得填。
    refusal_reason: RefusalReason | None = None
    # 对比型回答的分区；非对比问题为 None，且此时 ``facts`` 仍是单焦点形态。
    comparison: Comparison | None = None

    def __post_init__(self) -> None:
        """拒答必须说明原因，且不得携带任何结论或引用。"""
        refusing = self.path is AnswerPath.REFUSAL
        if refusing != (self.refusal_reason is not None):
            raise ValueError(
                "拒答路径必须且只能带拒答原因："
                f"{self.path.value} / {self.refusal_reason}"
            )
        if refusing and (self.facts or self.citations or self.comparison is not None):
            raise ValueError(
                "拒答答案不能包含结论或引用：拒答是「没有依据」，"
                "带结论或引用会让它变成低置信度答案"
            )


# 2：结论增加图路径、引用增加重排分数，答案增加模型调用记录与截断标记；
# 3：答案增加 refusal_reason（拒答原因）；
# 4：答案增加 comparison（对比型回答的分区），单焦点答案的字段不变；
# 5：引用增加 anchor（锚点区间），有锚点时 text 变为该区间原文；
# 6：短锚点的 text 并入同页相邻证据段作为展示上下文，anchor 仍是原区间。
# 字段只增不改，评估按版本号读。
ANSWER_SCHEMA_VERSION = 6


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    """答案的 JSON 形态；字段名即评估与接口层的契约，改动要升 schema 版本。"""
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "question": answer.question,
        "path": answer.path.value,
        "refusal_reason": (
            answer.refusal_reason.value
            if answer.refusal_reason is not None
            else None
        ),
        "detail": answer.detail,
        "truncated": answer.truncated,
        "focus": _match_to_dict(answer.focus),
        "matches": [_match_to_dict(match) for match in answer.matches],
        "comparison": _comparison_to_dict(answer.comparison),
        "facts": [_fact_to_dict(fact) for fact in answer.facts],
        "citations": [
            {
                "evidence_id": citation.evidence_id,
                "doc_id": citation.doc_id,
                "document": citation.document,
                "page": citation.page,
                "text": citation.text,
                "score": citation.score,
                "anchor": _anchor_to_dict(citation.anchor),
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


def _anchor_to_dict(anchor: EvidenceAnchor | None) -> dict[str, Any] | None:
    """引用的锚点：证据段内的字符区间；整段引用时为 ``None``（旧形态）。"""
    if anchor is None:
        return None
    return {
        "evidence_id": anchor.evidence_id,
        "start": anchor.start,
        "end": anchor.end,
        "ambiguous": anchor.ambiguous,
    }


def _comparison_to_dict(comparison: Comparison | None) -> dict[str, Any] | None:
    """对比型回答的分区；共同项保留两侧各自的结论与证据。"""
    if comparison is None:
        return None
    return {
        "left": _match_to_dict(comparison.left),
        "right": _match_to_dict(comparison.right),
        "shared": [
            {
                "relation": shared.relation,
                "tail": _concept_to_dict(shared.tail),
                "left": [_fact_to_dict(fact) for fact in shared.left],
                "right": [_fact_to_dict(fact) for fact in shared.right],
            }
            for shared in comparison.shared
        ],
        "only_left": [_fact_to_dict(fact) for fact in comparison.only_left],
        "only_right": [_fact_to_dict(fact) for fact in comparison.only_right],
        "discriminating": [
            _fact_to_dict(fact) for fact in comparison.discriminating
        ],
    }


def _fact_to_dict(fact: Fact) -> dict[str, Any]:
    return {
        "relation": fact.relation,
        "head": _concept_to_dict(fact.head),
        "tail": _concept_to_dict(fact.tail),
        "confidence": fact.confidence,
        "evidence_ids": list(fact.evidence_ids),
        "path": [_step_to_dict(step) for step in fact.path],
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


def _step_to_dict(step: PathStep) -> dict[str, Any]:
    return {
        "relation": step.relation,
        "head": _concept_to_dict(step.head),
        "tail": _concept_to_dict(step.tail),
        "confidence": step.confidence,
        "evidence_ids": list(step.evidence_ids),
    }

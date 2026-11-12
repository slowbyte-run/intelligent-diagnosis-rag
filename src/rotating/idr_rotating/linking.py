"""问题侧实体链接：用本体别名表在问句里找概念。

链接只用别名表，不调模型：同一句话永远得到同一组命中，命中结果能指出是哪条
别名、是整句相等还是问句里包含了它。这样「为什么答这个」在答案里可核对。

歧义不猜：一个别名可能指向多个规范实例（如「磨粒磨损」既可能是轴承的也可能
是齿面的），链接结果保留全部候选；选哪个当检索起点由门面按关系的定义域与
别名的具体程度决定。同一处提及的多个候选仍是歧义；问句里两处独立提及则交给
门面组织成对比（见 ``ask``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from idr_core.graph import Concept
from idr_rotating.aliases import AliasResolver, normalize_term
from idr_rotating.ontology import Ontology

# 本体里的规范实例在图中是共享节点，标识统一加这个前缀（见 graph_batch）。
CONCEPT_ID_PREFIX = "concept:"


class LinkKind(StrEnum):
    """命中方式：整句就是一个别名，或问句里包含了某个别名。"""

    EXACT = "exact"
    SUBSTRING = "substring"


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    """问题里的一个概念命中：链接到哪个节点、命中了哪条别名、怎么命中的。"""

    concept: Concept
    alias: str
    kind: LinkKind

    @property
    def specificity(self) -> tuple[bool, int]:
        """越具体越优先：整句相等优先于子串，长别名优先于短别名。"""
        return (self.kind is LinkKind.EXACT, len(self.alias))


@dataclass(frozen=True, slots=True)
class _Candidate:
    match: ConceptMatch
    position: int

    @property
    def end(self) -> int:
        return self.position + len(self.match.alias)


def link_question(question: str, ontology: Ontology) -> tuple[ConceptMatch, ...]:
    """找出问句里出现的全部本体概念，按在问句中的位置排序。

    同一个概念经多条别名命中时只保留最具体的一条（整句相等优先，其次别名更
    长）：答案是给不了「两条别名都命中」这种噪声的。
    """
    text = normalize_term(question)
    if not text:
        return ()

    best: dict[str, _Candidate] = {}
    for alias, entity in AliasResolver(ontology).entries():
        position = text.find(alias) if alias else -1
        if position < 0:
            continue
        candidate = _Candidate(
            match=ConceptMatch(
                concept=Concept(
                    id=f"{CONCEPT_ID_PREFIX}{entity.id}",
                    name=entity.label,
                    type=entity.type,
                ),
                alias=alias,
                kind=(
                    LinkKind.EXACT
                    if alias == text
                    else LinkKind.SUBSTRING
                ),
            ),
            position=position,
        )
        previous = best.get(candidate.match.concept.id)
        if (
            previous is None
            or candidate.match.specificity > previous.match.specificity
        ):
            best[candidate.match.concept.id] = candidate

    ordered = sorted(
        best.values(),
        key=lambda item: (
            item.position,
            -len(item.match.alias),
            item.match.concept.id,
        ),
    )
    return tuple(candidate.match for candidate in _without_nested_matches(ordered))


def _without_nested_matches(
    ordered: list[_Candidate],
) -> tuple[_Candidate, ...]:
    """去掉落在更长命中内部的短别名。

    「轴承外圈疲劳剥落」会同时包含「轴承」「轴」「疲劳」「剥落」这些别名；
    它们不是独立提及，报出来只会让「为什么命中这个概念」变得不可信。同一跨度
    上的不同概念不算嵌套（那是真正的歧义，要留给调用方判断）。
    """
    kept: list[_Candidate] = []
    for candidate in ordered:
        if any(_is_nested(candidate, other) for other in kept):
            continue
        kept.append(candidate)
    return tuple(kept)


def _is_nested(candidate: _Candidate, other: _Candidate) -> bool:
    if (candidate.position, candidate.end) == (other.position, other.end):
        return False
    return candidate.position >= other.position and candidate.end <= other.end

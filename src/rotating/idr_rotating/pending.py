"""待审队列：被本体拒绝的抽取不静默丢弃，而是按类型聚合成可审阅条目。

每条待审记录回答三个问题：什么越界了、出现了多少次、原文例句长什么样。
队列只描述事实，是否新增本体类型由人决定。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

MAX_EXAMPLES_PER_REVIEW = 3
MAX_NAMES_PER_REVIEW = 5
MAX_EXCERPT_CHARS = 160


class PendingKind(StrEnum):
    """越界对象是一实体还是关系。"""

    ENTITY = "entity"
    RELATION = "relation"


class RejectionReason(StrEnum):
    """抽取被拒绝的原因；前两项是本体边界，其余是数据错误。"""

    UNKNOWN_ENTITY_TYPE = "unknown_entity_type"
    UNKNOWN_RELATION_TYPE = "unknown_relation_type"
    MISSING_EVIDENCE = "missing_evidence"
    UNKNOWN_EVIDENCE = "unknown_evidence"
    UNKNOWN_RELATION_ENDPOINT = "unknown_relation_endpoint"
    DOMAIN_RANGE_MISMATCH = "domain_range_mismatch"


@dataclass(frozen=True, slots=True)
class PendingExample:
    """一条待审记录的原文例句及其出处。"""

    doc_id: str
    evidence_id: str
    text: str


@dataclass(frozen=True, slots=True)
class PendingObservation:
    """一次被拒绝的抽取；聚合前保留每次出现的明细。"""

    kind: PendingKind
    value: str
    reason: RejectionReason
    doc_id: str
    name: str
    examples: tuple[PendingExample, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingReview:
    """聚合后的待审条目：同一越界类型只出现一次，带频次与例句。"""

    kind: PendingKind
    value: str
    reason: RejectionReason
    frequency: int
    names: tuple[str, ...]
    examples: tuple[PendingExample, ...]


@dataclass(slots=True)
class _Accumulator:
    kind: PendingKind
    value: str
    reason: RejectionReason
    frequency: int = 0
    names: list[str] = field(default_factory=list)
    examples: list[PendingExample] = field(default_factory=list)


def source_excerpt(text: str, *, limit: int = MAX_EXCERPT_CHARS) -> str:
    """把原文压成一行例句；保留原文用词，只折叠空白并截断。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def example_for(
    doc_id: str, evidence_id: str, evidence_text: Mapping[str, str]
) -> PendingExample | None:
    """取证据段原文作为例句；证据段不存在时返回 None。"""
    text = evidence_text.get(evidence_id)
    if text is None:
        return None
    return PendingExample(
        doc_id=doc_id,
        evidence_id=evidence_id,
        text=source_excerpt(text),
    )


def aggregate_pending(
    observations: Iterable[PendingObservation],
) -> tuple[PendingReview, ...]:
    """按「对象类别 + 越界值 + 原因」聚合，保留首次出现顺序。"""
    groups: dict[tuple[PendingKind, str, RejectionReason], _Accumulator] = {}
    for observation in observations:
        key = (observation.kind, observation.value, observation.reason)
        accumulator = groups.get(key)
        if accumulator is None:
            accumulator = _Accumulator(
                kind=observation.kind,
                value=observation.value,
                reason=observation.reason,
            )
            groups[key] = accumulator
        accumulator.frequency += 1
        if (
            observation.name
            and observation.name not in accumulator.names
            and len(accumulator.names) < MAX_NAMES_PER_REVIEW
        ):
            accumulator.names.append(observation.name)
        for example in observation.examples:
            if (
                example not in accumulator.examples
                and len(accumulator.examples) < MAX_EXAMPLES_PER_REVIEW
            ):
                accumulator.examples.append(example)

    return tuple(
        PendingReview(
            kind=accumulator.kind,
            value=accumulator.value,
            reason=accumulator.reason,
            frequency=accumulator.frequency,
            names=tuple(accumulator.names),
            examples=tuple(accumulator.examples),
        )
        for accumulator in groups.values()
    )

"""抽取：模型输出经 Pydantic 解析、别名归一与本体校验后才允许入库。

- 解析失败有界重试；重试仍失败时抛出带尝试次数的 ``ExtractionError``，
  由入库门面记录原因并继续处理下一篇，不中断整批。
- 类型或关系越界不抛异常，而是进入待审队列；静默丢弃是禁止的。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from idr_core.llm import LLMClient
from idr_rotating.aliases import AliasResolver, normalize_term
from idr_rotating.ontology import Ontology
from idr_rotating.pending import (
    PendingExample,
    PendingKind,
    PendingObservation,
    RejectionReason,
    example_for,
)

DEFAULT_MAX_ATTEMPTS = 3


class ExtractionError(RuntimeError):
    """模型输出不是约定的 JSON 结构；``attempts`` 记录已尝试次数。"""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    head: str = Field(min_length=1)
    type: str = Field(min_length=1)
    tail: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class Extraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ValidatedEntity:
    """通过本体校验的实体。

    ``key`` 是概念身份：解析到规范实例时就是本体 ID，否则是「类型 + 归一标签」。
    概念节点 ID 由它推导，因此抽取顺序变化不会改变节点 ID。
    """

    name: str
    type: str
    evidence_ids: tuple[str, ...]
    confidence: float
    key: str


@dataclass(frozen=True, slots=True)
class ValidatedExtraction:
    entities: tuple[ValidatedEntity, ...]
    relations: tuple[ExtractedRelation, ...]
    rejected_entities: int
    rejected_relations: int
    pending: tuple[PendingObservation, ...] = ()


@dataclass(slots=True)
class _EntityAccumulator:
    name: str
    type: str
    evidence_ids: list[str]
    confidence: float
    key: str


@dataclass(slots=True)
class _RelationAccumulator:
    head: str
    type: str
    tail: str
    evidence_ids: list[str]
    confidence: float


def parse_extraction(response: str) -> Extraction:
    try:
        return Extraction.model_validate_json(response)
    except ValidationError as error:
        raise ExtractionError(f"模型输出不符合抽取契约：{error}") from None


def extract_with_retry(
    llm: LLMClient,
    prompt: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[Extraction, int]:
    """调用模型并在输出不合契约时重试，返回抽取结果与实际尝试次数。"""
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")

    last_error: ExtractionError | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = (
            prompt if last_error is None else _retry_prompt(prompt, last_error)
        )
        try:
            return parse_extraction(llm.complete(attempt_prompt)), attempt
        except ExtractionError as error:
            last_error = error

    raise ExtractionError(
        f"模型输出连续 {max_attempts} 次不符合抽取契约，"
        f"最后一次错误：{last_error}",
        attempts=max_attempts,
    )


def validate_extraction(
    extraction: Extraction,
    ontology: Ontology,
    *,
    doc_id: str,
    evidence_text: Mapping[str, str],
    aliases: AliasResolver | None = None,
) -> ValidatedExtraction:
    """归一别名并校验本体边界；越界项进入待审队列，合法项按概念去重。"""
    resolver = aliases if aliases is not None else AliasResolver(ontology)
    pending: list[PendingObservation] = []
    rejected_entities = 0
    rejected_relations = 0

    entities: dict[str, _EntityAccumulator] = {}
    raw_to_canonical: dict[str, set[str]] = {}

    for entity in extraction.entities:
        canonical_type = resolver.resolve_type(entity.type)
        if canonical_type is None:
            rejected_entities += 1
            pending.append(
                _observation(
                    kind=PendingKind.ENTITY,
                    value=entity.type,
                    reason=RejectionReason.TYPE_NOT_IN_ONTOLOGY,
                    doc_id=doc_id,
                    name=entity.name,
                    evidence_ids=entity.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        canonical = resolver.resolve_entity(
            entity.name, entity_type=canonical_type
        )
        name = canonical.label if canonical is not None else entity.name.strip()

        evidence_ids, reason = _valid_evidence(
            entity.evidence_ids, evidence_text
        )
        if reason is not None:
            rejected_entities += 1
            pending.append(
                _observation(
                    kind=PendingKind.ENTITY,
                    value=canonical_type,
                    reason=reason,
                    doc_id=doc_id,
                    name=entity.name.strip(),
                    evidence_ids=entity.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        key = (
            canonical.id
            if canonical is not None
            else f"{canonical_type}:{normalize_term(name)}"
        )
        _merge_entity(entities, key, name, canonical_type, evidence_ids, entity)
        raw_to_canonical.setdefault(entity.name.strip(), set()).add(name)
        raw_to_canonical.setdefault(name, set()).add(name)

    accepted_entities = tuple(
        ValidatedEntity(
            name=accumulator.name,
            type=accumulator.type,
            evidence_ids=tuple(accumulator.evidence_ids),
            confidence=accumulator.confidence,
            key=accumulator.key,
        )
        for accumulator in entities.values()
    )
    types_by_name: dict[str, set[str]] = {}
    for entity in accepted_entities:
        types_by_name.setdefault(entity.name, set()).add(entity.type)

    def resolve_endpoint(raw_name: str) -> tuple[str, str] | None:
        candidates = raw_to_canonical.get(raw_name.strip(), set())
        if not candidates:
            canonical = resolver.resolve_entity(raw_name)
            if canonical is not None:
                candidates = {canonical.label}
        if len(candidates) != 1:
            return None
        name = next(iter(candidates))
        types = types_by_name.get(name, set())
        if len(types) != 1:
            return None
        return name, next(iter(types))

    relations: dict[tuple[str, str, str], _RelationAccumulator] = {}
    for relation in extraction.relations:
        relation_type = resolver.resolve_relation_type(relation.type)
        if relation_type is None:
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation.type,
                    reason=RejectionReason.TYPE_NOT_IN_ONTOLOGY,
                    doc_id=doc_id,
                    name=_relation_name(relation),
                    evidence_ids=relation.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        head = resolve_endpoint(relation.head)
        tail = resolve_endpoint(relation.tail)
        if head is None or tail is None:
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation_type,
                    reason=RejectionReason.UNKNOWN_RELATION_ENDPOINT,
                    doc_id=doc_id,
                    name=_relation_name(relation),
                    evidence_ids=relation.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        definition = ontology.relation_types[relation_type]
        if head[1] not in definition.domain or tail[1] not in definition.range:
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation_type,
                    reason=RejectionReason.DOMAIN_RANGE_MISMATCH,
                    doc_id=doc_id,
                    name=_relation_name(relation),
                    evidence_ids=relation.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        evidence_ids, reason = _valid_evidence(
            relation.evidence_ids, evidence_text
        )
        if reason is not None:
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation_type,
                    reason=reason,
                    doc_id=doc_id,
                    name=_relation_name(relation),
                    evidence_ids=relation.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        key = (head[0], relation_type, tail[0])
        accumulator = relations.get(key)
        if accumulator is None:
            relations[key] = _RelationAccumulator(
                head=head[0],
                type=relation_type,
                tail=tail[0],
                evidence_ids=list(evidence_ids),
                confidence=relation.confidence,
            )
        else:
            _extend_unique(accumulator.evidence_ids, evidence_ids)
            accumulator.confidence = max(
                accumulator.confidence, relation.confidence
            )

    accepted_relations = tuple(
        ExtractedRelation(
            head=accumulator.head,
            type=accumulator.type,
            tail=accumulator.tail,
            evidence_ids=accumulator.evidence_ids,
            confidence=accumulator.confidence,
        )
        for accumulator in relations.values()
    )

    return ValidatedExtraction(
        entities=accepted_entities,
        relations=accepted_relations,
        rejected_entities=rejected_entities,
        rejected_relations=rejected_relations,
        pending=tuple(pending),
    )


def _merge_entity(
    entities: dict[str, _EntityAccumulator],
    key: str,
    name: str,
    canonical_type: str,
    evidence_ids: tuple[str, ...],
    entity: ExtractedEntity,
) -> None:
    accumulator = entities.get(key)
    if accumulator is None:
        entities[key] = _EntityAccumulator(
            name=name,
            type=canonical_type,
            evidence_ids=list(evidence_ids),
            confidence=entity.confidence,
            key=key,
        )
        return
    _extend_unique(accumulator.evidence_ids, evidence_ids)
    accumulator.confidence = max(accumulator.confidence, entity.confidence)


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _valid_evidence(
    evidence_ids: Sequence[str], evidence_text: Mapping[str, str]
) -> tuple[tuple[str, ...], RejectionReason | None]:
    if not evidence_ids:
        return (), RejectionReason.MISSING_EVIDENCE
    if any(evidence_id not in evidence_text for evidence_id in evidence_ids):
        return (), RejectionReason.UNKNOWN_EVIDENCE
    return tuple(dict.fromkeys(evidence_ids)), None


def _observation(
    *,
    kind: PendingKind,
    value: str,
    reason: RejectionReason,
    doc_id: str,
    name: str,
    evidence_ids: Sequence[str],
    evidence_text: Mapping[str, str],
) -> PendingObservation:
    return PendingObservation(
        kind=kind,
        value=value,
        reason=reason,
        doc_id=doc_id,
        name=name,
        examples=_examples(evidence_ids, doc_id, evidence_text),
    )


def _examples(
    evidence_ids: Sequence[str],
    doc_id: str,
    evidence_text: Mapping[str, str],
) -> tuple[PendingExample, ...]:
    examples: list[PendingExample] = []
    for evidence_id in dict.fromkeys(evidence_ids):
        example = example_for(doc_id, evidence_id, evidence_text)
        if example is not None:
            examples.append(example)
    return tuple(examples)


def _relation_name(relation: ExtractedRelation) -> str:
    return f"{relation.head} → {relation.tail}"


def _retry_prompt(prompt: str, error: ExtractionError) -> str:
    return (
        f"{prompt}\n\n"
        f"上一次输出无法解析：{error}\n"
        "请只输出符合上述格式的 JSON 对象，"
        "不要输出解释、Markdown 代码块或前后缀文字。"
    )

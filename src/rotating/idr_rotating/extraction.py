"""抽取：模型输出经 Pydantic 解析、别名归一与本体校验后才允许入库。

- 长文献按段数与字符数上限分批调用模型，每批单独校验引用，再按概念身份键
  合并实体与关系。
- 解析失败有界重试；重试仍失败时抛出带尝试次数的 ``ExtractionError``，
  由入库门面记录原因并继续处理下一篇，不中断整批。
- 类型或关系越界不抛异常，而是进入待审队列；静默丢弃是禁止的。
- 每条断言必须带一段支撑原文片段，且片段要在所引证据段里逐字出现；
  校验通过后固化成 ``(evidence_id, start, end)`` 锚点随出处一起入库。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from idr_core.anchors import EvidenceAnchor, locate_anchor
from idr_core.llm import LLMClient
from idr_core.models import EvidenceSegment
from idr_rotating.aliases import AliasResolver, CanonicalEntity, normalize_term
from idr_rotating.ontology import Ontology
from idr_rotating.pending import (
    PendingExample,
    PendingKind,
    PendingObservation,
    RejectionReason,
    example_for,
)

DEFAULT_MAX_ATTEMPTS = 3

# 单次模型调用可见的证据段上限，段数与字符数必须同时满足。
#
# 真实语料上单篇论文可达 600 多段、两万多 prompt token；整篇一次调用时模型
# 只能挤出十几个实体，概念间关系几乎为零。40 段按本语料约 1.2k–1.5k 字符，
# 单次调用仍能看到相邻段落，又不至于长到被模型忽略；6000 字符是长段落
# （专利常见）的兜底，防止少量长段再次把提示词撑大。
MAX_BATCH_SEGMENTS = 40
MAX_BATCH_CHARS = 6_000


class ExtractionError(RuntimeError):
    """模型输出不是约定的 JSON 结构；``attempts`` 记录已尝试次数。"""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


def batch_segments(
    segments: Sequence[EvidenceSegment],
    *,
    max_segments: int = MAX_BATCH_SEGMENTS,
    max_chars: int = MAX_BATCH_CHARS,
) -> tuple[tuple[EvidenceSegment, ...], ...]:
    """按段数与字符数双上限切分证据段，保持文档顺序。

    单个证据段超过字符上限时独占一批：切分证据段会改变它的标识与定位，
    而证据段是引用回溯的最小单位。这种批会超过字符上限，但不丢段，也不
    把一条引用拆成两条。
    """
    if max_segments < 1 or max_chars < 1:
        raise ValueError("批次上限必须是正整数")

    batches: list[tuple[EvidenceSegment, ...]] = []
    current: list[EvidenceSegment] = []
    current_chars = 0
    for segment in segments:
        size = len(segment.text)
        if current and (
            len(current) >= max_segments or current_chars + size > max_chars
        ):
            batches.append(tuple(current))
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += size
    if current:
        batches.append(tuple(current))
    return tuple(batches)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    # 支撑这条断言的原文片段，必须逐字摘自所引证据段。缺省为空串而不是
    # 必填：模型漏给片段是这一条断言的数据问题，不该让整篇文献判失败；
    # 校验阶段会把它送进待审队列。
    quote: str = ""
    # 本体为 fault_mode 声明的两个建模维度（ADR-0004）；其他类型可为空。
    location: str | None = None
    damage_form: str | None = None


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    head: str = Field(min_length=1)
    type: str = Field(min_length=1)
    tail: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    # 同实体：模型给出的支撑片段；校验通过后 ``anchors`` 固化它的区间。
    quote: str = ""
    anchors: tuple[EvidenceAnchor, ...] = ()


class Extraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ValidatedEntity:
    """通过本体校验的实体。

    ``key`` 是概念身份：解析到规范实例时就是本体 ID，否则是「类型 + 归一标签」。
    概念节点 ID 由它推导，因此抽取顺序变化不会改变节点 ID。

    ``canonical`` 标记 ``key`` 是否本体规范实例：只有规范实例允许跨文献共享
    一个概念节点，自由词的身份依赖具体文献。
    """

    name: str
    type: str
    evidence_ids: tuple[str, ...]
    confidence: float
    key: str
    canonical: bool = False
    attributes: dict[str, str] = field(default_factory=dict)
    # 每条被引用证据段至多一个锚点；没有锚点的证据段按整段引用。
    anchors: tuple[EvidenceAnchor, ...] = ()


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
    canonical: bool = False
    attributes: dict[str, str] = field(default_factory=dict)
    anchors: list[EvidenceAnchor] = field(default_factory=list)


@dataclass(slots=True)
class _RelationAccumulator:
    head: str
    type: str
    tail: str
    evidence_ids: list[str]
    confidence: float
    anchors: list[EvidenceAnchor] = field(default_factory=list)


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


def _resolve_anchor(
    quote: str, evidence_ids: Sequence[str], evidence_text: Mapping[str, str]
) -> tuple[EvidenceAnchor | None, bool]:
    """把模型给的支撑片段定位成锚点，并区分「没给」与「给错」。

    - 没给片段（空串）：断言本身可能没错，只是没有可核对的定位，保留它但不带
      锚点，引用时走查询期选句兜底。
    - 给了片段但对不上原文：这是编造的引用，必须拒收进待审。

    两者混为一谈会让模型在拿不准时干脆不下结论，知识因此丢失（实测关系产出
    从 7 条掉到 2 条）。
    """

    text = (quote or "").strip()
    if not text:
        return None, False
    anchor = locate_anchor(text, evidence_ids, evidence_text)
    return anchor, anchor is None


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
        if ontology.entity_types[canonical_type].internal:
            # 内部类型（如 evidence）由流水线维护，模型输出属于越界抽取；
            # 进待审保证可见，但不因此判整篇失败。
            rejected_entities += 1
            pending.append(
                _observation(
                    kind=PendingKind.ENTITY,
                    value=canonical_type,
                    reason=RejectionReason.INTERNAL_TYPE,
                    doc_id=doc_id,
                    name=entity.name.strip(),
                    evidence_ids=entity.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        candidates = resolver.lookup_entity(
            entity.name, entity_type=canonical_type
        )
        if len(candidates) > 1:
            rejected_entities += 1
            pending.append(
                _observation(
                    kind=PendingKind.ENTITY,
                    value=canonical_type,
                    reason=RejectionReason.AMBIGUOUS_ALIAS,
                    doc_id=doc_id,
                    name=entity.name.strip(),
                    evidence_ids=entity.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue
        canonical = candidates[0] if candidates else None
        name = canonical.label if canonical is not None else entity.name.strip()

        attributes, invalid_attribute = _validated_attributes(
            entity, canonical, canonical_type, ontology
        )
        if invalid_attribute is not None:
            attribute, raw_value = invalid_attribute
            rejected_entities += 1
            pending.append(
                _observation(
                    kind=PendingKind.ENTITY,
                    value=f"{canonical_type}.{attribute}",
                    reason=RejectionReason.ATTRIBUTE_NOT_IN_ONTOLOGY,
                    doc_id=doc_id,
                    name=f"{entity.name.strip()}（{raw_value}）",
                    evidence_ids=entity.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

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

        anchor, fabricated = _resolve_anchor(
            entity.quote, evidence_ids, evidence_text
        )
        if fabricated:
            # 模型给了片段但原文里找不到：这是编造的引用，不写图，进待审。
            rejected_entities += 1
            pending.append(
                _observation(
                    kind=PendingKind.ENTITY,
                    value=canonical_type,
                    reason=RejectionReason.ANCHOR_NOT_FOUND,
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
        _merge_entity(
            entities,
            key,
            name,
            canonical_type,
            evidence_ids,
            entity,
            attributes,
            (anchor,),
            canonical=canonical is not None,
        )
        raw_to_canonical.setdefault(entity.name.strip(), set()).add(name)
        raw_to_canonical.setdefault(name, set()).add(name)

    accepted_entities = tuple(
        ValidatedEntity(
            name=accumulator.name,
            type=accumulator.type,
            evidence_ids=tuple(accumulator.evidence_ids),
            confidence=accumulator.confidence,
            key=accumulator.key,
            canonical=accumulator.canonical,
            attributes=dict(accumulator.attributes),
            anchors=tuple(accumulator.anchors),
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
        if ontology.relation_types[relation_type].internal:
            # 同实体类型：内部关系由流水线的证据层维护，不能让模型直接产边。
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation_type,
                    reason=RejectionReason.INTERNAL_TYPE,
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

        anchor, fabricated = _resolve_anchor(
            relation.quote, evidence_ids, evidence_text
        )
        if fabricated:
            # 同实体：给了片段但对不上就是编造，拒收进待审。
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation_type,
                    reason=RejectionReason.ANCHOR_NOT_FOUND,
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
                anchors=[anchor] if anchor is not None else [],
            )
        else:
            _extend_unique(accumulator.evidence_ids, evidence_ids)
            _extend_anchors(accumulator.anchors, (anchor,))
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
            anchors=tuple(accumulator.anchors),
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
    attributes: Mapping[str, str],
    anchors: Sequence[EvidenceAnchor | None],
    *,
    canonical: bool,
) -> None:
    accumulator = entities.get(key)
    if accumulator is None:
        entities[key] = _EntityAccumulator(
            name=name,
            type=canonical_type,
            evidence_ids=list(evidence_ids),
            confidence=entity.confidence,
            key=key,
            canonical=canonical,
            attributes=dict(attributes),
            anchors=[anchor for anchor in anchors if anchor is not None],
        )
        return
    _extend_unique(accumulator.evidence_ids, evidence_ids)
    _extend_anchors(accumulator.anchors, anchors)
    accumulator.confidence = max(accumulator.confidence, entity.confidence)
    for attribute, value in attributes.items():
        accumulator.attributes.setdefault(attribute, value)


def merge_validated_extractions(
    parts: Sequence[ValidatedExtraction],
    *,
    doc_id: str,
    evidence_text: Mapping[str, str],
) -> ValidatedExtraction:
    """合并各批的校验结果：实体按身份键去重并累计证据，关系按三元组去重。

    每批已单独校验过引用；合并时把关系端点从批内写法归到首个规范名，再按
    全局概念集检查歧义：同一原名在不同批次落到不同类型时，关系进入待审
    队列，而不是被接到错误节点。端点完全缺失仍由写图前的数据错误检查拦下。
    """
    entities: dict[str, _EntityAccumulator] = {}
    # 关系保留批次内「名字 -> 身份键」的映射：同一概念在不同批次可能写成
    # 大小写或空白不同的形式，合并后必须归到首个规范名，否则写图时端点丢失。
    relations: list[tuple[ExtractedRelation, dict[str, str]]] = []
    pending: list[PendingObservation] = []
    rejected_entities = 0
    rejected_relations = 0

    for part in parts:
        rejected_entities += part.rejected_entities
        rejected_relations += part.rejected_relations
        pending.extend(part.pending)
        names_to_keys = {
            entity.name: entity.key for entity in part.entities
        }
        for entity in part.entities:
            _merge_validated_entity(entities, entity)
        relations.extend(
            (relation, names_to_keys) for relation in part.relations
        )

    types_by_name: dict[str, set[str]] = {}
    for entity in entities.values():
        types_by_name.setdefault(entity.name, set()).add(entity.type)

    merged_relations: dict[tuple[str, str, str], _RelationAccumulator] = {}
    for relation, names_to_keys in relations:
        head = _merged_endpoint_name(
            relation.head, names_to_keys, entities
        )
        tail = _merged_endpoint_name(
            relation.tail, names_to_keys, entities
        )
        ambiguous = _ambiguous_endpoint(
            head, types_by_name
        ) or _ambiguous_endpoint(tail, types_by_name)
        if ambiguous:
            rejected_relations += 1
            pending.append(
                _observation(
                    kind=PendingKind.RELATION,
                    value=relation.type,
                    reason=RejectionReason.UNKNOWN_RELATION_ENDPOINT,
                    doc_id=doc_id,
                    name=_relation_name(relation),
                    evidence_ids=relation.evidence_ids,
                    evidence_text=evidence_text,
                )
            )
            continue

        key = (head, relation.type, tail)
        accumulator = merged_relations.get(key)
        if accumulator is None:
            merged_relations[key] = _RelationAccumulator(
                head=head,
                type=relation.type,
                tail=tail,
                evidence_ids=list(relation.evidence_ids),
                confidence=relation.confidence,
                anchors=list(relation.anchors),
            )
        else:
            _extend_unique(accumulator.evidence_ids, relation.evidence_ids)
            _extend_anchors(accumulator.anchors, relation.anchors)
            accumulator.confidence = max(
                accumulator.confidence, relation.confidence
            )

    return ValidatedExtraction(
        entities=tuple(
            ValidatedEntity(
                name=accumulator.name,
                type=accumulator.type,
                evidence_ids=tuple(accumulator.evidence_ids),
                confidence=accumulator.confidence,
                key=accumulator.key,
                canonical=accumulator.canonical,
                attributes=dict(accumulator.attributes),
                anchors=tuple(accumulator.anchors),
            )
            for accumulator in entities.values()
        ),
        relations=tuple(
            ExtractedRelation(
                head=accumulator.head,
                type=accumulator.type,
                tail=accumulator.tail,
                evidence_ids=accumulator.evidence_ids,
                confidence=accumulator.confidence,
                anchors=tuple(accumulator.anchors),
            )
            for accumulator in merged_relations.values()
        ),
        rejected_entities=rejected_entities,
        rejected_relations=rejected_relations,
        pending=tuple(pending),
    )


def _merge_validated_entity(
    entities: dict[str, _EntityAccumulator], entity: ValidatedEntity
) -> None:
    accumulator = entities.get(entity.key)
    if accumulator is None:
        entities[entity.key] = _EntityAccumulator(
            name=entity.name,
            type=entity.type,
            evidence_ids=list(entity.evidence_ids),
            confidence=entity.confidence,
            key=entity.key,
            canonical=entity.canonical,
            attributes=dict(entity.attributes),
            anchors=list(entity.anchors),
        )
        return
    _extend_unique(accumulator.evidence_ids, entity.evidence_ids)
    _extend_anchors(accumulator.anchors, entity.anchors)
    accumulator.confidence = max(accumulator.confidence, entity.confidence)
    for attribute, value in entity.attributes.items():
        accumulator.attributes.setdefault(attribute, value)


def _merged_endpoint_name(
    name: str,
    names_to_keys: Mapping[str, str],
    entities: Mapping[str, _EntityAccumulator],
) -> str:
    """把批次内的概念名映射到合并后的规范名。

    批次内没有这个名字时原样返回，交给写图前的数据错误检查拦下，不在这里
    把关系悄悄改接到同名的另一个概念上。
    """
    key = names_to_keys.get(name)
    if key is None:
        return name
    accumulator = entities.get(key)
    return name if accumulator is None else accumulator.name


def _ambiguous_endpoint(
    name: str, types_by_name: Mapping[str, set[str]]
) -> bool:
    """同名概念在合并后落到多个类型时，关系端点无法唯一确定。"""
    return len(types_by_name.get(name, set())) > 1


def _validated_attributes(
    entity: ExtractedEntity,
    canonical: CanonicalEntity | None,
    canonical_type: str,
    ontology: Ontology,
) -> tuple[dict[str, str], tuple[str, str] | None]:
    """校验模型给出的属性值；返回 (规范属性, 第一个越界项)。

    越界项用 ``(属性名, 原始值)`` 表示，由调用方送进待审队列。已解析到规范
    实例时以本体定义的属性为准——规范实例的身份本身就包含这两个维度。
    """
    definition = ontology.entity_types[canonical_type]
    provided = {"location": entity.location, "damage_form": entity.damage_form}
    attributes: dict[str, str] = {}

    for attribute in definition.attributes:
        allowed = ontology.attribute_values(attribute)
        if not allowed:
            continue
        raw = provided.get(attribute)
        if raw is None:
            continue
        matched, value = _match_attribute_value(allowed, raw)
        if not matched:
            return {}, (attribute, raw)
        if value is not None:
            attributes[attribute] = value

    if canonical is not None:
        instance = ontology.instance(canonical_type, canonical.id)
        if instance is not None:
            for attribute in definition.attributes:
                value = getattr(instance, attribute, None)
                if value is not None:
                    attributes[attribute] = value

    return attributes, None


def _match_attribute_value(
    allowed: Mapping[str | None, str], raw: str
) -> tuple[bool, str | None]:
    """把标识或标签归一为允许取值；``null`` 表示「未限定」这一合法取值。"""
    needle = normalize_term(raw)
    if needle in ("", "null", "none"):
        return True, None
    for key, label in allowed.items():
        candidates = (label,) if key is None else (key, label)
        if any(normalize_term(candidate) == needle for candidate in candidates):
            return True, key
    return False, None


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _extend_anchors(
    target: list[EvidenceAnchor],
    anchors: Sequence[EvidenceAnchor | None],
) -> None:
    """每条证据段只留一个锚点；多条断言命中同一段时取首次出现的那个。

    没有锚点的断言（模型没给片段）照常放行，只是不加锚点。
    """
    seen = {anchor.evidence_id for anchor in target}
    for anchor in anchors:
        if anchor is None:
            continue
        if anchor.evidence_id not in seen:
            target.append(anchor)
            seen.add(anchor.evidence_id)


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

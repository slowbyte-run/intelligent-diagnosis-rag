"""抽取标注集与断言级评分。

标注集是领域资产（``extraction_annotations.json``）：一批取自真实语料的证据段，
每段手写「应该抽出哪些实体与关系」。评分只比对断言本身，不查图谱、不跑检索。

- 期望实体是 ``(name, type)``，期望关系是 ``(head, type, tail)``；每条断言记录
  它来自哪一段证据，漏抽时能定位到段落。
- 期望断言命中算召回；输出里不在标注中的断言算精度损失。
- 期望断言没命中时分成三类：完全没出现（漏抽）、同名不同类型（类型不符）、
  同类型但端点不同（端点不符）。三者分开计数。
- 格式错误（引用不存在的证据段、未知类型、别名未归一、端点缺失等）在加载期
  失败，而不是评出一组假数字。

评分按文献分组：抽取本来就是逐篇调用模型，跨文献的同名概念不应互相顶替。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from idr_core.llm import LLMClient
from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_rotating.aliases import AliasResolver, normalize_term
from idr_rotating.extraction import (
    ValidatedExtraction,
    parse_extraction,
    validate_extraction,
)
from idr_rotating.ontology import Ontology
from idr_rotating.prompts import build_extraction_prompt

ANNOTATION_SCHEMA_VERSION = 1
DEFAULT_ANNOTATIONS_PATH = (
    Path(__file__).resolve().with_name("extraction_annotations.json")
)

_EVIDENCE_ID_PATTERN = re.compile(
    r"^ev:([0-9a-f]{16}):([0-9]+):([0-9]+)-([0-9]+)$"
)
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "name", "description", "segments", "expected_entities", "expected_relations"}
)
_SEGMENT_FIELDS = frozenset({"id", "page", "text"})
_ENTITY_FIELDS = frozenset({"evidence_id", "name", "type"})
_RELATION_FIELDS = frozenset({"evidence_id", "head", "type", "tail"})

OutcomeStatus = Literal["matched", "missing", "type_mismatch", "endpoint_mismatch"]
AssertionKind = Literal["entity", "relation"]


class AnnotationSetError(ValueError):
    """标注集格式错误：让标注错误在加载阶段失败，而不是评出一组假数字。"""


@dataclass(frozen=True, slots=True)
class AnnotatedSegment:
    """一条冻结的证据段：标识、页码与原文。"""

    id: str
    doc_id: str
    page: int
    text: str
    charspan: CharSpan

    def to_evidence_segment(self) -> EvidenceSegment:
        return EvidenceSegment(
            id=self.id,
            doc_id=self.doc_id,
            page=self.page,
            bbox=BBox(0, 0, 0, 0),
            charspan=self.charspan,
            text=self.text,
        )


@dataclass(frozen=True, slots=True)
class ExpectedEntity:
    evidence_id: str
    name: str
    type: str


@dataclass(frozen=True, slots=True)
class ExpectedRelation:
    evidence_id: str
    head: str
    type: str
    tail: str


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    """一批冻结证据段及其期望断言。"""

    name: str
    description: str
    segments: tuple[AnnotatedSegment, ...]
    expected_entities: tuple[ExpectedEntity, ...]
    expected_relations: tuple[ExpectedRelation, ...]

    def documents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(segment.doc_id for segment in self.segments))

    def segments_by_document(self) -> dict[str, tuple[AnnotatedSegment, ...]]:
        grouped: dict[str, list[AnnotatedSegment]] = {}
        for segment in self.segments:
            grouped.setdefault(segment.doc_id, []).append(segment)
        return {doc_id: tuple(items) for doc_id, items in grouped.items()}

    def entities_for(self, doc_id: str) -> tuple[ExpectedEntity, ...]:
        return tuple(
            entity
            for entity in self.expected_entities
            if _document_of(entity.evidence_id) == doc_id
        )

    def relations_for(self, doc_id: str) -> tuple[ExpectedRelation, ...]:
        return tuple(
            relation
            for relation in self.expected_relations
            if _document_of(relation.evidence_id) == doc_id
        )

    def with_documents(self, doc_ids: Sequence[str]) -> AnnotationSet:
        """只保留指定文献的子集；用于控制真实调用成本。"""
        selected = tuple(dict.fromkeys(doc_ids))
        known = set(self.documents())
        missing = [doc_id for doc_id in selected if doc_id not in known]
        if missing:
            raise AnnotationSetError(f"标注集里没有这些文献：{missing}")
        keep = set(selected)
        return AnnotationSet(
            name=self.name,
            description=self.description,
            segments=tuple(
                segment for segment in self.segments if segment.doc_id in keep
            ),
            expected_entities=tuple(
                entity
                for entity in self.expected_entities
                if _document_of(entity.evidence_id) in keep
            ),
            expected_relations=tuple(
                relation
                for relation in self.expected_relations
                if _document_of(relation.evidence_id) in keep
            ),
        )


@dataclass(frozen=True, slots=True)
class AssertionOutcome:
    """一条期望断言的判定结果；``evidence_id`` 用于定位漏抽段落。"""

    kind: AssertionKind
    evidence_id: str
    label: str
    status: OutcomeStatus
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ExtraAssertion:
    """输出里有、标注里没有的断言，是精度损失。"""

    kind: AssertionKind
    doc_id: str
    label: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentScore:
    doc_id: str
    expected_entities: int
    expected_relations: int
    output_entities: int
    output_relations: int
    matched_entities: int
    matched_relations: int
    outcomes: tuple[AssertionOutcome, ...]
    extras: tuple[ExtraAssertion, ...]


@dataclass(frozen=True, slots=True)
class ExtractionScore:
    """断言级召回与精度，以及漏抽 / 类型不符 / 端点不符的分离计数。"""

    documents: tuple[DocumentScore, ...] = ()
    outcomes: tuple[AssertionOutcome, ...] = ()
    extras: tuple[ExtraAssertion, ...] = ()
    expected_entities: int = 0
    expected_relations: int = 0
    output_entities: int = 0
    output_relations: int = 0
    matched_entities: int = 0
    matched_relations: int = 0
    missing_entities: int = 0
    missing_relations: int = 0
    type_mismatch_entities: int = 0
    type_mismatch_relations: int = 0
    endpoint_mismatch_relations: int = 0
    extra_entities: int = 0
    extra_relations: int = 0

    @property
    def expected_assertions(self) -> int:
        return self.expected_entities + self.expected_relations

    @property
    def matched_assertions(self) -> int:
        return self.matched_entities + self.matched_relations

    @property
    def output_assertions(self) -> int:
        return self.output_entities + self.output_relations

    @property
    def recall(self) -> float | None:
        if self.expected_assertions == 0:
            return None
        return self.matched_assertions / self.expected_assertions

    @property
    def precision(self) -> float | None:
        if self.output_assertions == 0:
            return None
        return self.matched_assertions / self.output_assertions


def load_annotations(
    ontology: Ontology, path: Path | None = None
) -> AnnotationSet:
    """读取并校验标注集；任何结构错误都在这里失败，不进入抽取与评分。"""
    target = DEFAULT_ANNOTATIONS_PATH if path is None else Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as error:
        raise AnnotationSetError(f"无法读取标注集 {target}：{error}") from error
    except json.JSONDecodeError as error:
        raise AnnotationSetError(f"标注集不是合法 JSON：{error}") from error

    if not isinstance(payload, Mapping):
        raise AnnotationSetError("标注集顶层必须是 JSON 对象")
    unknown = set(payload) - _TOP_LEVEL_FIELDS
    if unknown:
        raise AnnotationSetError(f"标注集有未知字段：{sorted(unknown)}")
    version = payload.get("schema_version")
    if version != ANNOTATION_SCHEMA_VERSION:
        raise AnnotationSetError(
            f"标注集 schema_version 不支持：{version!r}（本工具支持 {ANNOTATION_SCHEMA_VERSION}）"
        )
    name = payload.get("name", "")
    description = payload.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise AnnotationSetError("标注集的 name 与 description 必须是字符串")

    segments = _segments(payload.get("segments"))
    segment_ids = {segment.id for segment in segments}

    resolver = AliasResolver(ontology)
    entities = _entities(
        payload.get("expected_entities"), segment_ids, ontology, resolver
    )
    relations = _relations(
        payload.get("expected_relations"),
        segment_ids,
        ontology,
        resolver,
    )
    if not entities and not relations:
        raise AnnotationSetError("标注集至少需要一条期望断言")

    annotated = {entity.evidence_id for entity in entities} | {
        relation.evidence_id for relation in relations
    }
    orphan = [
        segment.id for segment in segments if segment.id not in annotated
    ]
    if orphan:
        raise AnnotationSetError(
            f"这些证据段没有任何期望断言，标注不完整：{orphan}"
        )

    # 关系端点必须在同一段里被标注为实体；否则抽取输出里不可能出现该端点。
    types_by_segment: dict[str, dict[str, set[str]]] = {}
    for entity in entities:
        types_by_segment.setdefault(entity.evidence_id, {}).setdefault(
            normalize_term(entity.name), set()
        ).add(entity.type)
    for relation in relations:
        segment_types = types_by_segment.get(relation.evidence_id, {})
        for endpoint in (relation.head, relation.tail):
            if normalize_term(endpoint) not in segment_types:
                raise AnnotationSetError(
                    f"关系 {_relation_label(relation)} 的端点「{endpoint}」"
                    f"未在证据段 {relation.evidence_id} 的期望实体中出现"
                )
        head_types = segment_types[normalize_term(relation.head)]
        tail_types = segment_types[normalize_term(relation.tail)]
        if len(head_types) != 1 or len(tail_types) != 1:
            raise AnnotationSetError(
                f"关系 {_relation_label(relation)} 的端点在证据段内类型不唯一"
            )
        definition = ontology.relation_types[relation.type]
        if (
            next(iter(head_types)) not in definition.domain
            or next(iter(tail_types)) not in definition.range
        ):
            raise AnnotationSetError(
                f"关系 {_relation_label(relation)} 违反 domain/range："
                f"{next(iter(head_types))} -> {next(iter(tail_types))} 不在 "
                f"{definition.domain} -> {definition.range} 内"
            )

    # 同段内重复断言是标注错误，避免同一件事被计两次。
    seen_entities: set[tuple[str, str, str]] = set()
    for entity in entities:
        key = (entity.evidence_id, normalize_term(entity.name), entity.type)
        if key in seen_entities:
            raise AnnotationSetError(f"证据段 {entity.evidence_id} 有重复期望实体：{entity.name}")
        seen_entities.add(key)
    seen_relations: set[tuple[str, str, str, str]] = set()
    for relation in relations:
        key = (
            relation.evidence_id,
            normalize_term(relation.head),
            relation.type,
            normalize_term(relation.tail),
        )
        if key in seen_relations:
            raise AnnotationSetError(
                f"证据段 {relation.evidence_id} 有重复期望关系：{_relation_label(relation)}"
            )
        seen_relations.add(key)

    return AnnotationSet(
        name=name,
        description=description,
        segments=segments,
        expected_entities=entities,
        expected_relations=relations,
    )


def _segments(raw: object) -> tuple[AnnotatedSegment, ...]:
    if not isinstance(raw, list) or not raw:
        raise AnnotationSetError("标注集缺少非空的 segments 数组")
    segments: list[AnnotatedSegment] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise AnnotationSetError(f"第 {index + 1} 个证据段不是 JSON 对象")
        unknown = set(entry) - _SEGMENT_FIELDS
        if unknown:
            raise AnnotationSetError(f"第 {index + 1} 个证据段有未知字段：{sorted(unknown)}")
        segment_id = entry.get("id")
        page = entry.get("page")
        text = entry.get("text")
        if not isinstance(segment_id, str) or not segment_id:
            raise AnnotationSetError(f"第 {index + 1} 个证据段缺少非空 id")
        if segment_id in seen:
            raise AnnotationSetError(f"证据段 ID 重复：{segment_id}")
        seen.add(segment_id)
        match = _EVIDENCE_ID_PATTERN.fullmatch(segment_id)
        if match is None:
            raise AnnotationSetError(
                f"证据段 ID 格式不对（应为 ev:<16 位文献标识>:<页>:<区间>）：{segment_id}"
            )
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise AnnotationSetError(f"{segment_id} 的 page 必须是正整数")
        if page != int(match.group(2)):
            raise AnnotationSetError(
                f"{segment_id} 的 page={page} 与 ID 里的页码 {match.group(2)} 不一致"
            )
        if not isinstance(text, str) or not text.strip():
            raise AnnotationSetError(f"{segment_id} 的 text 不能为空")
        start, end = int(match.group(3)), int(match.group(4))
        if end - start != len(text):
            raise AnnotationSetError(
                f"{segment_id} 的 text 长度 {len(text)} 与字符区间 "
                f"[{start}, {end}) 不一致"
            )
        segments.append(
            AnnotatedSegment(
                id=segment_id,
                doc_id=str(match.group(1)),
                page=page,
                text=text,
                charspan=CharSpan(start, end),
            )
        )
    return tuple(segments)


def _entities(
    raw: object,
    segment_ids: set[str],
    ontology: Ontology,
    resolver: AliasResolver,
) -> tuple[ExpectedEntity, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise AnnotationSetError("expected_entities 必须是数组")
    entities: list[ExpectedEntity] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise AnnotationSetError(f"第 {index + 1} 条期望实体不是 JSON 对象")
        unknown = set(entry) - _ENTITY_FIELDS
        if unknown:
            raise AnnotationSetError(f"第 {index + 1} 条期望实体有未知字段：{sorted(unknown)}")
        evidence_id = entry.get("evidence_id")
        name = entry.get("name")
        type_id = entry.get("type")
        if not isinstance(evidence_id, str) or evidence_id not in segment_ids:
            raise AnnotationSetError(
                f"期望实体引用了不存在的证据段：{evidence_id!r}"
            )
        if not isinstance(name, str) or not name.strip():
            raise AnnotationSetError(f"第 {index + 1} 条期望实体缺少非空 name")
        if not isinstance(type_id, str) or not type_id:
            raise AnnotationSetError(f"期望实体「{name}」缺少非空 type")
        canonical_type = resolver.resolve_type(type_id)
        if canonical_type is None:
            raise AnnotationSetError(
                f"期望实体「{name}」的实体类型不在本体中：{type_id!r}"
            )
        if canonical_type != type_id:
            raise AnnotationSetError(
                f"期望实体「{name}」使用了实体类型别名 {type_id!r}，"
                f"应写规范类型 {canonical_type!r}"
            )
        if ontology.entity_types[type_id].internal:
            raise AnnotationSetError(
                f"期望实体「{name}」使用了内部类型 {type_id!r}，内部类型不可标注"
            )
        candidates = resolver.lookup_entity(name.strip(), entity_type=type_id)
        if len(candidates) > 1:
            raise AnnotationSetError(
                f"期望实体「{name}」在本体里是歧义别名，抽取会拒收；请写规范名"
            )
        if candidates and candidates[0].label != name.strip():
            raise AnnotationSetError(
                f"期望实体「{name}」是别名，流水线会归一为规范名 "
                f"「{candidates[0].label}」；标注必须写规范名"
            )
        entities.append(
            ExpectedEntity(
                evidence_id=evidence_id,
                name=name.strip(),
                type=type_id,
            )
        )
    return tuple(entities)


def _relations(
    raw: object,
    segment_ids: set[str],
    ontology: Ontology,
    resolver: AliasResolver,
) -> tuple[ExpectedRelation, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise AnnotationSetError("expected_relations 必须是数组")
    relations: list[ExpectedRelation] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise AnnotationSetError(f"第 {index + 1} 条期望关系不是 JSON 对象")
        unknown = set(entry) - _RELATION_FIELDS
        if unknown:
            raise AnnotationSetError(f"第 {index + 1} 条期望关系有未知字段：{sorted(unknown)}")
        evidence_id = entry.get("evidence_id")
        head = entry.get("head")
        type_id = entry.get("type")
        tail = entry.get("tail")
        if not isinstance(evidence_id, str) or evidence_id not in segment_ids:
            raise AnnotationSetError(
                f"期望关系引用了不存在的证据段：{evidence_id!r}"
            )
        if not isinstance(head, str) or not head.strip():
            raise AnnotationSetError(f"第 {index + 1} 条期望关系缺少非空 head")
        if not isinstance(tail, str) or not tail.strip():
            raise AnnotationSetError(f"第 {index + 1} 条期望关系缺少非空 tail")
        if not isinstance(type_id, str) or not type_id:
            raise AnnotationSetError(f"期望关系 {head} -> {tail} 缺少非空 type")
        canonical_type = resolver.resolve_relation_type(type_id)
        if canonical_type is None:
            raise AnnotationSetError(
                f"期望关系「{head} -> {tail}」的关系类型不在本体中：{type_id!r}"
            )
        if canonical_type != type_id:
            raise AnnotationSetError(
                f"期望关系「{head} -> {tail}」使用了关系类型别名 {type_id!r}，"
                f"应写规范类型 {canonical_type!r}"
            )
        if ontology.relation_types[type_id].internal:
            raise AnnotationSetError(
                f"期望关系「{head} -> {tail}」使用了内部类型 {type_id!r}，内部类型不可标注"
            )
        relations.append(
            ExpectedRelation(
                evidence_id=evidence_id,
                head=head.strip(),
                type=type_id,
                tail=tail.strip(),
            )
        )
    return tuple(relations)


def _document_of(evidence_id: str) -> str:
    match = _EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
    if match is None:
        raise AnnotationSetError(f"证据段 ID 格式不对：{evidence_id}")
    return match.group(1)


def _relation_label(relation: ExpectedRelation) -> str:
    return f"{relation.head} -[{relation.type}]-> {relation.tail}"


def extract_annotated(
    llm: LLMClient,
    ontology: Ontology,
    annotations: AnnotationSet,
    *,
    aliases: AliasResolver | None = None,
) -> dict[str, ValidatedExtraction]:
    """逐篇抽取标注集里的证据段；每篇一次模型调用，返回校验后的抽取。

    这是单次运行，不重复、不汇总分布。T4 的重复运行直接在本函数外面再跑 N
    次、把结果交给 :func:`score_extractions`，不需要改评分。
    """
    outputs: dict[str, ValidatedExtraction] = {}
    for doc_id, segments in annotations.segments_by_document().items():
        evidence = tuple(segment.to_evidence_segment() for segment in segments)
        prompt = build_extraction_prompt(ontology, evidence)
        outputs[doc_id] = validate_extraction(
            parse_extraction(llm.complete(prompt)),
            ontology,
            doc_id=doc_id,
            evidence_text={segment.id: segment.text for segment in evidence},
            aliases=aliases,
        )
    return outputs


def evaluate_extraction(
    llm: LLMClient,
    ontology: Ontology,
    annotations: AnnotationSet,
    *,
    aliases: AliasResolver | None = None,
) -> ExtractionScore:
    """跑一次抽取并评分；重复运行与分布属于 T4，不在本函数里。"""
    return score_extractions(
        annotations, extract_annotated(llm, ontology, annotations, aliases=aliases)
    )


def score_extractions(
    annotations: AnnotationSet,
    outputs: Mapping[str, ValidatedExtraction],
) -> ExtractionScore:
    """把逐篇抽取结果与标注比较，得到断言级召回、精度与三类未命中计数。"""
    expected_documents = set(annotations.documents())
    unknown = sorted(set(outputs) - expected_documents)
    if unknown:
        raise AnnotationSetError(f"输出里有未在标注集中的文献：{unknown}")

    documents: list[DocumentScore] = []
    for doc_id in annotations.documents():
        documents.append(
            _score_document(
                doc_id,
                annotations.entities_for(doc_id),
                annotations.relations_for(doc_id),
                outputs.get(doc_id),
            )
        )

    return ExtractionScore(
        documents=tuple(documents),
        outcomes=tuple(
            outcome for document in documents for outcome in document.outcomes
        ),
        extras=tuple(extra for document in documents for extra in document.extras),
        expected_entities=sum(document.expected_entities for document in documents),
        expected_relations=sum(document.expected_relations for document in documents),
        output_entities=sum(document.output_entities for document in documents),
        output_relations=sum(document.output_relations for document in documents),
        matched_entities=sum(document.matched_entities for document in documents),
        matched_relations=sum(document.matched_relations for document in documents),
        missing_entities=sum(
            1
            for outcome in _outcomes(documents, "entity", "missing")
        ),
        missing_relations=sum(
            1
            for outcome in _outcomes(documents, "relation", "missing")
        ),
        type_mismatch_entities=sum(
            1
            for outcome in _outcomes(documents, "entity", "type_mismatch")
        ),
        type_mismatch_relations=sum(
            1
            for outcome in _outcomes(documents, "relation", "type_mismatch")
        ),
        endpoint_mismatch_relations=sum(
            1
            for outcome in _outcomes(documents, "relation", "endpoint_mismatch")
        ),
        extra_entities=sum(
            1 for extra in _extras(documents, "entity")
        ),
        extra_relations=sum(
            1 for extra in _extras(documents, "relation")
        ),
    )


def _outcomes(
    documents: Sequence[DocumentScore],
    kind: AssertionKind,
    status: OutcomeStatus,
) -> tuple[AssertionOutcome, ...]:
    return tuple(
        outcome
        for document in documents
        for outcome in document.outcomes
        if outcome.kind == kind and outcome.status == status
    )


def _extras(
    documents: Sequence[DocumentScore], kind: AssertionKind
) -> tuple[ExtraAssertion, ...]:
    return tuple(
        extra
        for document in documents
        for extra in document.extras
        if extra.kind == kind
    )


def _score_document(
    doc_id: str,
    expected_entities: Sequence[ExpectedEntity],
    expected_relations: Sequence[ExpectedRelation],
    extraction: ValidatedExtraction | None,
) -> DocumentScore:
    output_entities = () if extraction is None else extraction.entities
    output_relations = () if extraction is None else extraction.relations

    expected_entity_keys: dict[tuple[str, str], str] = {}
    for entity in expected_entities:
        expected_entity_keys.setdefault(
            (normalize_term(entity.name), entity.type), entity.evidence_id
        )
    expected_relation_keys: dict[tuple[str, str, str], str] = {}
    for relation in expected_relations:
        expected_relation_keys.setdefault(
            (
                normalize_term(relation.head),
                relation.type,
                normalize_term(relation.tail),
            ),
            relation.evidence_id,
        )

    output_entity_types: dict[str, set[str]] = {}
    for entity in output_entities:
        output_entity_types.setdefault(normalize_term(entity.name), set()).add(
            entity.type
        )
    output_entity_keys = {
        (normalize_term(entity.name), entity.type)
        for entity in output_entities
    }

    output_relation_types: dict[tuple[str, str], set[str]] = {}
    for relation in output_relations:
        output_relation_types.setdefault(
            (normalize_term(relation.head), normalize_term(relation.tail)), set()
        ).add(relation.type)
    output_relation_keys = {
        (
            normalize_term(relation.head),
            relation.type,
            normalize_term(relation.tail),
        )
        for relation in output_relations
    }

    outcomes: list[AssertionOutcome] = []
    for key, evidence_id in expected_entity_keys.items():
        name, type_id = key
        status: OutcomeStatus
        detail: str | None = None
        if type_id in output_entity_types.get(name, set()):
            status = "matched"
        elif name in output_entity_types:
            status = "type_mismatch"
            detail = "输出类型：" + "、".join(
                sorted(output_entity_types[name])
            )
        else:
            status = "missing"
        outcomes.append(
            AssertionOutcome(
                kind="entity",
                evidence_id=evidence_id,
                label=_entity_label(name, type_id),
                status=status,
                detail=detail,
            )
        )

    for key, evidence_id in expected_relation_keys.items():
        head, type_id, tail = key
        status: OutcomeStatus = "missing"
        detail: str | None = None
        if type_id in output_relation_types.get((head, tail), set()):
            status = "matched"
        elif (head, tail) in output_relation_types:
            status = "type_mismatch"
            detail = "输出类型：" + "、".join(
                sorted(output_relation_types[(head, tail)])
            )
        else:
            same_type = [
                relation
                for relation in output_relations
                if relation.type == type_id
                and (
                    normalize_term(relation.head) == head
                    or normalize_term(relation.tail) == tail
                )
            ]
            if same_type:
                status = "endpoint_mismatch"
                detail = "；".join(
                    _relation_label(
                        ExpectedRelation(
                            evidence_id=evidence_id,
                            head=relation.head,
                            type=relation.type,
                            tail=relation.tail,
                        )
                    )
                    for relation in same_type
                )
        outcomes.append(
            AssertionOutcome(
                kind="relation",
                evidence_id=evidence_id,
                label=_entity_relation_label(head, type_id, tail),
                status=status,
                detail=detail,
            )
        )

    extras: list[ExtraAssertion] = []
    for entity in output_entities:
        key = (normalize_term(entity.name), entity.type)
        if key in expected_entity_keys:
            continue
        if any(
            extra.kind == "entity"
            and extra.label == _entity_label(*key)
            and extra.doc_id == doc_id
            for extra in extras
        ):
            continue
        extras.append(
            ExtraAssertion(
                kind="entity",
                doc_id=doc_id,
                label=_entity_label(*key),
                evidence_ids=tuple(entity.evidence_ids),
            )
        )
    for relation in output_relations:
        key = (
            normalize_term(relation.head),
            relation.type,
            normalize_term(relation.tail),
        )
        if key in expected_relation_keys:
            continue
        extras.append(
            ExtraAssertion(
                kind="relation",
                doc_id=doc_id,
                label=_entity_relation_label(*key),
                evidence_ids=tuple(relation.evidence_ids),
            )
        )

    matched_entities = sum(
        1
        for outcome in outcomes
        if outcome.kind == "entity" and outcome.status == "matched"
    )
    matched_relations = sum(
        1
        for outcome in outcomes
        if outcome.kind == "relation" and outcome.status == "matched"
    )
    return DocumentScore(
        doc_id=doc_id,
        expected_entities=len(expected_entity_keys),
        expected_relations=len(expected_relation_keys),
        output_entities=len(output_entity_keys),
        output_relations=len(output_relation_keys),
        matched_entities=matched_entities,
        matched_relations=matched_relations,
        outcomes=tuple(outcomes),
        extras=tuple(extras),
    )


def _entity_label(name: str, type_id: str) -> str:
    return f"{name}（{type_id}）"


def _entity_relation_label(head: str, type_id: str, tail: str) -> str:
    return f"{head} -[{type_id}]-> {tail}"

"""抽取：提示词由本体生成，模型输出经 Pydantic 解析后再过本体校验。

本票只要求「类型与关系命中本体才放行」。别名归一、越界类型待审队列与
有限重试留给 T4。
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from idr_core.models import EvidenceSegment
from idr_rotating.ontology import Ontology


class ExtractionError(RuntimeError):
    """模型输出不是约定的 JSON 结构。"""


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
class RejectedExtraction:
    """被本体校验拒绝的一条抽取，留作待审计的明细。

    ``reason`` 取值：``type_not_in_ontology``、``missing_evidence``、
    ``unknown_evidence``、``unknown_endpoint``、``domain_range_mismatch``。
    """

    kind: str
    label: str
    type: str
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidatedExtraction:
    entities: tuple[ExtractedEntity, ...]
    relations: tuple[ExtractedRelation, ...]
    rejected: tuple[RejectedExtraction, ...] = ()

    @property
    def rejected_entities(self) -> int:
        return sum(1 for item in self.rejected if item.kind == "entity")

    @property
    def rejected_relations(self) -> int:
        return sum(1 for item in self.rejected if item.kind == "relation")


def build_extraction_prompt(
    ontology: Ontology, segments: tuple[EvidenceSegment, ...]
) -> str:
    entity_lines = "\n".join(
        f"- {type_id}（{definition.label}）"
        for type_id, definition in ontology.entity_types.items()
    )
    relation_lines = "\n".join(
        f"- {type_id}（{definition.label}）："
        f"{'、'.join(definition.domain)} -> {'、'.join(definition.range)}"
        for type_id, definition in ontology.relation_types.items()
    )
    evidence_lines = "\n".join(
        f"[{segment.id}] {segment.text}" for segment in segments
    )
    return (
        "你是旋转设备故障诊断知识图谱的抽取器。只依据给出的证据段抽取，"
        "只输出一个 JSON 对象，不要输出解释或 Markdown。\n\n"
        "允许的实体类型：\n"
        f"{entity_lines}\n\n"
        "允许的关系类型：\n"
        f"{relation_lines}\n\n"
        "证据段（方括号内是证据段标识）：\n"
        f"{evidence_lines}\n\n"
        "输出格式：\n"
        '{"entities": [{"name": "实体名", "type": "实体类型", '
        '"evidence_ids": ["证据段标识"], "confidence": 0.0}], '
        '"relations": [{"head": "头实体名", "type": "关系类型", '
        '"tail": "尾实体名", "evidence_ids": ["证据段标识"], '
        '"confidence": 0.0}]}\n'
        "实体名使用原文术语；evidence_ids 只能引用上面出现过的标识；"
        "不得编造实体类型或关系类型。"
    )


def parse_extraction(response: str) -> Extraction:
    try:
        return Extraction.model_validate_json(response)
    except ValidationError as error:
        raise ExtractionError(f"模型输出不符合抽取契约：{error}") from None


def validate_extraction(
    extraction: Extraction,
    ontology: Ontology,
    *,
    available_evidence_ids: set[str],
) -> ValidatedExtraction:
    rejected: list[RejectedExtraction] = []

    def reject_entity(entity: ExtractedEntity, reason: str) -> None:
        rejected.append(
            RejectedExtraction(
                kind="entity",
                label=entity.name,
                type=entity.type,
                reason=reason,
                evidence_ids=tuple(entity.evidence_ids),
            )
        )

    def reject_relation(relation: ExtractedRelation, reason: str) -> None:
        rejected.append(
            RejectedExtraction(
                kind="relation",
                label=f"{relation.head} → {relation.tail}",
                type=relation.type,
                reason=reason,
                evidence_ids=tuple(relation.evidence_ids),
            )
        )

    accepted_entities: list[ExtractedEntity] = []
    for entity in extraction.entities:
        if entity.type not in ontology.entity_types:
            reject_entity(entity, "type_not_in_ontology")
            continue
        if not entity.evidence_ids:
            reject_entity(entity, "missing_evidence")
            continue
        if not set(entity.evidence_ids) <= available_evidence_ids:
            reject_entity(entity, "unknown_evidence")
            continue
        accepted_entities.append(entity)

    type_by_name: dict[str, str] = {}
    for entity in accepted_entities:
        type_by_name.setdefault(entity.name, entity.type)

    accepted_relations: list[ExtractedRelation] = []
    for relation in extraction.relations:
        definition = ontology.relation_types.get(relation.type)
        if definition is None:
            reject_relation(relation, "type_not_in_ontology")
            continue
        head_type = type_by_name.get(relation.head)
        tail_type = type_by_name.get(relation.tail)
        if head_type is None or tail_type is None:
            reject_relation(relation, "unknown_endpoint")
            continue
        if head_type not in definition.domain or tail_type not in definition.range:
            reject_relation(relation, "domain_range_mismatch")
            continue
        if not relation.evidence_ids:
            reject_relation(relation, "missing_evidence")
            continue
        if not set(relation.evidence_ids) <= available_evidence_ids:
            reject_relation(relation, "unknown_evidence")
            continue
        accepted_relations.append(relation)

    return ValidatedExtraction(
        entities=tuple(accepted_entities),
        relations=tuple(accepted_relations),
        rejected=tuple(rejected),
    )

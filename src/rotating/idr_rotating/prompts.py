"""抽取提示词：实体类型、关系类型、别名与取值约束全部由本体生成。

提示词里不允许出现第二份硬编码清单；改本体 YAML 就是改提示词。
"""

from __future__ import annotations

from idr_core.models import EvidenceSegment
from idr_rotating.ontology import EntityTypeDef, Ontology


def build_extraction_prompt(
    ontology: Ontology, segments: tuple[EvidenceSegment, ...]
) -> str:
    """按本体生成抽取提示词，并把证据段原文附在末尾供模型引用。"""
    evidence_lines = "\n".join(
        f"[{segment.id}] {segment.text}" for segment in segments
    )
    return (
        "你是一个受本体约束的知识图谱抽取器。只依据给出的证据段抽取，"
        "只输出一个 JSON 对象，不要输出解释或 Markdown。\n\n"
        f"本体：{ontology.domain} v{ontology.version}\n\n"
        "允许的实体类型（type 必须使用方括号内的标识）：\n"
        f"{_entity_type_lines(ontology)}\n\n"
        "规范概念（name 使用规范名；文献中的同义术语归一到规范名）：\n"
        f"{_instance_lines(ontology)}\n\n"
        "允许的关系类型（type 必须使用方括号内的标识；head/tail 必须满足 domain -> range）：\n"
        f"{_relation_lines(ontology)}\n\n"
        "属性取值（来自本体）：\n"
        f"{_attribute_value_lines(ontology)}\n\n"
        "证据段（方括号内是证据段标识）：\n"
        f"{evidence_lines}\n\n"
        "输出格式：\n"
        '{"entities": [{"name": "实体名", "type": "实体类型", '
        '"evidence_ids": ["证据段标识"], "confidence": 0.0, '
        '"location": "属性取值或 null", "damage_form": "属性取值或 null"}], '
        '"relations": [{"head": "头实体名", "type": "关系类型", '
        '"tail": "尾实体名", "evidence_ids": ["证据段标识"], '
        '"confidence": 0.0}]}\n'
        "规则：\n"
        "- name 使用文献原文术语；命中规范概念或别名时，必须使用规范名。\n"
        "- evidence_ids 只能引用上面出现过的证据段标识。\n"
        "- 仅当实体类型声明了 location/damage_form 时填写这两个字段，"
        "取值必须来自上面的属性取值清单；不确定时填 null。\n"
        "- 不得编造实体类型、关系类型或证据段标识。"
    )


def _entity_type_lines(ontology: Ontology) -> str:
    lines = [
        _definition_line(type_id, definition)
        for type_id, definition in ontology.entity_types.items()
    ]
    return "\n".join(lines) if lines else "（本体没有定义实体类型）"


def _definition_line(type_id: str, definition: EntityTypeDef) -> str:
    details: list[str] = []
    if definition.description:
        details.append(definition.description)
    if definition.aliases:
        details.append("别名：" + "、".join(definition.aliases))
    if definition.attributes:
        details.append("属性：" + "、".join(definition.attributes))
    suffix = f"：{'；'.join(details)}" if details else ""
    return f"- [{type_id}] {definition.label}{suffix}"


def _instance_lines(ontology: Ontology) -> str:
    lines: list[str] = []
    for type_id, instances in ontology.instances.items():
        if not instances:
            continue
        label = ontology.entity_types[type_id].label
        lines.append(f"【{label} {type_id}】")
        for instance in instances:
            aliases = (
                f"（别名：{'、'.join(instance.aliases)}）"
                if instance.aliases
                else ""
            )
            lines.append(f"- {instance.label} [{instance.id}]{aliases}")
    return "\n".join(lines) if lines else "（本体未列出规范实例）"


def _relation_lines(ontology: Ontology) -> str:
    lines: list[str] = []
    for relation_id, definition in ontology.relation_types.items():
        details: list[str] = []
        if definition.description:
            details.append(definition.description)
        details.append(
            f"{'、'.join(definition.domain)} -> {'、'.join(definition.range)}"
        )
        lines.append(
            f"- [{relation_id}] {definition.label}：{'；'.join(details)}"
        )
    return "\n".join(lines) if lines else "（本体没有定义关系类型）"


def _attribute_value_lines(ontology: Ontology) -> str:
    lines: list[str] = []
    for type_id, definition in ontology.entity_types.items():
        for attribute in definition.attributes:
            values = ontology.attribute_values(attribute)
            if not values:
                continue
            rendered = "、".join(
                f"{'null' if key is None else key}（{label}）"
                for key, label in values.items()
            )
            lines.append(f"- {type_id}.{attribute}：{rendered}")
    return "\n".join(lines) if lines else "（本体没有额外的取值约束）"


"""本体加载与格式校验。

本体 YAML 是实体类型、关系类型与别名的唯一事实源。格式错误在加载阶段立即失败，
不进入处理流程。
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1] / "ontology" / "rotating.yaml"
)


class OntologyError(RuntimeError):
    """本体文件缺失、不是合法 YAML，或结构不符合约定。"""


class StandardDef(BaseModel):
    """本体所锚定的外部标准；只记录标识与用途，不复制标准正文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    usage: str = ""
    modeling: str = ""


class EntityTypeDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    external_code: str | None = None


class RelationTypeDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    domain: list[str]
    range: list[str]
    description: str = ""


class InstanceDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    location: str | None = None
    damage_form: str | None = None
    external_code: str | None = None
    source: str | None = None


class Ontology(BaseModel):
    # 顶层只允许已知段落：拼错的键必须让本体在启动时失败，而不是被静默忽略。
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    domain: str
    updated: date | None = None
    standards: list[StandardDef] = Field(default_factory=list)
    entity_types: dict[str, EntityTypeDef]
    relation_types: dict[str, RelationTypeDef]
    locations: dict[str | None, str] = Field(default_factory=dict)
    damage_forms: dict[str, str] = Field(default_factory=dict)
    instances: dict[str, list[InstanceDef]] = Field(default_factory=dict)
    edge_properties: dict[str, str] = Field(default_factory=dict)
    pending: list[str] = Field(default_factory=list)

    def attribute_values(self, attribute: str) -> dict[str | None, str]:
        """某属性在本体里的允许取值：``标识 -> 标签``；未定义时为空。"""
        if attribute == "location":
            return dict(self.locations)
        if attribute == "damage_form":
            return dict(self.damage_forms)
        return {}

    def instance(self, entity_type: str, instance_id: str) -> InstanceDef | None:
        """按实体类型与实例 ID 取规范实例；不存在时返回 ``None``。"""
        for instance in self.instances.get(entity_type, ()):
            if instance.id == instance_id:
                return instance
        return None

    @model_validator(mode="after")
    def _check_references(self) -> Ontology:
        for relation_id, relation in self.relation_types.items():
            for entity_type in (*relation.domain, *relation.range):
                if entity_type not in self.entity_types:
                    raise ValueError(
                        f"关系 {relation_id} 引用了未定义的实体类型 {entity_type}"
                    )
        for entity_type in self.instances:
            if entity_type not in self.entity_types:
                raise ValueError(f"实例分组 {entity_type} 不是已定义的实体类型")
        return self


def load_ontology(path: Path | None = None) -> Ontology:
    target = DEFAULT_ONTOLOGY_PATH if path is None else path

    try:
        raw_text = target.read_text(encoding="utf-8")
    except OSError as error:
        raise OntologyError(f"无法读取本体文件 {target}：{error}") from error

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise OntologyError(f"本体文件不是合法 YAML：{error}") from error

    if not isinstance(data, dict):
        raise OntologyError("本体文件顶层必须是映射")

    try:
        return Ontology.model_validate(data)
    except ValidationError as error:
        details = "；".join(
            f"{'.'.join(str(part) for part in item['loc'])}：{item['msg']}"
            for item in error.errors()
        )
        raise OntologyError(f"本体格式错误：{details}") from None


def ontology_fingerprint(path: Path | None = None) -> str:
    """本体文件的内容指纹；写进报告即可确认两次运行是否用了同一份本体。"""
    target = DEFAULT_ONTOLOGY_PATH if path is None else path
    try:
        content = target.read_bytes()
    except OSError as error:
        raise OntologyError(f"无法读取本体文件 {target}：{error}") from error
    return hashlib.sha256(content).hexdigest()

"""本体加载与格式校验。

本体 YAML 是实体类型、关系类型与别名的唯一事实源。格式错误在加载阶段立即失败，
不进入处理流程。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1] / "ontology" / "rotating.yaml"
)


class OntologyError(RuntimeError):
    """本体文件缺失、不是合法 YAML，或结构不符合约定。"""


class EntityTypeDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    external_code: str | None = None


class RelationTypeDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str
    domain: list[str]
    range: list[str]
    description: str = ""


class InstanceDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    location: str | None = None
    damage_form: str | None = None
    external_code: str | None = None
    source: str | None = None


class Ontology(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    version: str
    domain: str
    entity_types: dict[str, EntityTypeDef]
    relation_types: dict[str, RelationTypeDef]
    locations: dict[str | None, str] = Field(default_factory=dict)
    damage_forms: dict[str, str] = Field(default_factory=dict)
    instances: dict[str, list[InstanceDef]] = Field(default_factory=dict)
    pending: list[str] = Field(default_factory=list)

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

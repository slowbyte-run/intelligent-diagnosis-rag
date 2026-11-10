"""别名归一：把文献里的同义术语收拢到本体定义的规范实例与规范类型。

归一表完全由本体 YAML 推导。同一个别名若在本体里指向多个实例（如「塑性变形」
既可能是轴承过载塑性变形，也可能是齿面塑性变形），本模块不做猜测，交由待审
队列或人工处理。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TypeVar

from idr_rotating.ontology import Ontology

_Value = TypeVar("_Value")


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    """本体中的一个规范实例。"""

    id: str
    label: str
    type: str


def normalize_term(term: str) -> str:
    """术语匹配键：统一全角半角、大小写，并去掉全部空白。"""
    normalized = unicodedata.normalize("NFKC", term).casefold()
    return "".join(normalized.split())


class AliasResolver:
    """本体驱动的术语解析器，同时服务实体、实体类型与关系类型。"""

    def __init__(self, ontology: Ontology) -> None:
        self._entity_type_ids = frozenset(ontology.entity_types)
        self._relation_type_ids = frozenset(ontology.relation_types)
        self._entities: dict[str, tuple[CanonicalEntity, ...]] = {}
        self._entity_types: dict[str, tuple[str, ...]] = {}
        self._relation_types: dict[str, tuple[str, ...]] = {}

        for type_id, instances in ontology.instances.items():
            for instance in instances:
                canonical = CanonicalEntity(
                    id=instance.id, label=instance.label, type=type_id
                )
                for term in (instance.id, instance.label, *instance.aliases):
                    _add(self._entities, normalize_term(term), canonical)

        for type_id, definition in ontology.entity_types.items():
            for term in (type_id, definition.label, *definition.aliases):
                _add(self._entity_types, normalize_term(term), type_id)

        for relation_id, relation in ontology.relation_types.items():
            for term in (relation_id, relation.label):
                _add(self._relation_types, normalize_term(term), relation_id)

    def resolve_entity(
        self, name: str, *, entity_type: str | None = None
    ) -> CanonicalEntity | None:
        """把实体名解析为规范实例；名字未知或别名有歧义时返回 None。"""
        candidates = self._entities.get(normalize_term(name), ())
        if entity_type is not None:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.type == entity_type
            )
        return candidates[0] if len(candidates) == 1 else None

    def resolve_type(self, name: str) -> str | None:
        """把实体类型名或别名解析为类型标识。"""
        if name in self._entity_type_ids:
            return name
        candidates = self._entity_types.get(normalize_term(name), ())
        return candidates[0] if len(candidates) == 1 else None

    def resolve_relation_type(self, name: str) -> str | None:
        """把关系类型名或标签解析为关系标识。"""
        if name in self._relation_type_ids:
            return name
        candidates = self._relation_types.get(normalize_term(name), ())
        return candidates[0] if len(candidates) == 1 else None


def _add(index: dict[str, tuple[_Value, ...]], key: str, value: _Value) -> None:
    existing = index.get(key, ())
    if value not in existing:
        index[key] = (*existing, value)

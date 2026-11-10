"""别名归一的公开契约：同义术语落到本体里的规范实例与规范类型。"""

from __future__ import annotations

from idr_rotating.aliases import AliasResolver, CanonicalEntity
from idr_rotating.ontology import load_ontology


def test_alias_and_canonical_label_resolve_to_the_same_instance() -> None:
    resolver = AliasResolver(load_ontology())

    canonical = CanonicalEntity(
        id="bearing_outer_race_fatigue_spalling",
        label="轴承外圈疲劳剥落",
        type="fault_mode",
    )

    assert resolver.resolve_entity("外套圈剥落") == canonical
    assert resolver.resolve_entity("外圈剥落") == canonical
    assert resolver.resolve_entity("轴承外圈疲劳剥落") == canonical
    assert resolver.resolve_entity("轴承外圈疲劳剥落") == resolver.resolve_entity(
        "外圈剥落"
    )


def test_matching_ignores_case_width_and_whitespace() -> None:
    resolver = AliasResolver(load_ontology())

    by_alias = resolver.resolve_entity("BPFO 倍频")
    assert by_alias is not None
    assert by_alias.label == "外圈故障频率及其倍频"
    assert resolver.resolve_entity("ｂｐｆｏ倍频") == by_alias
    assert resolver.resolve_entity("BPFO   倍频") == by_alias


def test_entity_type_aliases_resolve_to_the_canonical_type_id() -> None:
    resolver = AliasResolver(load_ontology())

    assert resolver.resolve_type("故障模式") == "fault_mode"
    assert resolver.resolve_type("fault_mode") == "fault_mode"
    assert resolver.resolve_type("不存在的类型") is None


def test_relation_labels_resolve_to_the_canonical_relation_id() -> None:
    resolver = AliasResolver(load_ontology())

    assert resolver.resolve_relation_type("表现为") == "indicated_by"
    assert resolver.resolve_relation_type("indicated_by") == "indicated_by"
    assert resolver.resolve_relation_type("magic_link") is None


def test_an_alias_shared_by_two_instances_is_not_guessed() -> None:
    resolver = AliasResolver(load_ontology())

    # 本体里「塑性变形」同时是轴承过载塑性变形与齿面塑性变形的别名。
    assert resolver.resolve_entity("塑性变形") is None
    assert resolver.resolve_entity("塑性变形", entity_type="fault_mode") is None


def test_unknown_entity_stays_unresolved() -> None:
    resolver = AliasResolver(load_ontology())

    assert resolver.resolve_entity("一种本体里没有的新征兆") is None

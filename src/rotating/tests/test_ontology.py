"""本体加载的公开契约：路径可注入、格式错误立即失败。"""

from __future__ import annotations

from pathlib import Path

import pytest

from idr_rotating.ontology import OntologyError, load_ontology


def test_default_ontology_exposes_entity_and_relation_types() -> None:
    ontology = load_ontology()

    assert ontology.entity_types["fault_mode"].label == "故障模式"
    assert ontology.relation_types["indicated_by"].domain == ["fault_mode"]
    assert ontology.relation_types["indicated_by"].range == ["symptom"]
    assert ontology.locations[None] == "未限定（跨多个部位或不区分部位）"


def test_malformed_ontology_fails_during_load(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(
        """
version: 0.1.0
domain: rotating
entity_types:
  fault_mode:
    description: 缺少 label
relation_types: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(OntologyError, match="本体格式错误"):
        load_ontology(path)


def test_relation_pointing_at_unknown_entity_type_fails_during_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken-reference.yaml"
    path.write_text(
        """
version: 0.1.0
domain: rotating
entity_types:
  fault_mode:
    label: 故障模式
relation_types:
  indicated_by:
    label: 表现为
    domain: [fault_mode]
    range: [symptom]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(OntologyError, match="未定义的实体类型"):
        load_ontology(path)


def test_unknown_definition_field_fails_during_load(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text(
        """
version: 0.1.0
domain: rotating
entity_types:
  fault_mode:
    label: 故障模式
    alias: [故障形式]
relation_types: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(OntologyError, match="本体格式错误"):
        load_ontology(path)


def test_unknown_top_level_key_fails_during_load(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text(
        """
version: 0.1.0
domain: rotating
entity_types: {}
relation_types: {}
location: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(OntologyError, match="本体格式错误"):
        load_ontology(path)


def test_known_top_level_sections_are_all_parsed() -> None:
    ontology = load_ontology()

    # 这些段落以前被 extra="ignore" 静默吞掉；现在必须真正解析。
    assert ontology.updated is not None
    assert ontology.standards
    assert ontology.edge_properties["confidence"] == "置信度，0–1，抽取时的模型自评"

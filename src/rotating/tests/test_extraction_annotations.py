"""抽取标注集的加载与校验：格式错误在加载期失败，不进入评分。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idr_rotating.extraction_evaluation import (
    AnnotationSetError,
    load_annotations,
)
from idr_rotating.ontology import load_ontology

SEGMENT_ID = "ev:aaaaaaaaaaaaaaaa:2:10-20"
SEGMENT_TEXT = "外圈剥落表现为振动。"


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": "测试标注集",
        "segments": [
            {"id": SEGMENT_ID, "page": 2, "text": SEGMENT_TEXT},
        ],
        "expected_entities": [
            {
                "evidence_id": SEGMENT_ID,
                "name": "轴承外圈疲劳剥落",
                "type": "fault_mode",
            }
        ],
        "expected_relations": [],
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_valid_set_loads_with_segment_and_expectation_binding(
    tmp_path: Path,
) -> None:
    annotations = load_annotations(load_ontology(), _write(tmp_path, _payload()))

    assert len(annotations.segments) == 1
    segment = annotations.segments[0]
    assert segment.id == SEGMENT_ID
    assert segment.doc_id == "aaaaaaaaaaaaaaaa"
    assert segment.page == 2
    assert segment.text == SEGMENT_TEXT
    assert annotations.expected_entities[0].evidence_id == SEGMENT_ID
    assert segment.to_evidence_segment().charspan.start == 10
    assert segment.to_evidence_segment().charspan.end == 20


def test_referencing_a_segment_that_does_not_exist_fails_at_load_time(
    tmp_path: Path,
) -> None:
    payload = _payload(
        expected_entities=[
            {
                "evidence_id": "ev:bbbbbbbbbbbbbbbb:1:0-4",
                "name": "轴承外圈疲劳剥落",
                "type": "fault_mode",
            }
        ]
    )

    with pytest.raises(AnnotationSetError, match="不存在的证据段"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_an_unknown_entity_type_fails_at_load_time(tmp_path: Path) -> None:
    payload = _payload(
        expected_entities=[
            {"evidence_id": SEGMENT_ID, "name": "外圈剥落", "type": "fault"}
        ]
    )

    with pytest.raises(AnnotationSetError, match="实体类型"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_an_internal_entity_type_is_not_annotatable(tmp_path: Path) -> None:
    payload = _payload(
        expected_entities=[
            {"evidence_id": SEGMENT_ID, "name": "证据段", "type": "evidence"}
        ]
    )

    with pytest.raises(AnnotationSetError, match="内部类型"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_an_alias_that_the_pipeline_would_canonicalize_is_rejected(
    tmp_path: Path,
) -> None:
    """标注写成别名时，抽取输出会被归一成规范名，分数必然错；加载期就要拦下。"""
    payload = _payload(
        expected_entities=[
            {"evidence_id": SEGMENT_ID, "name": "外圈剥落", "type": "fault_mode"}
        ]
    )

    with pytest.raises(AnnotationSetError, match="规范名"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_a_relation_whose_endpoints_are_not_annotated_entities_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _payload(
        expected_entities=[
            {
                "evidence_id": SEGMENT_ID,
                "name": "轴承外圈疲劳剥落",
                "type": "fault_mode",
            }
        ],
        expected_relations=[
            {
                "evidence_id": SEGMENT_ID,
                "head": "轴承外圈疲劳剥落",
                "type": "indicated_by",
                "tail": "振动",
            }
        ],
    )

    with pytest.raises(AnnotationSetError, match="端点"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_a_relation_that_violates_domain_range_is_rejected(tmp_path: Path) -> None:
    payload = _payload(
        expected_entities=[
            {
                "evidence_id": SEGMENT_ID,
                "name": "轴承外圈疲劳剥落",
                "type": "fault_mode",
            },
            {"evidence_id": SEGMENT_ID, "name": "振动", "type": "symptom"},
        ],
        expected_relations=[
            {
                "evidence_id": SEGMENT_ID,
                "head": "振动",
                "type": "has_fault_mode",
                "tail": "轴承外圈疲劳剥落",
            }
        ],
    )

    with pytest.raises(AnnotationSetError, match="domain"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_segment_text_must_match_its_charspan(tmp_path: Path) -> None:
    payload = _payload(
        segments=[{"id": SEGMENT_ID, "page": 2, "text": "短"}],
    )

    with pytest.raises(AnnotationSetError, match="字符区间"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def test_a_segment_without_any_expectation_is_rejected(tmp_path: Path) -> None:
    other_id = "ev:aaaaaaaaaaaaaaaa:3:0-4"
    payload = _payload(
        segments=[
            {"id": SEGMENT_ID, "page": 2, "text": SEGMENT_TEXT},
            {"id": other_id, "page": 3, "text": "振动信号"},
        ],
        expected_entities=[
            {
                "evidence_id": other_id,
                "name": "振动信号",
                "type": "symptom",
            }
        ],
    )

    with pytest.raises(AnnotationSetError, match="没有任何期望断言"):
        load_annotations(load_ontology(), _write(tmp_path, payload))


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def test_the_bundled_set_is_real_corpus_with_the_main_types() -> None:
    ontology = load_ontology()
    annotations = load_annotations(ontology)

    assert len(annotations.segments) >= 20
    # 中英文两种版面都要有，否则只是测了一种文字质量。
    documents = {segment.doc_id for segment in annotations.segments}
    assert len(documents) >= 4
    assert any(_has_chinese(segment.text) for segment in annotations.segments)
    assert any(not _has_chinese(segment.text) for segment in annotations.segments)

    entity_types = {entity.type for entity in annotations.expected_entities}
    assert {
        "equipment_class",
        "component",
        "fault_mode",
        "fault_cause",
        "symptom",
        "detection_method",
    } <= entity_types
    relation_types = {relation.type for relation in annotations.expected_relations}
    assert {"has_component", "has_fault_mode", "has_cause", "indicated_by", "detected_by"} <= relation_types

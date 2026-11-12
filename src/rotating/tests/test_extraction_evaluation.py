"""抽取评估的门面测试：假模型回放固定响应，断言召回、精度与分离计数。

不碰网络；真实模型的一次有界运行由命令行层负责。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idr_rotating.extraction import (
    ExtractedRelation,
    ValidatedEntity,
    ValidatedExtraction,
)
from idr_rotating.extraction_evaluation import (
    AnnotationSetError,
    evaluate_extraction,
    extract_annotated,
    load_annotations,
    score_extractions,
)
from idr_rotating.ontology import load_ontology

DOC_A = "aaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbb"
SEG_A = f"ev:{DOC_A}:1:0-10"
TEXT_A = "外圈剥落表现为振动。"
SEG_B = f"ev:{DOC_B}:2:0-5"
TEXT_B = "齿面点蚀。"


def _payload(
    *,
    expected_entities: list[dict[str, str]],
    expected_relations: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    relations = expected_relations or []
    used = {
        entry["evidence_id"]
        for entry in [*expected_entities, *relations]
    }
    known = {
        SEG_A: {"id": SEG_A, "page": 1, "text": TEXT_A},
        SEG_B: {"id": SEG_B, "page": 2, "text": TEXT_B},
    }
    return {
        "schema_version": 1,
        "segments": [known[evidence_id] for evidence_id in known if evidence_id in used],
        "expected_entities": expected_entities,
        "expected_relations": relations,
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _default_payload() -> dict[str, object]:
    return _payload(
        expected_entities=[
            {
                "evidence_id": SEG_A,
                "name": "轴承外圈疲劳剥落",
                "type": "fault_mode",
            },
            {"evidence_id": SEG_A, "name": "振动", "type": "symptom"},
            {"evidence_id": SEG_B, "name": "齿轮副", "type": "component"},
            {"evidence_id": SEG_B, "name": "齿面点蚀", "type": "fault_mode"},
            {"evidence_id": SEG_B, "name": "振动信号", "type": "symptom"},
        ],
        expected_relations=[
            {
                "evidence_id": SEG_A,
                "head": "轴承外圈疲劳剥落",
                "type": "indicated_by",
                "tail": "振动",
            },
            {
                "evidence_id": SEG_B,
                "head": "齿轮副",
                "type": "has_fault_mode",
                "tail": "齿面点蚀",
            },
            {
                "evidence_id": SEG_B,
                "head": "齿面点蚀",
                "type": "indicated_by",
                "tail": "振动信号",
            },
        ],
    )


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _response(
    entities: list[dict[str, object]],
    relations: list[dict[str, object]] | None = None,
) -> str:
    return json.dumps(
        {"entities": entities, "relations": relations or []}, ensure_ascii=False
    )


def _entity(name: str, type_id: str, evidence_id: str) -> dict[str, object]:
    return {
        "name": name,
        "type": type_id,
        "evidence_ids": [evidence_id],
        "confidence": 0.9,
    }


def _relation(
    head: str, type_id: str, tail: str, evidence_id: str
) -> dict[str, object]:
    return {
        "head": head,
        "type": type_id,
        "tail": tail,
        "evidence_ids": [evidence_id],
        "confidence": 0.8,
    }


def _score(
    tmp_path: Path,
    payload: dict[str, object],
    responses: list[str],
):
    ontology = load_ontology()
    annotations = load_annotations(ontology, _write(tmp_path, payload))
    return evaluate_extraction(FakeLLM(responses), ontology, annotations)


def test_a_perfect_run_reports_full_recall_and_precision(tmp_path: Path) -> None:
    response_a = _response(
        [
            # 输出写成别名也没关系：评估比的是校验、归一之后的断言。
            _entity("外圈剥落", "fault_mode", SEG_A),
            _entity("振动", "symptom", SEG_A),
        ],
        [_relation("外圈剥落", "indicated_by", "振动", SEG_A)],
    )
    response_b = _response(
        [
            _entity("齿轮副", "component", SEG_B),
            _entity("齿面点蚀", "fault_mode", SEG_B),
            _entity("振动信号", "symptom", SEG_B),
        ],
        [
            _relation("齿轮副", "has_fault_mode", "齿面点蚀", SEG_B),
            _relation("齿面点蚀", "indicated_by", "振动信号", SEG_B),
        ],
    )

    score = _score(tmp_path, _default_payload(), [response_a, response_b])

    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.matched_entities == 5
    assert score.matched_relations == 3
    assert score.missing_entities == 0
    assert score.missing_relations == 0
    assert all(outcome.status == "matched" for outcome in score.outcomes)


def test_missing_and_extra_assertions_are_counted_separately(tmp_path: Path) -> None:
    response_a = _response(
        [
            _entity("轴承外圈疲劳剥落", "fault_mode", SEG_A),
            _entity("振动", "symptom", SEG_A),
        ],
        [_relation("轴承外圈疲劳剥落", "indicated_by", "振动", SEG_A)],
    )
    response_b = _response(
        [
            _entity("齿轮副", "component", SEG_B),
            _entity("齿面点蚀", "fault_mode", SEG_B),
            # 期望的「振动信号」没抽到，却多抽了一个「异常噪声」。
            _entity("异常噪声", "symptom", SEG_B),
        ],
        [
            _relation("齿轮副", "has_fault_mode", "齿面点蚀", SEG_B),
            # 类型对、头端对、尾端错：算端点不符，不算漏抽。
            _relation("齿面点蚀", "indicated_by", "异常噪声", SEG_B),
        ],
    )

    score = _score(tmp_path, _default_payload(), [response_a, response_b])

    assert score.expected_entities == 5
    assert score.matched_entities == 4
    assert score.missing_entities == 1
    assert score.extra_entities == 1
    assert score.expected_relations == 3
    assert score.matched_relations == 2
    assert score.endpoint_mismatch_relations == 1
    assert score.missing_relations == 0
    assert score.extra_relations == 1
    missing = next(
        outcome
        for outcome in score.outcomes
        if outcome.status == "missing"
    )
    assert missing.evidence_id == SEG_B
    assert score.recall == pytest.approx(6 / 8)
    assert score.precision == pytest.approx(6 / 8)


def test_a_wrong_type_is_counted_as_a_mismatch_not_a_match(tmp_path: Path) -> None:
    payload = _payload(
        expected_entities=[
            {"evidence_id": SEG_A, "name": "异常振动", "type": "fault_mode"},
            {"evidence_id": SEG_A, "name": "滚动轴承", "type": "component"},
        ]
    )
    response = _response(
        [
            _entity("异常振动", "symptom", SEG_A),
            _entity("滚动轴承", "component", SEG_A),
        ]
    )

    score = _score(tmp_path, payload, [response])

    assert score.matched_entities == 1
    assert score.type_mismatch_entities == 1
    assert score.missing_entities == 0
    assert score.recall == pytest.approx(0.5)
    assert score.precision == pytest.approx(0.5)
    outcome = next(item for item in score.outcomes if "异常振动" in item.label)
    assert outcome.status == "type_mismatch"


def test_a_wrong_relation_type_is_counted_as_a_type_mismatch(tmp_path: Path) -> None:
    """关系类型不符在真实抽取路径上会被本体校验拦下；这里直接测判定逻辑。"""
    ontology = load_ontology()
    payload = _payload(
        expected_entities=[
            {"evidence_id": SEG_A, "name": "转子不平衡", "type": "fault_mode"},
            {"evidence_id": SEG_A, "name": "转子不对中", "type": "fault_mode"},
        ],
        expected_relations=[
            {
                "evidence_id": SEG_A,
                "head": "转子不平衡",
                "type": "confused_with",
                "tail": "转子不对中",
            }
        ],
    )
    annotations = load_annotations(ontology, _write(tmp_path, payload))
    entities = (
        ValidatedEntity(
            name="转子不平衡",
            type="fault_mode",
            evidence_ids=(SEG_A,),
            confidence=0.9,
            key="fault_mode:转子不平衡",
        ),
        ValidatedEntity(
            name="转子不对中",
            type="fault_mode",
            evidence_ids=(SEG_A,),
            confidence=0.9,
            key="fault_mode:转子不对中",
        ),
    )
    output = ValidatedExtraction(
        entities=entities,
        relations=(
            ExtractedRelation(
                head="转子不平衡",
                type="detected_by",
                tail="转子不对中",
                evidence_ids=[SEG_A],
                confidence=0.8,
            ),
        ),
        rejected_entities=0,
        rejected_relations=0,
    )

    score = score_extractions(annotations, {DOC_A: output})

    assert score.type_mismatch_relations == 1
    assert score.matched_relations == 0
    relation_outcome = next(
        outcome for outcome in score.outcomes if outcome.kind == "relation"
    )
    assert relation_outcome.status == "type_mismatch"


def test_assertions_are_scored_per_document(tmp_path: Path) -> None:
    """两篇文献都抽到同名概念时，各自命中，不能一篇顶掉另一篇。"""
    payload = _payload(
        expected_entities=[
            {"evidence_id": SEG_A, "name": "滚动轴承", "type": "component"},
            {"evidence_id": SEG_B, "name": "滚动轴承", "type": "component"},
        ]
    )
    response_a = _response([_entity("滚动轴承", "component", SEG_A)])
    response_b = _response([_entity("滚动轴承", "component", SEG_B)])

    score = _score(tmp_path, payload, [response_a, response_b])

    assert score.recall == 1.0
    assert score.precision == 1.0
    assert [document.doc_id for document in score.documents] == [DOC_A, DOC_B]


def test_the_runner_calls_the_model_once_per_document(tmp_path: Path) -> None:
    ontology = load_ontology()
    annotations = load_annotations(ontology, _write(tmp_path, _default_payload()))
    response_a = _response([_entity("振动", "symptom", SEG_A)])
    response_b = _response([_entity("振动信号", "symptom", SEG_B)])
    llm = FakeLLM([response_a, response_b])

    outputs = extract_annotated(llm, ontology, annotations)

    assert llm.calls == 2
    assert set(outputs) == {DOC_A, DOC_B}
    assert outputs[DOC_A].entities[0].name == "振动"


def test_scoring_rejects_output_for_a_document_outside_the_annotation_set(
    tmp_path: Path,
) -> None:
    ontology = load_ontology()
    annotations = load_annotations(ontology, _write(tmp_path, _default_payload()))
    output = ValidatedExtraction(
        entities=(), relations=(), rejected_entities=0, rejected_relations=0
    )

    with pytest.raises(AnnotationSetError, match="未在标注集中"):
        score_extractions(annotations, {"cccccccccccccccc": output})

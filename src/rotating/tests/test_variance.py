"""方差度量的门面测试：用假模型断言分布，不碰网络。"""

from __future__ import annotations

import json

import pytest

from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_rotating.ontology import load_ontology
from idr_rotating.variance import measure_extraction_variance

SEGMENT = EvidenceSegment(
    id="ev:doc:1:0-20",
    doc_id="doc",
    page=1,
    text="外圈剥落表现为 BPFO 倍频振动。",
    charspan=CharSpan(0, 20),
    bbox=BBox(0, 0, 10, 10),
)


class FakeLLM:
    """按给定顺序回放响应，并记录调用次数。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _payload(entity: str, relation: tuple[str, str, str] | None = None) -> str:
    body: dict[str, object] = {
        "entities": [
            {
                "name": entity,
                "type": "fault_mode",
                "evidence_ids": [SEGMENT.id],
                "confidence": 0.9,
            }
        ],
        "relations": [],
    }
    if relation is not None:
        head, relation_type, tail = relation
        body["entities"].append(  # type: ignore[union-attr]
            {
                "name": tail,
                "type": "symptom",
                "evidence_ids": [SEGMENT.id],
                "confidence": 0.8,
            }
        )
        body["relations"] = [
            {
                "head": head,
                "type": relation_type,
                "tail": tail,
                "evidence_ids": [SEGMENT.id],
                "confidence": 0.8,
            }
        ]
    return json.dumps(body, ensure_ascii=False)


def test_identical_runs_report_full_overlap() -> None:
    llm = FakeLLM([_payload("外圈剥落")])

    report = measure_extraction_variance(
        llm, load_ontology(), (SEGMENT,), runs=3
    )

    assert report.entity_counts == (1, 1, 1)
    assert report.identical is True
    assert report.mean_overlap == 1.0
    assert llm.calls == 3


def test_different_runs_report_the_observed_spread() -> None:
    """同一提示词回放不同产出，报告要如实反映分散程度。"""
    llm = FakeLLM(
        [
            _payload("外圈剥落"),
            _payload("内圈剥落"),
            _payload("外圈剥落"),
        ]
    )

    report = measure_extraction_variance(
        llm, load_ontology(), (SEGMENT,), runs=3
    )

    assert report.identical is False
    assert report.mean_overlap is not None
    assert 0.0 < report.mean_overlap < 1.0
    # 三次两两比较共 3 对。
    assert len(report.pairwise_overlap) == 3


def test_relation_counts_are_reported_per_run() -> None:
    llm = FakeLLM(
        [
            _payload("外圈剥落"),
            _payload("外圈剥落", ("外圈剥落", "indicated_by", "BPFO")),
        ]
    )

    report = measure_extraction_variance(
        llm, load_ontology(), (SEGMENT,), runs=2
    )

    assert report.relation_counts == (0, 1)


def test_a_single_run_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少为 2"):
        measure_extraction_variance(FakeLLM([_payload("外圈剥落")]), load_ontology(), (SEGMENT,), runs=1)

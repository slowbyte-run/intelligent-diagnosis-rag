"""抽取评估重复运行的门面测试：假模型回放固定响应，断言分布而不是单点。

不碰网络；真实模型的多次运行由命令行层负责。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idr_rotating.extraction_distribution import measure_extraction_distribution
from idr_rotating.extraction_evaluation import AnnotationSet, load_annotations
from idr_rotating.ontology import load_ontology

DOC_A = "aaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbb"
SEG_A = f"ev:{DOC_A}:1:0-10"
TEXT_A = "外圈剥落表现为振动。"
SEG_B = f"ev:{DOC_B}:2:0-5"
TEXT_B = "齿面点蚀。"


class FakeLLM:
    """按给定顺序回放响应，并记录调用次数。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _entity(name: str, type_id: str, evidence_id: str) -> dict[str, object]:
    return {
        "name": name,
        "type": type_id,
        "evidence_ids": [evidence_id],
        "confidence": 0.9,
    }


def _response(entities: list[dict[str, object]]) -> str:
    return json.dumps(
        {"entities": entities, "relations": []}, ensure_ascii=False
    )


def _annotations(tmp_path: Path) -> AnnotationSet:
    """两篇文献、各一条期望实体；评分按文献分组，不会互相顶替。"""
    payload = {
        "schema_version": 1,
        "name": "测试标注集",
        "segments": [
            {"id": SEG_A, "page": 1, "text": TEXT_A},
            {"id": SEG_B, "page": 2, "text": TEXT_B},
        ],
        "expected_entities": [
            {"evidence_id": SEG_A, "name": "振动", "type": "symptom"},
            {"evidence_id": SEG_B, "name": "齿面点蚀", "type": "fault_mode"},
        ],
        "expected_relations": [],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return load_annotations(load_ontology(), path)


def test_identical_runs_report_zero_spread_and_full_overlap(
    tmp_path: Path,
) -> None:
    annotations = _annotations(tmp_path)
    response_a = _response([_entity("振动", "symptom", SEG_A)])
    response_b = _response([_entity("齿面点蚀", "fault_mode", SEG_B)])
    llm = FakeLLM([response_a, response_b] * 3)

    report = measure_extraction_distribution(
        llm, load_ontology(), annotations, runs=3
    )

    assert [run.recall for run in report.runs] == [1.0, 1.0, 1.0]
    assert [run.precision for run in report.runs] == [1.0, 1.0, 1.0]
    assert report.recall.spread == 0.0
    assert report.precision.spread == 0.0
    assert report.pairwise_overlap == (1.0, 1.0, 1.0)
    assert report.mean_overlap == 1.0
    assert report.identical_hits is True
    # 两篇文献 × 三次运行，每篇每轮一次模型调用。
    assert llm.calls == 6


def test_different_runs_report_the_observed_spread_and_overlap(
    tmp_path: Path,
) -> None:
    """同一提示词回放不同产出，报告要如实反映每次的分数与分散程度。"""
    annotations = _annotations(tmp_path)
    llm = FakeLLM(
        [
            # 第 1 次：两条期望断言全部命中。
            _response([_entity("振动", "symptom", SEG_A)]),
            _response([_entity("齿面点蚀", "fault_mode", SEG_B)]),
            # 第 2 次：第二篇只抽到一个不在标注里的实体。
            _response([_entity("振动", "symptom", SEG_A)]),
            _response([_entity("异常噪声", "symptom", SEG_B)]),
            # 第 3 次：全部命中，另带一个多出的断言。
            _response([_entity("振动", "symptom", SEG_A)]),
            _response(
                [
                    _entity("齿面点蚀", "fault_mode", SEG_B),
                    _entity("异常噪声", "symptom", SEG_B),
                ]
            ),
        ]
    )

    report = measure_extraction_distribution(
        llm, load_ontology(), annotations, runs=3
    )

    assert [run.recall for run in report.runs] == pytest.approx([1.0, 0.5, 1.0])
    assert [run.precision for run in report.runs] == pytest.approx(
        [1.0, 0.5, 2 / 3]
    )
    assert report.recall.spread == pytest.approx(0.5)
    assert report.precision.spread == pytest.approx(0.5)
    # 三次两两比较：命中集合分别是 {A,B}、{A}、{A,B}。
    assert report.pairwise_overlap == pytest.approx((0.5, 1.0, 0.5))
    assert report.mean_overlap == pytest.approx(2 / 3)
    assert report.identical_hits is False

    first, second, third = report.runs
    assert first.hit_keys == third.hit_keys
    assert second.hit_keys < first.hit_keys
    assert report.consistent_hits == first.hit_keys & second.hit_keys & third.hit_keys
    assert len(report.consistent_hits) == 1
    assert len(report.flaky_hits) == 1
    assert len(report.ever_hits) == 2


@pytest.mark.parametrize("runs", [0, 1])
def test_fewer_than_two_runs_is_rejected(tmp_path: Path, runs: int) -> None:
    annotations = _annotations(tmp_path)

    with pytest.raises(ValueError, match="至少为 2"):
        measure_extraction_distribution(
            FakeLLM([_response([])]), load_ontology(), annotations, runs=runs
        )

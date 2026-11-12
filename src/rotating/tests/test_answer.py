"""答案值对象：拒答原因，以及「拒答不带任何结论」的不变量。

拒答是答案的一种显式形态：``path`` 说明没答，``refusal_reason`` 说明为什么没
答（问题没被理解，还是理解了但没依据）。两条不变量由值对象自己守住，不指望
每个调用点自觉：拒绝必须带原因，且拒绝时不得携带结论或引用。
"""

from __future__ import annotations

import pytest

from idr_core.graph import Concept
from idr_rotating import (
    Answer,
    AnswerPath,
    Citation,
    Comparison,
    ConceptMatch,
    Fact,
    LinkKind,
    RefusalReason,
    SharedFact,
)
from idr_rotating.answer import answer_to_dict

QUESTION = "今天天气怎么样"


def refusal_answer(
    *, reason: RefusalReason | None = RefusalReason.NO_CONCEPT
) -> Answer:
    return Answer(
        question=QUESTION,
        path=AnswerPath.REFUSAL,
        matches=(),
        focus=None,
        facts=(),
        citations=(),
        detail="本体别名表没有在问题中匹配到概念。",
        refusal_reason=reason,
    )


def test_refusal_reason_is_part_of_the_answer_contract() -> None:
    payload = answer_to_dict(refusal_answer())

    assert payload["schema_version"] == 4
    assert payload["path"] == "refusal"
    assert payload["refusal_reason"] == "no_concept"
    assert payload["facts"] == []
    assert payload["citations"] == []


def test_answered_answers_do_not_carry_a_refusal_reason() -> None:
    answer = Answer(
        question=QUESTION,
        path=AnswerPath.VECTOR,
        matches=(),
        focus=None,
        facts=(),
        citations=(),
    )

    payload = answer_to_dict(answer)

    assert payload["refusal_reason"] is None


def test_a_refusal_must_say_why_it_refuses() -> None:
    with pytest.raises(ValueError, match="拒答原因"):
        refusal_answer(reason=None)


def test_a_refusal_must_not_carry_conclusions_or_citations() -> None:
    concept = Concept(id="concept:steam_turbine", name="汽轮机", type="equipment")
    fact = Fact(
        relation="has_fault_mode",
        head=concept,
        tail=concept,
        confidence=0.9,
        evidence_ids=("ev:doc-a:1:0-4",),
    )
    citation = Citation(
        evidence_id="ev:doc-a:1:0-4",
        doc_id="doc-a",
        document="论文A.pdf",
        page=1,
        text="汽轮机故障。",
    )

    with pytest.raises(ValueError, match="拒答"):
        Answer(
            question=QUESTION,
            path=AnswerPath.REFUSAL,
            matches=(),
            focus=None,
            facts=(fact,),
            citations=(),
            detail="没有依据。",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    with pytest.raises(ValueError, match="拒答"):
        Answer(
            question=QUESTION,
            path=AnswerPath.REFUSAL,
            matches=(),
            focus=None,
            facts=(),
            citations=(citation,),
            detail="没有依据。",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )


OUTER = Concept(id="concept:outer_race", name="外圈剥落", type="fault_mode")
INNER = Concept(id="concept:inner_race", name="内圈剥落", type="fault_mode")
BPFO = Concept(id="concept:bpfo", name="外圈故障频率", type="symptom")
BPFI = Concept(id="concept:bpfi", name="内圈故障频率", type="symptom")
SPECTRUM = Concept(id="concept:fft", name="频谱分析", type="detection_method")
EV_OUTER_DET = "ev:doc-a:3:0-4"
EV_INNER_DET = "ev:doc-a:3:5-9"
EV_OUTER_SYMPTOM = "ev:doc-a:2:0-4"


def comparison_answer() -> Answer:
    left = ConceptMatch(concept=OUTER, alias="外圈剥落", kind=LinkKind.SUBSTRING)
    right = ConceptMatch(concept=INNER, alias="内圈剥落", kind=LinkKind.SUBSTRING)

    def fact(relation: str, head: Concept, tail: Concept, evidence_id: str) -> Fact:
        return Fact(
            relation=relation,
            head=head,
            tail=tail,
            confidence=0.8,
            evidence_ids=(evidence_id,),
        )

    shared_left = fact("detected_by", OUTER, SPECTRUM, EV_OUTER_DET)
    shared_right = fact("detected_by", INNER, SPECTRUM, EV_INNER_DET)
    return Answer(
        question="外圈剥落和内圈剥落怎么区分？",
        path=AnswerPath.GRAPH,
        matches=(left, right),
        focus=None,
        facts=(),
        citations=(
            Citation(
                evidence_id=EV_OUTER_DET,
                doc_id="doc-a",
                document="论文A.pdf",
                page=3,
                text="外圈剥落常用频谱分析确认。",
            ),
            Citation(
                evidence_id=EV_INNER_DET,
                doc_id="doc-a",
                document="论文A.pdf",
                page=3,
                text="内圈剥落也常用频谱分析确认。",
            ),
            Citation(
                evidence_id=EV_OUTER_SYMPTOM,
                doc_id="doc-a",
                document="论文A.pdf",
                page=2,
                text="外圈剥落会出现 BPFO 及其倍频。",
            ),
        ),
        comparison=Comparison(
            left=left,
            right=right,
            shared=(
                SharedFact(
                    relation="detected_by",
                    tail=SPECTRUM,
                    left=(shared_left,),
                    right=(shared_right,),
                ),
            ),
            only_left=(
                fact("indicated_by", OUTER, BPFO, EV_OUTER_SYMPTOM),
            ),
            only_right=(fact("indicated_by", INNER, BPFI, "ev:doc-a:2:5-9"),),
            discriminating=(),
        ),
    )


def test_comparison_is_part_of_the_answer_contract() -> None:
    payload = answer_to_dict(comparison_answer())

    assert payload["schema_version"] == 4
    comparison = payload["comparison"]
    assert comparison is not None
    assert comparison["left"]["concept"]["id"] == OUTER.id
    assert comparison["right"]["concept"]["id"] == INNER.id
    assert comparison["shared"] == [
        {
            "relation": "detected_by",
            "tail": {"id": SPECTRUM.id, "name": SPECTRUM.name, "type": SPECTRUM.type},
            "left": [
                {
                    "relation": "detected_by",
                    "head": {"id": OUTER.id, "name": OUTER.name, "type": OUTER.type},
                    "tail": {"id": SPECTRUM.id, "name": SPECTRUM.name, "type": SPECTRUM.type},
                    "confidence": 0.8,
                    "evidence_ids": [EV_OUTER_DET],
                    "path": [],
                }
            ],
            "right": [
                {
                    "relation": "detected_by",
                    "head": {"id": INNER.id, "name": INNER.name, "type": INNER.type},
                    "tail": {"id": SPECTRUM.id, "name": SPECTRUM.name, "type": SPECTRUM.type},
                    "confidence": 0.8,
                    "evidence_ids": [EV_INNER_DET],
                    "path": [],
                }
            ],
        }
    ]
    assert [item["relation"] for item in comparison["only_left"]] == ["indicated_by"]
    assert [item["relation"] for item in comparison["only_right"]] == ["indicated_by"]
    assert comparison["discriminating"] == []
    assert payload["facts"] == []


def test_non_comparative_answers_keep_the_single_focus_shape() -> None:
    payload = answer_to_dict(refusal_answer())

    assert payload["comparison"] is None


def test_a_refusal_must_not_carry_a_comparison() -> None:
    with pytest.raises(ValueError, match="拒答"):
        Answer(
            question=QUESTION,
            path=AnswerPath.REFUSAL,
            matches=(),
            focus=None,
            facts=(),
            citations=(),
            detail="没有依据。",
            refusal_reason=RefusalReason.NO_EVIDENCE,
            comparison=comparison_answer().comparison,
        )

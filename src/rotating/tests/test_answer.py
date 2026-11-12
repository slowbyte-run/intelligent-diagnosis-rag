"""答案值对象：拒答原因，以及「拒答不带任何结论」的不变量。

拒答是答案的一种显式形态：``path`` 说明没答，``refusal_reason`` 说明为什么没
答（问题没被理解，还是理解了但没依据）。两条不变量由值对象自己守住，不指望
每个调用点自觉：拒绝必须带原因，且拒绝时不得携带结论或引用。
"""

from __future__ import annotations

import pytest

from idr_core.graph import Concept
from idr_rotating import Answer, AnswerPath, Citation, Fact, RefusalReason
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

    assert payload["schema_version"] == 3
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

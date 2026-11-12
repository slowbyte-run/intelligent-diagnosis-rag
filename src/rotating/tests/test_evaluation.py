"""金标准集加载与三项指标评分的单元测试。

评分逻辑只依赖答案的 dict 形态（``answer_to_dict`` 的契约）与一个可注入的
引用校验函数，因此全部用假答案测试，不需要 Neo4j、PDF 或网络。回源 PDF 的
真实验证另见 ``test_citation_check.py``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from idr_core.documents import document_id
from idr_rotating.citation_check import CitationCheck
from idr_rotating.evaluation import (
    EvaluationError,
    GoldenQuestion,
    GoldenSetError,
    evaluate_golden,
    load_golden_set,
    report_to_dict,
)
from idr_rotating.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_A = "0123456789abcdef"
DOC_B = "fedcba9876543210"
EV_A = f"ev:{DOC_A}:1:0-5"
EV_B = f"ev:{DOC_B}:3:10-20"


def make_question(
    *,
    question_id: str = "T-01",
    question_class: str = "factual",
    question_text: str = "阿尔法有什么征兆？",
    expected_concepts: tuple[str, ...] = ("concept:alpha",),
    expected_evidence_ids: tuple[str, ...] = (EV_A,),
    expect_refusal: bool = False,
) -> GoldenQuestion:
    return GoldenQuestion(
        id=question_id,
        question_class=question_class,
        question=question_text,
        expected_concepts=expected_concepts,
        expected_evidence_ids=expected_evidence_ids,
        expect_refusal=expect_refusal,
        expected_refusal_reason="no_evidence" if expect_refusal else None,
        note="",
    )


def make_answer(
    *,
    path: str = "graph",
    focus_concept: str | None = "concept:alpha",
    facts: tuple[tuple[str, str, str], ...] = (("indicated_by", "concept:alpha", "concept:beta"),),
    citations: tuple[tuple[str, str], ...] = ((EV_A, DOC_A),),
) -> dict[str, Any]:
    """构造 ``answer_to_dict`` 形态的假答案：facts 是 (关系, 头, 尾)，citations 是 (证据, 文献)。"""
    concept_names = {
        "concept:alpha": ("阿尔法", "fault_mode"),
        "concept:beta": ("贝塔", "symptom"),
    }

    def concept(concept_id: str) -> dict[str, str]:
        name, concept_type = concept_names.get(concept_id, (concept_id, "fault_mode"))
        return {"id": concept_id, "name": name, "type": concept_type}

    return {
        "schema_version": 1,
        "question": "阿尔法有什么征兆？",
        "path": path,
        "detail": None,
        "focus": (
            {
                "concept": concept(focus_concept),
                "alias": "阿尔法",
                "kind": "substring",
            }
            if focus_concept is not None
            else None
        ),
        "matches": [],
        "facts": [
            {
                "relation": relation,
                "head": concept(head),
                "tail": concept(tail),
                "confidence": 0.9,
                "evidence_ids": [evidence_id for evidence_id, _ in citations],
            }
            for relation, head, tail in facts
        ],
        "citations": [
            {
                "evidence_id": evidence_id,
                "doc_id": doc_id,
                "document": None,
                "page": 1,
                "text": "证据原文",
            }
            for evidence_id, doc_id in citations
        ],
    }


def always_ok(citation: dict[str, Any]) -> CitationCheck:
    return CitationCheck(ok=True, reason="一致", offset=0)


def write_golden(path: Path, question: dict[str, Any]) -> Path:
    payload = {
        "schema_version": 1,
        "questions": [question],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def valid_question(**overrides: object) -> dict[str, Any]:
    question = {
        "id": "T-01",
        "class": "factual",
        "question": "阿尔法有什么征兆？",
        "expected_concepts": ["concept:alpha"],
        "expected_evidence_ids": [EV_A],
    }
    question.update(overrides)
    return question


def test_shipped_golden_set_has_fifty_questions_across_three_classes() -> None:
    questions = load_golden_set()

    assert len(questions) == 50
    assert len({question.id for question in questions}) == 50
    assert {question.question_class for question in questions} == {
        "factual",
        "comparative",
        "cross_document",
    }
    assert sum(q.question_class == "factual" for q in questions) == 26
    assert sum(q.question_class == "comparative" for q in questions) == 12
    assert sum(q.question_class == "cross_document" for q in questions) == 12
    assert sum(q.expect_refusal for q in questions) == 2


def test_shipped_golden_set_binds_only_ontology_concept_ids() -> None:
    ontology = load_ontology()
    known = {
        f"concept:{instance.id}"
        for instances in ontology.instances.values()
        for instance in instances
    }

    for question in load_golden_set():
        unknown = set(question.expected_concepts) - known
        assert not unknown, f"{question.id} 绑定了本体里不存在的概念：{sorted(unknown)}"


def test_shipped_golden_set_binds_evidence_from_the_local_corpus() -> None:
    """证据段必须来自本地语料的 PDF；语料不入库，因此没有语料时跳过。"""
    corpus_dir = REPO_ROOT / "data" / "input"
    if not corpus_dir.is_dir():
        pytest.skip("data/input 不入库；本机有语料时才校验证据段绑定")

    known_documents = {
        document_id(path.read_bytes()) for path in corpus_dir.glob("*.pdf")
    }

    for question in load_golden_set():
        for evidence_id in question.expected_evidence_ids:
            doc_id = evidence_id.split(":")[1]
            assert doc_id in known_documents, (
                f"{question.id} 绑定的证据段不属于语料目录里的任何文献：{evidence_id}"
            )


def test_load_golden_set_rejects_unknown_class(tmp_path: Path) -> None:
    path = write_golden(tmp_path / "golden.json", valid_question(**{"class": "opinion"}))

    with pytest.raises(GoldenSetError, match="class"):
        load_golden_set(path)


def test_load_golden_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "questions": [valid_question(), valid_question()],
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(GoldenSetError, match="重复"):
        load_golden_set(path)


def test_load_golden_set_requires_expectations_for_answerable_questions(
    tmp_path: Path,
) -> None:
    path = write_golden(tmp_path / "golden.json", valid_question(expected_evidence_ids=[]))

    with pytest.raises(GoldenSetError, match="expected_evidence_ids"):
        load_golden_set(path)


def test_load_golden_set_refuses_to_mix_refusal_with_expectations(
    tmp_path: Path,
) -> None:
    path = write_golden(
        tmp_path / "golden.json",
        valid_question(expect_refusal=True, expected_refusal_reason="no_concept"),
    )

    with pytest.raises(GoldenSetError, match="拒答"):
        load_golden_set(path)


def test_load_golden_set_requires_two_documents_for_cross_document(
    tmp_path: Path,
) -> None:
    path = write_golden(
        tmp_path / "golden.json",
        valid_question(**{"class": "cross_document"}, expected_evidence_ids=[EV_A]),
    )

    with pytest.raises(GoldenSetError, match="跨文献"):
        load_golden_set(path)


def test_load_golden_set_rejects_evidence_ids_without_a_document_prefix(
    tmp_path: Path,
) -> None:
    path = write_golden(
        tmp_path / "golden.json",
        valid_question(expected_evidence_ids=["not-an-evidence-id"]),
    )

    with pytest.raises(GoldenSetError, match="证据段"):
        load_golden_set(path)


def test_load_golden_set_rejects_free_word_concept_ids(tmp_path: Path) -> None:
    path = write_golden(
        tmp_path / "golden.json",
        valid_question(expected_concepts=["0123456789abcdef:concept:fault_mode:外圈故障"]),
    )

    with pytest.raises(GoldenSetError, match="概念"):
        load_golden_set(path)


def test_factual_answer_hits_when_every_expected_concept_and_evidence_is_present() -> None:
    report = evaluate_golden(
        [
            make_question(
                expected_concepts=("concept:alpha", "concept:beta"),
                expected_evidence_ids=(EV_A,),
            )
        ],
        answer_for=lambda question: make_answer(),
        verify_citation=always_ok,
    )

    case = report.cases[0]
    assert case.hit is True
    assert case.concept_ok is True
    assert case.evidence_ok is True
    assert report.metrics.retrieval_hits == 1
    assert report.metrics.retrieval_hit_rate == 1.0
    assert report.metrics.citation_accuracy == 1.0
    assert case.refusal_correct is True


def test_concept_coverage_is_reported_but_does_not_gate_the_hit() -> None:
    """命中只看证据：期望概念是某一次抽取的产物，抽取有采样方差。

    同一篇同提示词两次重抽，实体 9→13、关系 7→11；把概念并进命中判据会让
    分数随抽样浮动几十个点。概念覆盖作为独立诊断项保留。
    """
    report = evaluate_golden(
        [make_question(expected_concepts=("concept:alpha", "concept:missing"))],
        answer_for=lambda question: make_answer(),
        verify_citation=always_ok,
    )

    case = report.cases[0]
    assert case.hit is True
    assert case.concept_ok is False
    assert case.missing_concepts == ("concept:missing",)
    assert report.metrics.retrieval_hit_rate == 1.0


def test_hit_requires_at_least_one_expected_evidence_segment() -> None:
    report = evaluate_golden(
        [make_question(expected_evidence_ids=(EV_B,))],
        answer_for=lambda question: make_answer(),
        verify_citation=always_ok,
    )

    case = report.cases[0]
    assert case.hit is False
    assert case.evidence_ok is False
    assert case.matched_evidence_ids == ()


def test_cross_document_hit_requires_every_expected_document() -> None:
    question = make_question(
        question_class="cross_document",
        expected_evidence_ids=(EV_A, EV_B),
    )

    report = evaluate_golden(
        [question],
        answer_for=lambda item: make_answer(),
        verify_citation=always_ok,
    )

    case = report.cases[0]
    assert case.hit is False
    assert case.document_ok is False
    assert case.missing_documents == (DOC_B,)


def test_cross_document_hits_when_all_documents_are_cited() -> None:
    question = make_question(
        question_class="cross_document",
        expected_evidence_ids=(EV_A, EV_B),
    )

    report = evaluate_golden(
        [question],
        answer_for=lambda item: make_answer(citations=((EV_A, DOC_A), (EV_B, DOC_B))),
        verify_citation=always_ok,
    )

    assert report.cases[0].hit is True
    assert report.cases[0].document_ok is True


def test_citation_accuracy_is_verified_citations_over_emitted_citations() -> None:
    def verify(citation: dict[str, Any]) -> CitationCheck:
        if citation["evidence_id"] == EV_B:
            return CitationCheck(ok=False, reason="引文未在源 PDF 该页逐字出现")
        return CitationCheck(ok=True, reason="一致", offset=0)

    report = evaluate_golden(
        [make_question()],
        answer_for=lambda question: make_answer(
            citations=((EV_A, DOC_A), (EV_B, DOC_B))
        ),
        verify_citation=verify,
    )

    assert report.metrics.citations_total == 2
    assert report.metrics.citations_verified == 1
    assert report.metrics.citation_accuracy == 0.5
    assert report.cases[0].citation_failures == (
        (EV_B, "引文未在源 PDF 该页逐字出现"),
    )


def test_refusal_correctness_scores_both_directions() -> None:
    questions = [
        make_question(question_id="A-01", question_text="可答题一"),
        make_question(question_id="A-02", question_text="可答题二"),
        make_question(
            question_id="R-01",
            question_text="拒答题一",
            expect_refusal=True,
            expected_concepts=(),
            expected_evidence_ids=(),
        ),
        make_question(
            question_id="R-02",
            question_text="拒答题二",
            expect_refusal=True,
            expected_concepts=(),
            expected_evidence_ids=(),
        ),
    ]
    refusal_answer = make_answer(
        path="refusal", focus_concept=None, facts=(), citations=()
    )
    answers = {
        "可答题一": make_answer(),
        "可答题二": refusal_answer,
        "拒答题一": refusal_answer,
        "拒答题二": make_answer(),
    }

    report = evaluate_golden(
        questions,
        answer_for=lambda question: answers[question],
        verify_citation=always_ok,
    )

    assert [case.refusal_correct for case in report.cases] == [True, False, True, False]
    assert report.metrics.refusal_correct == 2
    assert report.metrics.refusal_correctness == 0.5


def test_refusal_questions_are_excluded_from_the_hit_rate_denominator() -> None:
    refusal = make_question(
        question_id="R-01",
        expect_refusal=True,
        expected_concepts=(),
        expected_evidence_ids=(),
    )

    report = evaluate_golden(
        [refusal],
        answer_for=lambda question: make_answer(
            path="refusal", focus_concept=None, facts=(), citations=()
        ),
        verify_citation=always_ok,
    )

    assert report.cases[0].hit is None
    assert report.metrics.answerable_count == 0
    assert report.metrics.retrieval_hit_rate is None
    assert report.metrics.refusal_correctness == 1.0


def test_answer_reader_tolerates_additive_schema_changes() -> None:
    """#15 会把 schema_version 升到 2 并新增 Fact.path / truncated；评估器不应因此失败。"""
    answer = make_answer()
    answer["schema_version"] = 2
    answer["truncated"] = False
    for fact in answer["facts"]:
        fact["path"] = []

    report = evaluate_golden(
        [make_question()],
        answer_for=lambda question: answer,
        verify_citation=always_ok,
    )

    assert report.cases[0].hit is True


def test_answer_reader_reads_intermediate_concepts_from_a_fact_path() -> None:
    """#15 的 Fact.path 把中间概念放在路径里；只读 head/tail 会漏掉它。"""
    answer = make_answer()
    answer["schema_version"] = 2
    answer["facts"][0]["path"] = [
        {
            "relation": "has_fault_mode",
            "head": {"id": "concept:alpha", "name": "阿尔法", "type": "fault_mode"},
            "tail": {"id": "concept:gamma", "name": "伽马", "type": "fault_mode"},
            "confidence": 0.9,
            "evidence_ids": [EV_A],
        },
        {
            "relation": "indicated_by",
            "head": {"id": "concept:gamma", "name": "伽马", "type": "fault_mode"},
            "tail": {"id": "concept:beta", "name": "贝塔", "type": "symptom"},
            "confidence": 0.9,
            "evidence_ids": [EV_A],
        },
    ]

    report = evaluate_golden(
        [make_question(expected_concepts=("concept:alpha", "concept:gamma"))],
        answer_for=lambda question: answer,
        verify_citation=always_ok,
    )

    assert report.cases[0].hit is True
    assert report.cases[0].concept_ok is True


def test_answer_without_a_valid_schema_version_is_rejected() -> None:
    answer = make_answer()
    answer["schema_version"] = 0

    with pytest.raises(ValueError, match="schema_version"):
        evaluate_golden(
            [make_question()],
            answer_for=lambda question: answer,
            verify_citation=always_ok,
        )


def test_report_to_dict_is_json_serializable_and_keeps_per_case_detail() -> None:
    report = evaluate_golden(
        [make_question()],
        answer_for=lambda question: make_answer(),
        verify_citation=always_ok,
        generated_at="2026-11-12T00:00:00+00:00",
    )

    payload = report_to_dict(report)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-11-12T00:00:00+00:00"
    assert payload["metrics"]["retrieval_hit_rate"] == 1.0
    assert payload["metrics"]["by_class"]["factual"]["retrieval_hit_rate"] == 1.0
    assert payload["cases"][0]["id"] == "T-01"
    assert payload["cases"][0]["hit"] is True
    assert "阿尔法" in serialized


def comparison_answer() -> dict[str, Any]:
    """``answer_to_dict`` 形态的对比答案：两个锚点 + 一个共同结论。"""
    payload = make_answer(
        focus_concept=None,
        facts=(),
        citations=((EV_A, DOC_A), (EV_B, DOC_B)),
    )
    payload["comparison"] = {
        "left": {
            "concept": {"id": "concept:alpha", "name": "阿尔法", "type": "fault_mode"},
            "alias": "阿尔法",
            "kind": "substring",
        },
        "right": {
            "concept": {"id": "concept:beta", "name": "贝塔", "type": "fault_mode"},
            "alias": "贝塔",
            "kind": "substring",
        },
        "shared": [
            {
                "relation": "detected_by",
                "tail": {"id": "concept:gamma", "name": "伽马", "type": "detection_method"},
                "left": [
                    {
                        "relation": "detected_by",
                        "head": {"id": "concept:alpha", "name": "阿尔法", "type": "fault_mode"},
                        "tail": {"id": "concept:gamma", "name": "伽马", "type": "detection_method"},
                        "confidence": 0.7,
                        "evidence_ids": [EV_A],
                        "path": [],
                    }
                ],
                "right": [
                    {
                        "relation": "detected_by",
                        "head": {"id": "concept:beta", "name": "贝塔", "type": "fault_mode"},
                        "tail": {"id": "concept:gamma", "name": "伽马", "type": "detection_method"},
                        "confidence": 0.7,
                        "evidence_ids": [EV_B],
                        "path": [],
                    }
                ],
            }
        ],
        "only_left": [],
        "only_right": [],
        "discriminating": [],
    }
    return payload


def test_comparison_answers_expose_both_anchors_and_partition_concepts() -> None:
    question = make_question(
        question_id="P-01",
        question_class="comparative",
        question_text="阿尔法和贝塔怎么区分？",
        expected_concepts=("concept:alpha", "concept:beta", "concept:gamma"),
        expected_evidence_ids=(EV_A, EV_B),
    )

    report = evaluate_golden(
        [question],
        answer_for=lambda item: comparison_answer(),
        verify_citation=always_ok,
    )

    case = report.cases[0]
    assert case.concept_ok is True
    assert case.missing_concepts == ()
    assert case.evidence_ok is True
    assert case.hit is True
    assert report.metrics.citations_total == 2
    assert report.metrics.citation_accuracy == 1.0


def test_malformed_comparison_is_an_evaluation_error_not_a_silent_miss() -> None:
    payload = comparison_answer()
    payload["comparison"] = {"left": None}

    with pytest.raises(EvaluationError, match="comparison"):
        evaluate_golden(
            [make_question()],
            answer_for=lambda item: payload,
            verify_citation=always_ok,
        )

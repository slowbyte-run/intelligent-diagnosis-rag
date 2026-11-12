"""快照内的引用校验：引文必须是快照里所引证据段原文的逐字片段。

这是离线评估能给出的「引用准确率」口径：它核对答案的引用拼装与冻结图谱是否
一致，不能替代回源 PDF。两者的分工写在 ``docs/retrieval-eval.md``。
"""

from __future__ import annotations

from idr_core.graph.read import Concept, EvidenceRecord, FactEdge
from idr_core.graph.snapshot import GraphSnapshot, SnapshotClosure
from idr_core.models import CharSpan
from idr_rotating.citation_check import SnapshotCitationVerifier

DOC = "0123456789abcdef"
EVIDENCE_ID = f"ev:{DOC}:7:100-130"
TEXT = "外圈故障频率及其倍频是外圈剥落的典型征兆。"


def snapshot() -> GraphSnapshot:
    return GraphSnapshot.build(
        closure=SnapshotClosure(
            seed_concept_ids=("concept:alpha",),
            relation_types=("indicated_by",),
            structural_relations=(),
            max_hops=1,
        ),
        concepts=(
            Concept(id="concept:alpha", name="甲", type="fault_mode"),
            Concept(id="concept:beta", name="乙", type="symptom"),
        ),
        edges=(
            FactEdge(
                head_id="concept:alpha",
                relation="indicated_by",
                tail_id="concept:beta",
                evidence_ids=(EVIDENCE_ID,),
                confidence=0.9,
                doc_id=DOC,
            ),
        ),
        evidence=(
            EvidenceRecord(
                id=EVIDENCE_ID,
                doc_id=DOC,
                page=7,
                text=TEXT,
                charspan=CharSpan(100, 100 + len(TEXT)),
            ),
        ),
    )


def citation(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "evidence_id": EVIDENCE_ID,
        "doc_id": DOC,
        "page": 7,
        "text": "外圈故障频率及其倍频",
    }
    payload.update(overrides)
    return payload


def test_a_verbatim_slice_of_the_snapshot_evidence_is_consistent() -> None:
    check = SnapshotCitationVerifier(snapshot()).verify(citation())

    assert check.ok is True
    assert "逐字" in check.reason


def test_the_whole_evidence_text_is_also_consistent() -> None:
    assert SnapshotCitationVerifier(snapshot()).verify(citation(text=TEXT)).ok is True


def test_text_that_is_not_a_verbatim_slice_fails() -> None:
    check = SnapshotCitationVerifier(snapshot()).verify(
        citation(text="外圈故障频率是外圈剥落的典型征兆。")
    )

    assert check.ok is False
    assert "逐字" in check.reason


def test_a_citation_pointing_at_evidence_outside_the_snapshot_fails() -> None:
    check = SnapshotCitationVerifier(snapshot()).verify(
        citation(evidence_id="ev:ffffffffffffffff:1:0-1")
    )

    assert check.ok is False
    assert "快照" in check.reason


def test_a_page_that_does_not_match_the_snapshot_record_fails() -> None:
    check = SnapshotCitationVerifier(snapshot()).verify(citation(page=8))

    assert check.ok is False
    assert "页" in check.reason


def test_a_document_that_does_not_match_the_snapshot_record_fails() -> None:
    check = SnapshotCitationVerifier(snapshot()).verify(
        citation(doc_id="fedcba9876543210")
    )

    assert check.ok is False
    assert "文献" in check.reason


def test_empty_text_or_missing_page_is_reported_not_silently_passed() -> None:
    verifier = SnapshotCitationVerifier(snapshot())

    assert verifier.verify(citation(text="")).ok is False
    assert verifier.verify(citation(page=None)).ok is False

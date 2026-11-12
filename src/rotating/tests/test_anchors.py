"""引用锚点在入库门面上的行为：片段校验、区间固化、歧义标记与指纹。

锚点叠在证据段之上：切分与证据段标识不因锚点改变；片段对不上的断言进待审
队列而不是带着编造的引用入库。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from idr_core.graph import EvidenceAnchor
from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest
from idr_rotating.pending import PendingKind, RejectionReason

from fakes import (
    FakeEmbedder,
    FakeGraphStore,
    FakeLLM,
    FakeParser,
    FakeStateStore,
    FakeVectorStore,
)

PAGE_BBOX = BBox(0.0, 0.0, 595.0, 842.0)


def content_doc_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def evidence_id(doc_id: str, page: int, text: str) -> str:
    return f"ev:{doc_id}:{page}:0-{len(text)}"


def write_pdf(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def deps_for(
    response: str | list[str],
    *,
    text: str,
    graph_store: FakeGraphStore,
    page_count: int = 1,
    state_store: FakeStateStore | None = None,
) -> IngestDeps:
    return IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [
                ParsedPage(number=number, text=text, bbox=PAGE_BBOX)
                for number in range(1, page_count + 1)
            ]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        state_store=state_store,
    )


def entity(
    name: str, type_: str, segment_id: str, quote: str, confidence: float
) -> dict[str, object]:
    return {
        "name": name,
        "type": type_,
        "evidence_ids": [segment_id],
        "quote": quote,
        "confidence": confidence,
    }


def relation(
    head: str,
    type_: str,
    tail: str,
    segment_id: str,
    quote: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "head": head,
        "type": type_,
        "tail": tail,
        "evidence_ids": [segment_id],
        "quote": quote,
        "confidence": confidence,
    }


def test_verbatim_quote_is_frozen_as_a_character_span_on_both_edge_kinds(
    tmp_path: Path,
) -> None:
    """校验通过的支撑片段固化成 (evidence_id, start, end)，两类边共用一份结构。"""
    content = b"%PDF-1.4 anchor fixture"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
    segment_id = evidence_id(doc_id, 1, text)
    fault_quote = "外圈剥落表现为 BPFO 倍频振动"
    symptom_quote = "能量集中在高频段"

    response = json.dumps(
        {
            "entities": [
                entity("外圈剥落", "fault_mode", segment_id, fault_quote, 0.91),
                entity("BPFO 倍频", "symptom", segment_id, symptom_quote, 0.88),
            ],
            "relations": [
                relation(
                    "外圈剥落",
                    "indicated_by",
                    "BPFO 倍频",
                    segment_id,
                    fault_quote,
                    0.8,
                )
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path, deps=deps_for(response, text=text, graph_store=graph_store)
    )

    document = report.documents[0]
    assert document.error is None
    assert (document.entities, document.relations) == (2, 1)
    assert (document.rejected_entities, document.rejected_relations) == (0, 0)

    batch = graph_store.batches[0]
    assert [(segment.id, segment.charspan) for segment in batch.evidence] == [
        (segment_id, CharSpan(0, len(text)))
    ]

    concepts_by_id = {concept.id: concept for concept in batch.concepts}
    quotes_by_name = {
        "轴承外圈疲劳剥落": fault_quote,
        "外圈故障频率及其倍频": symptom_quote,
    }
    assert len(batch.evidence_links) == 2
    for link in batch.evidence_links:
        quote = quotes_by_name[concepts_by_id[link.concept_id].name]
        (anchor,) = link.provenance.anchors
        assert anchor == EvidenceAnchor(
            evidence_id=segment_id,
            start=text.index(quote),
            end=text.index(quote) + len(quote),
        )
        assert text[anchor.start : anchor.end] == quote
        assert anchor.ambiguous is False

    (graph_relation,) = batch.relations
    (relation_anchor,) = graph_relation.provenance.anchors
    assert relation_anchor == EvidenceAnchor(
        evidence_id=segment_id,
        start=text.index(fault_quote),
        end=text.index(fault_quote) + len(fault_quote),
    )
    assert text[relation_anchor.start : relation_anchor.end] == fault_quote


def test_assertions_whose_quote_is_not_in_the_segment_are_queued_not_written(
    tmp_path: Path,
) -> None:
    """片段不是所引证据段的子串（含模型没给片段）时，断言不得入库。"""
    content = b"%PDF-1.4 fabricated quote fixture"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
    segment_id = evidence_id(doc_id, 1, text)
    good_quote = "外圈剥落表现为 BPFO 倍频振动"
    fabricated_quote = "外圈剥落会导致轴承烧毁"

    response = json.dumps(
        {
            "entities": [
                entity("外圈剥落", "fault_mode", segment_id, good_quote, 0.9),
                entity("BPFO 倍频", "symptom", segment_id, "能量集中在高频段", 0.8),
                entity("内圈剥落", "fault_mode", segment_id, fabricated_quote, 0.7),
            ],
            "relations": [
                relation(
                    "外圈剥落",
                    "indicated_by",
                    "BPFO 倍频",
                    segment_id,
                    fabricated_quote,
                    0.8,
                )
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path, deps=deps_for(response, text=text, graph_store=graph_store)
    )

    document = report.documents[0]
    assert document.error is None, "片段对不上是待审数据，不是整篇失败"
    assert (document.entities, document.relations) == (2, 0)
    assert (document.rejected_entities, document.rejected_relations) == (1, 1)

    batch = graph_store.batches[0]
    assert {concept.name for concept in batch.concepts} == {
        "轴承外圈疲劳剥落",
        "外圈故障频率及其倍频",
    }
    assert batch.relations == ()
    assert len(batch.evidence_links) == 2
    assert all(len(link.provenance.anchors) == 1 for link in batch.evidence_links)

    reviews = {(review.kind, review.value): review for review in document.pending_reviews}
    entity_review = reviews[(PendingKind.ENTITY, "fault_mode")]
    assert entity_review.reason is RejectionReason.ANCHOR_NOT_FOUND
    assert entity_review.frequency == 1
    assert "内圈剥落" in entity_review.label
    assert entity_review.evidence_ids == (segment_id,)
    assert entity_review.examples[0].text == text

    relation_review = reviews[(PendingKind.RELATION, "indicated_by")]
    assert relation_review.reason is RejectionReason.ANCHOR_NOT_FOUND
    assert "外圈剥落" in relation_review.label
    assert "BPFO 倍频" in relation_review.label


def test_an_assertion_without_any_quote_is_queued_not_written(tmp_path: Path) -> None:
    """模型漏给片段与给出对不上的片段同样处理：没有可核对的出处就不入库。"""
    content = b"%PDF-1.4 missing quote fixture"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
    segment_id = evidence_id(doc_id, 1, text)

    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path, deps=deps_for(response, text=text, graph_store=graph_store)
    )

    document = report.documents[0]
    assert document.error is None
    assert (document.entities, document.rejected_entities) == (0, 1)
    assert graph_store.batches[0].concepts == ()

    (review,) = document.pending_reviews
    assert review.reason is RejectionReason.ANCHOR_NOT_FOUND


def test_a_quote_that_occurs_twice_anchors_the_first_occurrence_and_is_flagged(
    tmp_path: Path,
) -> None:
    """片段在同一证据段里出现多次时取首次并标记歧义，不猜是哪一处。"""
    content = b"%PDF-1.4 repeated quote fixture"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落会产生特征频率；内圈剥落同样会产生特征频率。"
    segment_id = evidence_id(doc_id, 1, text)
    quote = "剥落"

    response = json.dumps(
        {
            "entities": [
                entity("外圈剥落", "fault_mode", segment_id, quote, 0.9),
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path, deps=deps_for(response, text=text, graph_store=graph_store)
    )

    assert report.documents[0].error is None
    (link,) = graph_store.batches[0].evidence_links
    (anchor,) = link.provenance.anchors
    assert anchor.start == text.index(quote) == 2
    assert text[anchor.start : anchor.end] == quote
    assert anchor.ambiguous is True, "同一片段出现多次必须标记歧义"


def test_anchors_do_not_change_segmentation_or_evidence_ids(tmp_path: Path) -> None:
    """锚点只描述段内区间：两次不同抽取的证据段标识、区间与原文完全一致。"""
    content = b"%PDF-1.4 segmentation invariance fixture"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
    segment_id = evidence_id(doc_id, 1, text)

    first = json.dumps(
        {
            "entities": [
                entity("外圈剥落", "fault_mode", segment_id, "外圈剥落", 0.9)
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    second = json.dumps(
        {
            "entities": [
                entity("BPFO 倍频", "symptom", segment_id, "BPFO 倍频", 0.8)
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()

    ingest(tmp_path, deps=deps_for(first, text=text, graph_store=graph_store))
    ingest(tmp_path, deps=deps_for(second, text=text, graph_store=graph_store))

    first_batch, second_batch = graph_store.batches
    assert [
        (segment.id, segment.charspan, segment.text) for segment in first_batch.evidence
    ] == [
        (segment.id, segment.charspan, segment.text) for segment in second_batch.evidence
    ]
    assert first_batch.evidence[0].id == segment_id
    assert first_batch.evidence[0].charspan == CharSpan(0, len(text))
    assert first_batch.evidence[0].text == text

    for batch in graph_store.batches:
        (link,) = batch.evidence_links
        (anchor,) = link.provenance.anchors
        # 锚点落在证据段区间之内，不越界、不改变证据段本身。
        assert 0 <= anchor.start < anchor.end <= len(batch.evidence[0].text)


def test_changing_the_anchor_rule_invalidates_completed_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """锚点规则参与抽取指纹：改了规则，已完成的文献必须重抽而不是静默跳过。"""
    content = b"%PDF-1.4 anchor fingerprint fixture"
    write_pdf(tmp_path, "paper.pdf", content)
    state_store = FakeStateStore()
    llm = FakeLLM('{"entities": [], "relations": []}')
    deps = IngestDeps(
        llm=llm,
        parser=FakeParser(
            [
                ParsedPage(
                    number=1,
                    text="外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。",
                    bbox=PAGE_BBOX,
                )
            ]
        ),
        graph_store=FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        state_store=state_store,
    )

    first = ingest(tmp_path, deps=deps)
    second = ingest(tmp_path, deps=deps)
    assert first.documents[0].skipped is False
    assert second.documents[0].skipped is True
    assert len(llm.prompts) == 1

    monkeypatch.setattr("idr_core.anchors.ANCHOR_RULE_VERSION", "test-2")
    third = ingest(tmp_path, deps=deps)

    assert third.documents[0].skipped is False
    assert len(llm.prompts) == 2

"""分批抽取在入库门面上的行为：调用有界、跨批合并、引用仍限本批。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from idr_core.llm import LLMClient
from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_core.state import DocumentStatus
from idr_rotating import IngestDeps, ingest
from idr_rotating.extraction import MAX_BATCH_CHARS, MAX_BATCH_SEGMENTS

from fakes import (
    BatchwiseFakeLLM,
    FakeEmbedder,
    FakeGraphStore,
    FakeLLM,
    FakeParser,
    FakeStateStore,
    FakeVectorStore,
    evidence_lines_from_prompt,
)

PAGE_BBOX = BBox(0.0, 0.0, 595.0, 842.0)
SEGMENT_COUNT = 60
SHARED_NAME = "跨批共现部件"


def content_doc_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def patent_text(count: int) -> str:
    """生成 count 个带专利段号的短证据段。"""
    return " ".join(
        f"[{index:04d}] 第 {index} 段记录了一种可观测的征兆。"
        for index in range(1, count + 1)
    )


def deps_for(llm: LLMClient, graph_store: FakeGraphStore) -> IngestDeps:
    return IngestDeps(
        llm=llm,
        parser=FakeParser(
            [ParsedPage(number=1, text=patent_text(SEGMENT_COUNT), bbox=PAGE_BBOX)]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )


def test_a_long_document_is_extracted_in_bounded_batches(tmp_path: Path) -> None:
    content = b"%PDF-1.4 batched extraction"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = content_doc_id(content)
    llm = BatchwiseFakeLLM(shared_name=SHARED_NAME)
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    document = report.documents[0]
    assert document.error is None
    assert document.evidence_segments == SEGMENT_COUNT

    # 一篇 60 段的文献不能再整篇塞进一次调用；每次调用都受段数与字符数双重约束。
    assert len(llm.calls) > 1, "整篇 60 段不能仍只发一次模型调用"
    for evidence in llm.calls:
        assert len(evidence) <= MAX_BATCH_SEGMENTS
        assert sum(len(text) for _, text in evidence) <= MAX_BATCH_CHARS

    # 每条证据段恰好进入一批，分批不丢段也不重复发送。
    sent_ids = [evidence[0] for call in llm.calls for evidence in call]
    assert len(sent_ids) == SEGMENT_COUNT
    assert len(set(sent_ids)) == SEGMENT_COUNT

    # 每批各记一次调用，报告里的成本记录与真实调用一一对应。
    assert len(report.model_calls) == len(llm.calls)
    assert {call.document_id for call in report.model_calls} == {doc_id}

    # 报告计数按去重后的概念/关系算：共有一份 + 每批独有一份。
    batch_count = len(llm.calls)
    assert (document.entities, document.relations) == (
        batch_count + 1,
        batch_count,
    )

    (graph_batch,) = graph_store.batches
    assert len(graph_batch.concepts) == batch_count + 1
    assert {concept.name for concept in graph_batch.concepts} == {
        SHARED_NAME,
        *(f"本批故障{index}" for index in range(1, batch_count + 1)),
    }

    # 同一概念在每批都被抽到，合并后证据是各批证据的并集。
    shared = next(
        concept for concept in graph_batch.concepts if concept.name == SHARED_NAME
    )
    shared_evidence = {
        link.evidence_id
        for link in graph_batch.evidence_links
        if link.concept_id == shared.id
    }
    assert shared_evidence == {evidence[0][0] for evidence in llm.calls}


def test_batches_are_also_bounded_by_characters(tmp_path: Path) -> None:
    content = b"%PDF-1.4 long paragraphs"
    (tmp_path / "long.pdf").write_bytes(content)
    # 每段正好半页字符上限：5 段没到段数上限，字符数却已超过上限，必须拆批。
    long_text = "故障征兆" * (MAX_BATCH_CHARS // 2 // 4)
    pages = [
        ParsedPage(number=number, text=long_text, bbox=PAGE_BBOX)
        for number in range(1, 6)
    ]
    llm = BatchwiseFakeLLM(shared_name=SHARED_NAME)
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=FakeParser(pages),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    assert report.documents[0].error is None
    assert report.documents[0].evidence_segments == len(pages)
    assert len(llm.calls) > 1, "段数未到上限时分批仍须受字符数约束"
    for evidence in llm.calls:
        assert len(evidence) <= MAX_BATCH_SEGMENTS
        assert sum(len(text) for _, text in evidence) <= MAX_BATCH_CHARS


class CrossBatchCitationLLM:
    """第二批故意引用第一批的证据段，用来守住「引用必须属于本批」。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._first_evidence_id: str | None = None

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        evidence = evidence_lines_from_prompt(prompt)
        if not evidence:
            raise AssertionError("抽取提示词里没有任何证据段")
        if self._first_evidence_id is None:
            self._first_evidence_id = evidence[0][0]
            name, entity_type = "本批部件", "component"
        else:
            name, entity_type = "越批征兆", "symptom"

        return json.dumps(
            {
                "entities": [
                    {
                        "name": name,
                        "type": entity_type,
                        "evidence_ids": [self._first_evidence_id],
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            },
            ensure_ascii=False,
        )


def test_evidence_from_another_batch_is_still_a_validation_failure(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 cross batch citation"
    (tmp_path / "paper.pdf").write_bytes(content)
    llm = CrossBatchCitationLLM()
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    document = report.documents[0]
    assert document.error is None
    assert len(llm.prompts) > 1
    assert document.entities == 1
    assert document.rejected_entities == 1

    (pending,) = document.pending_reviews
    assert pending.kind == "entity"
    assert pending.reason == "unknown_evidence"
    assert pending.label == "越批征兆"

    (graph_batch,) = graph_store.batches
    assert [concept.name for concept in graph_batch.concepts] == ["本批部件"]


class SecondBatchFailsLLM:
    """第一批正常，之后每次都返回无法解析的文本。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) > 1:
            return "这不是 JSON"
        evidence = evidence_lines_from_prompt(prompt)
        return json.dumps(
            {
                "entities": [
                    {
                        "name": "首批部件",
                        "type": "component",
                        "evidence_ids": [evidence[0][0]],
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            },
            ensure_ascii=False,
        )


def test_a_batch_that_keeps_failing_fails_the_whole_document(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 second batch fails"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = content_doc_id(content)
    llm = SecondBatchFailsLLM()
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()
    state_store = FakeStateStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=llm,
            parser=FakeParser(
                [
                    ParsedPage(
                        number=1,
                        text=patent_text(SEGMENT_COUNT),
                        bbox=PAGE_BBOX,
                    )
                ]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=vector_store,
            state_store=state_store,
        ),
    )

    document = report.documents[0]
    assert report.failed == 1
    assert document.status == "failed"
    assert document.error is not None
    assert "第 2/2 批" in document.error
    assert document.evidence_segments == 0

    # 第一批成功一次，第二批重试 3 次，全部记入调用成本与尝试次数。
    assert len(report.model_calls) == 4
    assert document.extraction_attempts == 4

    # 整篇失败：既不留没有报告的半份图谱，也不写向量。
    assert graph_store.batches == []
    assert vector_store.vectors == {}
    state = state_store.states[doc_id]
    assert state.status is DocumentStatus.FAILED
    assert state.reason is not None and "第 2/2 批" in state.reason


class RepeatedBatchFakeLLM:
    """每一批都返回同一对概念和同一条关系，只引用本批证据。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def complete(self, prompt: str) -> str:
        evidence = evidence_lines_from_prompt(prompt)
        self.calls.append(evidence)
        evidence_id = evidence[0][0]
        return json.dumps(
            {
                "entities": [
                    {
                        "name": "跨批共现部件",
                        "type": "component",
                        "evidence_ids": [evidence_id],
                        "confidence": 0.9,
                    },
                    {
                        "name": "跨批共现故障",
                        "type": "fault_mode",
                        "evidence_ids": [evidence_id],
                        "confidence": 0.8,
                    },
                ],
                "relations": [
                    {
                        "head": "跨批共现部件",
                        "type": "has_fault_mode",
                        "tail": "跨批共现故障",
                        "evidence_ids": [evidence_id],
                        "confidence": 0.7,
                    }
                ],
            },
            ensure_ascii=False,
        )


def test_repeated_concepts_and_relations_merge_with_their_evidence(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 repeated extraction"
    (tmp_path / "paper.pdf").write_bytes(content)
    llm = RepeatedBatchFakeLLM()
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    document = report.documents[0]
    assert document.error is None
    assert len(llm.calls) > 1
    assert (document.entities, document.relations) == (2, 1)

    (graph_batch,) = graph_store.batches
    assert len(graph_batch.concepts) == 2
    (relation,) = graph_batch.relations
    expected_evidence = {evidence[0][0] for evidence in llm.calls}
    assert set(relation.provenance.evidence_ids) == expected_evidence
    assert len(relation.provenance.evidence_ids) == len(llm.calls)


class DriftingNameFakeLLM:
    """不同批次用大小写不同的同一术语，模拟模型对同一概念的写法漂移。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def complete(self, prompt: str) -> str:
        evidence = evidence_lines_from_prompt(prompt)
        self.calls.append(evidence)
        index = len(self.calls)
        shared_name = "ABC 征兆" if index == 1 else "abc 征兆"
        unique_name = f"本批故障{index}"
        return json.dumps(
            {
                "entities": [
                    {
                        "name": shared_name,
                        "type": "symptom",
                        "evidence_ids": [evidence[0][0]],
                        "confidence": 0.9,
                    },
                    {
                        "name": unique_name,
                        "type": "fault_mode",
                        "evidence_ids": [evidence[-1][0]],
                        "confidence": 0.8,
                    },
                ],
                "relations": [
                    {
                        "head": unique_name,
                        "type": "indicated_by",
                        "tail": shared_name,
                        "evidence_ids": [evidence[-1][0]],
                        "confidence": 0.7,
                    }
                ],
            },
            ensure_ascii=False,
        )


def test_name_spelling_drift_across_batches_does_not_fail_the_document(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 drifting names"
    (tmp_path / "paper.pdf").write_bytes(content)
    llm = DriftingNameFakeLLM()
    graph_store = FakeGraphStore()

    report = ingest(tmp_path, deps=deps_for(llm, graph_store))

    document = report.documents[0]
    assert document.error is None
    assert len(llm.calls) > 1
    assert (document.entities, document.relations) == (3, 2)

    (graph_batch,) = graph_store.batches
    assert {concept.name for concept in graph_batch.concepts} == {
        "ABC 征兆",
        "本批故障1",
        "本批故障2",
    }


def test_changing_the_batch_limits_invalidates_completed_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"%PDF-1.4 batch fingerprint"
    (tmp_path / "paper.pdf").write_bytes(content)
    text = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
    state_store = FakeStateStore()
    llm = FakeLLM('{"entities": [], "relations": []}')
    deps = IngestDeps(
        llm=llm,
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=PAGE_BBOX)]
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

    # 分批上限是抽取依据的一部分：改上限后旧的单批结果不得继续被跳过。
    monkeypatch.setattr("idr_rotating.prompts.MAX_BATCH_SEGMENTS", 20)
    third = ingest(tmp_path, deps=deps)

    assert third.documents[0].skipped is False
    assert len(llm.prompts) == 2

"""幂等重跑与断点续跑：只通过 ingest() 观察外部行为，状态用内存假实现。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from idr_core.state import DocumentStatus
from idr_rotating import IngestDeps, IngestError, ingest

from fakes import (
    EMPTY_EXTRACTION,
    FakeEmbedder,
    FakeGraphStore,
    FakeParser,
    FakeStateStore,
    FakeVectorStore,
    ScriptedLLM,
    parsed_page,
)

TEXT = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"


def content_doc_id(content: bytes) -> str:
    """文献标识由内容哈希推导，与文件名无关。"""
    return hashlib.sha256(content).hexdigest()[:16]


def write_pdf(directory: Path, name: str) -> bytes:
    """写入一篇假 PDF，返回内容字节；文件内容不同则内容哈希不同。"""
    content = f"%PDF-1.4 {name}".encode()
    (directory / name).write_bytes(content)
    return content


def make_deps(
    *,
    llm: ScriptedLLM,
    state_store: FakeStateStore,
    graph_store: FakeGraphStore | None = None,
    parser: FakeParser | None = None,
) -> IngestDeps:
    return IngestDeps(
        llm=llm,
        parser=parser or FakeParser([parsed_page(1, TEXT)]),
        graph_store=graph_store or FakeGraphStore(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        state_store=state_store,
    )


def test_second_run_skips_completed_documents_and_writes_nothing(tmp_path: Path) -> None:
    write_pdf(tmp_path, "a.pdf")
    write_pdf(tmp_path, "b.pdf")
    llm = ScriptedLLM([EMPTY_EXTRACTION, EMPTY_EXTRACTION])
    state_store = FakeStateStore()
    graph_store = FakeGraphStore()
    deps = make_deps(llm=llm, state_store=state_store, graph_store=graph_store)

    first = ingest(tmp_path, deps=deps)
    second = ingest(tmp_path, deps=deps)

    assert [document.skipped for document in first.documents] == [False, False]
    assert len(llm.prompts) == 2
    assert len(graph_store.batches) == 2

    assert [document.skipped for document in second.documents] == [True, True]
    assert second.skipped == 2
    assert len(llm.prompts) == 2, "第二次运行不应再调用模型"
    assert len(graph_store.batches) == 2, "第二次运行不应再写图谱"
    assert {state.status for state in state_store.states.values()} == {
        DocumentStatus.COMPLETED
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.pdf", "b.pdf"]


def test_same_file_name_with_different_content_is_processed_again(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 original content")
    llm = ScriptedLLM([EMPTY_EXTRACTION, EMPTY_EXTRACTION])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    first = ingest(tmp_path, deps=deps)
    paper.write_bytes(b"%PDF-1.4 revised content")
    second = ingest(tmp_path, deps=deps)

    assert first.documents[0].skipped is False
    assert second.documents[0].skipped is False
    assert first.documents[0].doc_id != second.documents[0].doc_id
    assert len(llm.prompts) == 2
    assert len(state_store.states) == 2, "内容不同应按两个 doc_id 各记一行状态"


def test_failed_document_is_retried_on_the_next_run(tmp_path: Path) -> None:
    content = write_pdf(tmp_path, "paper.pdf")
    llm = ScriptedLLM([ValueError("模型输出不合规"), EMPTY_EXTRACTION])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    first = ingest(tmp_path, deps=deps)
    assert first.documents[0].error is not None
    assert state_store.states[content_doc_id(content)].status == DocumentStatus.FAILED

    second = ingest(tmp_path, deps=deps)
    assert second.documents[0].skipped is False
    assert second.documents[0].error is None
    assert state_store.states[content_doc_id(content)].status == (
        DocumentStatus.COMPLETED
    )


def test_interrupted_run_resumes_with_only_the_unfinished_documents(
    tmp_path: Path,
) -> None:
    first_content = write_pdf(tmp_path, "a.pdf")
    second_content = write_pdf(tmp_path, "b.pdf")
    llm = ScriptedLLM([EMPTY_EXTRACTION, KeyboardInterrupt()])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    with pytest.raises(KeyboardInterrupt):
        ingest(tmp_path, deps=deps)

    assert state_store.states[content_doc_id(first_content)].status == (
        DocumentStatus.COMPLETED
    )
    assert state_store.states[content_doc_id(second_content)].status == (
        DocumentStatus.PROCESSING
    )

    resumed_llm = ScriptedLLM([EMPTY_EXTRACTION])
    resumed = ingest(tmp_path, deps=replace(deps, llm=resumed_llm))

    assert [
        (document.path.name, document.skipped) for document in resumed.documents
    ] == [("a.pdf", True), ("b.pdf", False)]
    assert len(resumed_llm.prompts) == 1, "只应补做未完成的那一篇"
    assert state_store.states[content_doc_id(second_content)].status == (
        DocumentStatus.COMPLETED
    )


def test_limit_processes_only_the_first_documents_of_the_corpus(
    tmp_path: Path,
) -> None:
    write_pdf(tmp_path, "a.pdf")
    write_pdf(tmp_path, "b.pdf")
    third_content = write_pdf(tmp_path, "c.pdf")
    llm = ScriptedLLM([EMPTY_EXTRACTION, EMPTY_EXTRACTION, EMPTY_EXTRACTION])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    first = ingest(tmp_path, deps=deps, limit=2)
    assert [document.path.name for document in first.documents] == ["a.pdf", "b.pdf"]
    assert len(llm.prompts) == 2

    second = ingest(tmp_path, deps=deps, limit=2)
    assert [document.path.name for document in second.documents] == ["a.pdf", "b.pdf"]
    assert [document.skipped for document in second.documents] == [True, True]
    assert content_doc_id(third_content) not in state_store.states
    assert len(llm.prompts) == 2

    third = ingest(tmp_path, deps=deps)
    assert [
        (document.path.name, document.skipped) for document in third.documents
    ] == [("a.pdf", True), ("b.pdf", True), ("c.pdf", False)]


def test_quarantined_document_is_skipped_but_the_reason_stays_visible(
    tmp_path: Path,
) -> None:
    content = write_pdf(tmp_path, "scan.pdf")
    parser = FakeParser([parsed_page(1, "")])
    llm = ScriptedLLM([])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store, parser=parser)

    first = ingest(tmp_path, deps=deps)
    reason = first.documents[0].quarantine_reason
    assert reason is not None
    assert first.documents[0].skipped is False

    second = ingest(tmp_path, deps=deps)

    assert second.documents[0].skipped is True
    assert second.documents[0].quarantine_reason == reason
    assert second.documents[0].status == "quarantined"
    assert second.documents[0].prior_status is DocumentStatus.QUARANTINED
    assert (second.skipped, second.quarantined) == (0, 1), (
        "此前隔离的文献仍应计入隔离，而不是混进跳过"
    )
    assert len(parser.calls) == 1, "第二次运行不应重复解析已隔离的文献"
    assert len(llm.prompts) == 0
    assert state_store.states[content_doc_id(content)].status == (
        DocumentStatus.QUARANTINED
    )


def test_limit_must_be_positive(tmp_path: Path) -> None:
    write_pdf(tmp_path, "a.pdf")
    deps = make_deps(llm=ScriptedLLM([]), state_store=FakeStateStore())

    with pytest.raises(IngestError):
        ingest(tmp_path, deps=deps, limit=0)


def test_ontology_or_prompt_change_invalidates_completed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ontology_path = tmp_path / "ontology.yaml"

    def base_ontology() -> str:
        return (
            "version: 0.1.0\n"
            "domain: alpha\n"
            "entity_types:\n"
            "  entity_type_a:\n"
            "    label: 甲类实体\n"
            "relation_types: {}\n"
        )

    ontology_path.write_text(base_ontology(), encoding="utf-8")
    monkeypatch.setattr(
        "idr_rotating.ontology.DEFAULT_ONTOLOGY_PATH", ontology_path
    )
    content = write_pdf(tmp_path, "a.pdf")
    llm = ScriptedLLM([EMPTY_EXTRACTION, EMPTY_EXTRACTION])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    first = ingest(tmp_path, deps=deps)
    assert first.documents[0].skipped is False
    recorded = state_store.states[content_doc_id(content)]
    assert recorded.fingerprint is not None
    assert len(recorded.fingerprint) == 64

    second = ingest(tmp_path, deps=deps)
    assert second.documents[0].skipped is True
    assert len(llm.prompts) == 1, "提示词与本体都没变时不应重新抽取"

    # 本体文件变化（哪怕只是注释）意味着抽取依据变了，已完成的文献要重做。
    ontology_path.write_text(
        base_ontology() + "\n# 本体版本变化\n", encoding="utf-8"
    )
    third = ingest(tmp_path, deps=deps)

    assert third.documents[0].skipped is False
    assert len(llm.prompts) == 2
    assert state_store.states[content_doc_id(content)].fingerprint != (
        recorded.fingerprint
    )


def test_skipped_documents_carry_their_previously_recorded_counts(
    tmp_path: Path,
) -> None:
    content = write_pdf(tmp_path, "paper.pdf")
    doc_id = content_doc_id(content)
    segment_id = f"ev:{doc_id}:1:0-{len(TEXT)}"
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "quote": TEXT,
                    "confidence": 0.9,
                },
                {
                    "name": "BPFO 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "quote": TEXT,
                    "confidence": 0.8,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "BPFO 倍频",
                    "evidence_ids": [segment_id],
                    "quote": TEXT,
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    llm = ScriptedLLM([response])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    first = ingest(tmp_path, deps=deps)
    second = ingest(tmp_path, deps=deps)

    assert (first.documents[0].evidence_segments, first.documents[0].entities,
            first.documents[0].relations) == (1, 2, 1)

    document = second.documents[0]
    assert document.skipped is True
    assert document.status == "skipped"
    assert document.prior_status is DocumentStatus.COMPLETED
    assert (document.evidence_segments, document.entities, document.relations) == (
        1,
        2,
        1,
    ), "跳过的文献必须在报告里带回此前的计数"
    assert second.evidence_segments == 1


def test_extraction_contract_change_invalidates_completed_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """图谱身份与边归属规则也是抽取契约：契约版本变了就要重抽。"""
    write_pdf(tmp_path, "paper.pdf")
    llm = ScriptedLLM([EMPTY_EXTRACTION, EMPTY_EXTRACTION])
    state_store = FakeStateStore()
    deps = make_deps(llm=llm, state_store=state_store)

    first = ingest(tmp_path, deps=deps)
    second = ingest(tmp_path, deps=deps)
    assert first.documents[0].skipped is False
    assert second.documents[0].skipped is True
    assert len(llm.prompts) == 1

    # 模拟旧契约下完成的状态：当前契约版本参与指纹，必须触发重抽。
    monkeypatch.setattr(
        "idr_rotating.prompts.EXTRACTION_CONTRACT_VERSION", "previous"
    )
    third = ingest(tmp_path, deps=deps)

    assert third.documents[0].skipped is False
    assert len(llm.prompts) == 2

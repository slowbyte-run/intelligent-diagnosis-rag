"""入库门面的行为测试：只通过 ingest() 观察外部结果。"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from idr_core.embedding import EMBEDDING_DIM
from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedPage
from idr_core.vector_store import VectorDimensionError
from idr_rotating import IngestDeps, IngestReport, OntologyError, ingest
from idr_rotating.extraction import (
    ExtractedRelation,
    ValidatedExtraction,
)

from fakes import (
    FailingVectorStore,
    FakeEmbedder,
    FakeGraphStore,
    FakeLLM,
    FakeParser,
    FakeVectorStore,
)


def content_doc_id(content: bytes) -> str:
    """文献标识由内容哈希推导，与文件名无关。"""
    return hashlib.sha256(content).hexdigest()[:16]


def evidence_id(doc_id: str, page: int, text: str) -> str:
    """证据段标识由文献标识、页码与字符区间稳定推导。"""
    return f"ev:{doc_id}:{page}:0-{len(text)}"


def write_pdf(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def test_single_document_ingest_writes_traceable_concepts(tmp_path: Path) -> None:
    content = b"%PDF-1.4 fake paper"
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
                    "quote": text,
                    "confidence": 0.91,
                },
                {
                    "name": "BPFO 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.88,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "BPFO 倍频",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(72.0, 72.0, 540.0, 720.0))]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    report = ingest(tmp_path, deps=deps)

    assert isinstance(report, IngestReport)
    assert len(report.documents) == 1
    document = report.documents[0]
    assert document.doc_id == doc_id
    assert (document.evidence_segments, document.entities, document.relations) == (
        1,
        2,
        1,
    )
    assert document.error is None

    batch = graph_store.batches[0]
    assert [(segment.id, segment.page, segment.charspan, segment.bbox) for segment in batch.evidence] == [
        (segment_id, 1, CharSpan(0, len(text)), BBox(72.0, 72.0, 540.0, 720.0))
    ]

    concepts_by_id = {concept.id: concept for concept in batch.concepts}
    assert {concept.name for concept in batch.concepts} == {
        "轴承外圈疲劳剥落",
        "外圈故障频率及其倍频",
    }
    assert {concept.type for concept in batch.concepts} == {"fault_mode", "symptom"}

    assert len(batch.evidence_links) == 2
    link_confidences = {
        (concepts_by_id[link.concept_id].name, link.provenance.confidence)
        for link in batch.evidence_links
    }
    assert link_confidences == {
        ("轴承外圈疲劳剥落", 0.91),
        ("外圈故障频率及其倍频", 0.88),
    }
    assert all(link.evidence_id == segment_id for link in batch.evidence_links)
    assert all(
        link.provenance.evidence_ids == (segment_id,)
        for link in batch.evidence_links
    )
    assert all(link.provenance.source == "extracted" for link in batch.evidence_links)

    (relation,) = batch.relations
    assert relation.type == "indicated_by"
    assert relation.provenance.evidence_ids == (segment_id,)
    assert relation.provenance.confidence == 0.8
    assert relation.provenance.source == "extracted"
    assert concepts_by_id[relation.head_id].name == "轴承外圈疲劳剥落"
    assert concepts_by_id[relation.tail_id].name == "外圈故障频率及其倍频"


def test_extractions_that_fail_ontology_validation_are_not_written(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 fake paper with invalid extractions"
    write_pdf(tmp_path, "mixed.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为外圈故障频率；滚动轴承是部件。"
    segment_id = evidence_id(doc_id, 1, text)

    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
                {
                    "name": "外圈故障频率",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
                {
                    "name": "滚动轴承",
                    "type": "component",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
                {
                    "name": "神秘实体",
                    "type": "alien_type",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
                {
                    "name": "无出处实体",
                    "type": "fault_mode",
                    "evidence_ids": [],
                    "confidence": 0.9,
                },
                {
                    "name": "假证据实体",
                    "type": "fault_mode",
                    "evidence_ids": ["ev:does-not-exist"],
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "外圈故障频率",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "magic_link",
                    "tail": "外圈故障频率",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "has_component",
                    "tail": "滚动轴承",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "滚动轴承",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
                {
                    "head": "外圈剥落",
                    "type": "indicated_by",
                    "tail": "不存在",
                    "evidence_ids": [segment_id],
                    "confidence": 0.8,
                },
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert (document.entities, document.relations) == (3, 1)
    assert (document.rejected_entities, document.rejected_relations) == (3, 4)

    batch = graph_store.batches[0]
    assert {concept.name for concept in batch.concepts} == {
        "轴承外圈疲劳剥落",
        "外圈故障频率",
        "滚动轴承",
    }
    assert [relation.type for relation in batch.relations] == ["indicated_by"]
    assert len(batch.evidence_links) == 3


def test_reprocessing_same_document_yields_same_identifiers(tmp_path: Path) -> None:
    content = b"%PDF-1.4 stable identifiers"
    write_pdf(tmp_path, "stable.pdf", content)
    doc_id = content_doc_id(content)
    text = "内圈剥落表现为 BPFI 倍频，并伴随转频边带。"
    segment_id = evidence_id(doc_id, 1, text)
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "内圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
                {
                    "name": "BPFI 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "内圈剥落",
                    "type": "indicated_by",
                    "tail": "BPFI 倍频",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.8,
                }
            ],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    deps = IngestDeps(
        llm=FakeLLM(response),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    ingest(tmp_path, deps=deps)
    ingest(tmp_path, deps=deps)

    first, second = graph_store.batches
    assert [segment.id for segment in first.evidence] == [
        segment.id for segment in second.evidence
    ]
    assert [concept.id for concept in first.concepts] == [
        concept.id for concept in second.concepts
    ]
    assert first.relations == second.relations


def test_limit_processes_only_the_first_documents(tmp_path: Path) -> None:
    write_pdf(tmp_path, "a.pdf", b"%PDF-1.4 first")
    write_pdf(tmp_path, "b.pdf", b"%PDF-1.4 second")
    text = "外圈剥落。"
    doc_id = content_doc_id(b"%PDF-1.4 first")
    segment_id = evidence_id(doc_id, 1, text)
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=FakeGraphStore(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
        limit=1,
    )

    assert [document.path.name for document in report.documents] == ["a.pdf"]


def test_every_evidence_segment_gets_a_vector_keyed_by_its_id(tmp_path: Path) -> None:
    content = b"%PDF-1.4 fake two page paper"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    first_text = "外圈剥落表现为 BPFO 倍频。"
    second_text = "润滑不足会加速疲劳剥落。"
    first_id = evidence_id(doc_id, 1, first_text)
    second_id = evidence_id(doc_id, 2, second_text)
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [first_id],
                    "quote": first_text,
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [
                    ParsedPage(
                        number=1, text=first_text, bbox=BBox(0.0, 0.0, 595.0, 842.0)
                    ),
                    ParsedPage(
                        number=2, text=second_text, bbox=BBox(0.0, 0.0, 595.0, 842.0)
                    ),
                ]
            ),
            graph_store=FakeGraphStore(),
            embedder=embedder,
            vector_store=vector_store,
        ),
    )

    assert report.documents[0].error is None
    assert embedder.texts == [first_text, second_text]
    assert set(vector_store.vectors) == {first_id, second_id}
    assert vector_store.get_evidence_vector(first_id) == [
        float(len(first_text))
    ] * EMBEDDING_DIM
    assert vector_store.get_evidence_vector(second_id) == [
        float(len(second_text))
    ] * EMBEDDING_DIM


def test_vector_write_failure_does_not_write_graph_data(tmp_path: Path) -> None:
    content = b"%PDF-1.4 fake paper whose vectors fail"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频，能量集中在高频段。"
    segment_id = evidence_id(doc_id, 1, text)
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    graph_store = FakeGraphStore()
    vector_store = FailingVectorStore(VectorDimensionError("维度与服务端约定不符"))

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM(response),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=vector_store,
        ),
    )

    document = report.documents[0]
    assert document.error is not None
    assert "VectorDimensionError" in document.error
    assert graph_store.batches == []


def test_malformed_ontology_fails_before_any_document_is_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(tmp_path, "paper.pdf", b"%PDF-1.4 fake paper")
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        "version: 0.1.0\ndomain: rotating\n"
        "entity_types:\n  fault_mode:\n    description: 缺少 label\n"
        "relation_types: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "idr_rotating.ontology.DEFAULT_ONTOLOGY_PATH", broken
    )
    llm = FakeLLM('{"entities": [], "relations": []}')
    graph_store = FakeGraphStore()

    with pytest.raises(OntologyError):
        ingest(
            tmp_path,
            deps=IngestDeps(
                llm=llm,
                parser=FakeParser([]),
                graph_store=graph_store,
                embedder=FakeEmbedder(),
                vector_store=FakeVectorStore(),
            ),
        )

    assert llm.prompts == []
    assert graph_store.batches == []


def test_concept_node_ids_follow_concept_identity_not_extraction_order(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 concept identity"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动，新概念甲也出现了。"
    segment_id = evidence_id(doc_id, 1, text)

    def entity(name: str, type_: str, confidence: float) -> dict[str, object]:
        return {
            "name": name,
            "type": type_,
            "evidence_ids": [segment_id],
            "quote": text,
            "confidence": confidence,
        }

    entities = [
        entity("外圈剥落", "fault_mode", 0.9),
        entity("BPFO 倍频", "symptom", 0.8),
        entity("新概念甲", "symptom", 0.7),
    ]
    responses = [
        json.dumps({"entities": entities, "relations": []}, ensure_ascii=False),
        json.dumps(
            {"entities": list(reversed(entities)), "relations": []},
            ensure_ascii=False,
        ),
    ]
    graph_store = FakeGraphStore()
    deps = IngestDeps(
        llm=FakeLLM(responses),
        parser=FakeParser(
            [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
        ),
        graph_store=graph_store,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    ingest(tmp_path, deps=deps)
    ingest(tmp_path, deps=deps)

    first_ids = {concept.name: concept.id for concept in graph_store.batches[0].concepts}
    second_ids = {
        concept.name: concept.id for concept in graph_store.batches[1].concepts
    }
    assert first_ids == second_ids, "抽取顺序变化不得改变概念节点 ID"

    # 规范实例用跨文献共享的本体 ID；自由词限定在本文献内。
    assert first_ids["轴承外圈疲劳剥落"] == (
        "concept:bearing_outer_race_fatigue_spalling"
    )
    assert first_ids["新概念甲"] == f"{doc_id}:concept:symptom:新概念甲"


def test_relation_endpoint_missing_from_validated_entities_fails_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校验阶段放行了不存在的端点时，入库必须报数据错误而不是静默丢边。"""
    content = b"%PDF-1.4 dangling relation"
    write_pdf(tmp_path, "paper.pdf", content)
    doc_id = content_doc_id(content)
    text = "外圈剥落表现为 BPFO 倍频振动，能量集中在高频段。"
    segment_id = evidence_id(doc_id, 1, text)
    relation = ExtractedRelation(
        head="幽灵甲",
        type="indicated_by",
        tail="幽灵乙",
        evidence_ids=[segment_id],
        confidence=0.9,
    )

    def dangling_validation(*args: object, **kwargs: object) -> ValidatedExtraction:
        return ValidatedExtraction(
            entities=(),
            relations=(relation,),
            rejected_entities=0,
            rejected_relations=0,
        )

    ingest_module = importlib.import_module("idr_rotating.ingest")
    monkeypatch.setattr(ingest_module, "validate_extraction", dangling_validation)
    graph_store = FakeGraphStore()

    report = ingest(
        tmp_path,
        deps=IngestDeps(
            llm=FakeLLM('{"entities": [], "relations": []}'),
            parser=FakeParser(
                [ParsedPage(number=1, text=text, bbox=BBox(0.0, 0.0, 595.0, 842.0))]
            ),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )

    document = report.documents[0]
    assert document.error is not None, "端点缺失是数据错误，不是可以忽略的分支"
    assert "幽灵甲" in document.error
    assert "幽灵乙" in document.error
    assert report.failed == 1
    assert graph_store.batches == []


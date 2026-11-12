"""别名归一的端到端行为：图谱里同一概念只留一个节点。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_rotating import IngestDeps, ingest

from fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeParser, FakeVectorStore


def test_synonyms_collapse_into_one_canonical_concept_in_the_graph(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 alias normalisation fixture"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    text = "外套圈剥落与外圈剥落都会产生 BPFO 倍频。"
    segment_id = f"ev:{doc_id}:1:0-{len(text)}"
    response = json.dumps(
        {
            "entities": [
                {
                    "name": "外套圈剥落",
                    "type": "fault_mode",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
                {
                    "name": "外圈剥落",
                    "type": "故障模式",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.8,
                },
                {
                    "name": "BPFO 倍频",
                    "type": "symptom",
                    "evidence_ids": [segment_id],
                    "quote": text,
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "外套圈剥落",
                    "type": "表现为",
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
    assert (document.entities, document.relations) == (2, 1)

    batch = graph_store.batches[0]
    fault_modes = [
        concept for concept in batch.concepts if concept.type == "fault_mode"
    ]
    assert [concept.name for concept in fault_modes] == ["轴承外圈疲劳剥落"]
    assert {concept.name for concept in batch.concepts} == {
        "轴承外圈疲劳剥落",
        "外圈故障频率及其倍频",
    }

    (relation,) = batch.relations
    concepts_by_id = {concept.id: concept for concept in batch.concepts}
    assert relation.type == "indicated_by"
    assert concepts_by_id[relation.head_id].name == "轴承外圈疲劳剥落"
    assert concepts_by_id[relation.tail_id].name == "外圈故障频率及其倍频"

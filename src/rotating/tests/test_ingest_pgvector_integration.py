"""入库写向量的集成契约：真实 Postgres，假模型、假解析器与假图存储。

默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import psycopg
import pytest

from idr_core.embedding import EMBEDDING_DIM
from idr_core.models import BBox
from idr_core.parsing import ParsedPage
from idr_core.vector_store.pgvector import PgVectorStore
from idr_rotating import IngestDeps, ingest

from fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeParser

DATABASE_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="需要 DATABASE_URL（先 source .env）",
    ),
]


def test_ingest_writes_one_vector_per_evidence_segment(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4 integration two pages"
    (tmp_path / "paper.pdf").write_bytes(content)
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    first_text = "外圈剥落表现为 BPFO 倍频。"
    second_text = "润滑不足会加速疲劳剥落。"
    first_id = f"ev:{doc_id}:1:0-{len(first_text)}"
    second_id = f"ev:{doc_id}:2:0-{len(second_text)}"
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
    store = PgVectorStore(DATABASE_URL)

    try:
        report = ingest(
            tmp_path,
            deps=IngestDeps(
                llm=FakeLLM(response),
                parser=FakeParser(
                    [
                        ParsedPage(
                            number=1,
                            text=first_text,
                            bbox=BBox(0.0, 0.0, 595.0, 842.0),
                        ),
                        ParsedPage(
                            number=2,
                            text=second_text,
                            bbox=BBox(0.0, 0.0, 595.0, 842.0),
                        ),
                    ]
                ),
                graph_store=FakeGraphStore(),
                embedder=FakeEmbedder(),
                vector_store=store,
            ),
        )

        assert report.documents[0].error is None
        for evidence_id, text in ((first_id, first_text), (second_id, second_text)):
            vector = store.get_evidence_vector(evidence_id)
            assert vector is not None
            assert len(vector) == EMBEDDING_DIM
            assert vector[0] == float(len(text))
    finally:
        with psycopg.connect(DATABASE_URL) as connection:
            table = connection.execute(
                "SELECT to_regclass('evidence_vectors')"
            ).fetchone()
            if table is not None and table[0] is not None:
                connection.execute(
                    "DELETE FROM evidence_vectors WHERE evidence_id = ANY(%s)",
                    ([first_id, second_id],),
                )

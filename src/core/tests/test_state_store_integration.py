"""文档处理状态存储的集成契约测试。

需要 docker compose 起的真实 Postgres，且已 source .env；默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest

from idr_core.state import DocumentProcessingState, DocumentStatus
from idr_core.state.postgres import PostgresStateStore

DATABASE_URL = os.environ.get("DATABASE_URL", "")
TABLE = "document_ingest_states"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="需要 DATABASE_URL（先 source .env）",
    ),
]


@pytest.fixture
def store() -> PostgresStateStore:
    return PostgresStateStore(DATABASE_URL)


@pytest.fixture
def doc_ids() -> Iterator[list[str]]:
    identifiers = [f"it-t5:{uuid4().hex}:{index}" for index in range(2)]
    yield identifiers
    with psycopg.connect(DATABASE_URL) as connection:
        table = connection.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()
        if table is not None and table[0] is not None:
            connection.execute(
                f"DELETE FROM {TABLE} WHERE doc_id = ANY(%s)", (identifiers,)
            )


def test_unknown_document_has_no_state(
    store: PostgresStateStore, doc_ids: list[str]
) -> None:
    assert store.get(doc_ids[0]) is None


def test_completed_state_survives_a_new_store_instance(
    store: PostgresStateStore, doc_ids: list[str]
) -> None:
    completed = DocumentProcessingState(
        doc_id=doc_ids[0],
        status=DocumentStatus.COMPLETED,
        evidence_segments=12,
        entities=7,
        relations=4,
    )

    store.record(completed)

    assert store.get(doc_ids[0]) == completed
    # 新实例模拟新进程：状态在 Postgres 里，不在进程内存或数据目录。
    assert PostgresStateStore(DATABASE_URL).get(doc_ids[0]) == completed


def test_recording_again_updates_the_same_row(
    store: PostgresStateStore, doc_ids: list[str]
) -> None:
    store.record(
        DocumentProcessingState(doc_id=doc_ids[0], status=DocumentStatus.PROCESSING)
    )
    failed = DocumentProcessingState(
        doc_id=doc_ids[0], status=DocumentStatus.FAILED, reason="模型输出不合规"
    )

    store.record(failed)

    assert store.get(doc_ids[0]) == failed
    with psycopg.connect(DATABASE_URL) as connection:
        row = connection.execute(
            f"SELECT count(*) FROM {TABLE} WHERE doc_id = %s", (doc_ids[0],)
        ).fetchone()
    assert row == (1,)


def test_states_are_isolated_by_document_id(
    store: PostgresStateStore, doc_ids: list[str]
) -> None:
    store.record(
        DocumentProcessingState(
            doc_id=doc_ids[0],
            status=DocumentStatus.COMPLETED,
            evidence_segments=3,
            entities=2,
            relations=1,
        )
    )
    store.record(
        DocumentProcessingState(
            doc_id=doc_ids[1],
            status=DocumentStatus.QUARANTINED,
            reason="扫描件：未检测到可用的文字层",
        )
    )

    completed = store.get(doc_ids[0])
    quarantined = store.get(doc_ids[1])

    assert completed is not None
    assert completed.status == DocumentStatus.COMPLETED
    assert (completed.evidence_segments, completed.entities, completed.relations) == (
        3,
        2,
        1,
    )
    assert quarantined is not None
    assert quarantined.status == DocumentStatus.QUARANTINED
    assert quarantined.reason == "扫描件：未检测到可用的文字层"

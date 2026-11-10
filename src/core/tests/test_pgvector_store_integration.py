"""pgvector 证据段向量适配器的集成契约测试。

需要 docker compose 起的真实 Postgres（已装 vector 扩展），且已 source .env；
默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest

from idr_core.embedding import EMBEDDING_DIM
from idr_core.vector_store import VectorDimensionError
from idr_core.vector_store.pgvector import PgVectorStore

DATABASE_URL = os.environ.get("DATABASE_URL", "")
TABLE = "evidence_vectors"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="需要 DATABASE_URL（先 source .env）",
    ),
]


def deterministic_vector(seed: int) -> list[float]:
    """取值都可用 float4 精确表示，读回后可以逐位比较。"""
    return [((seed + index) % 9) / 8.0 for index in range(EMBEDDING_DIM)]


@pytest.fixture
def store() -> PgVectorStore:
    return PgVectorStore(DATABASE_URL)


@pytest.fixture
def evidence_ids() -> Iterator[list[str]]:
    ids = [f"it-t6:{uuid4().hex}:{index}" for index in range(3)]
    yield ids
    with psycopg.connect(DATABASE_URL) as connection:
        table = connection.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()
        if table is not None and table[0] is not None:
            connection.execute(
                f"DELETE FROM {TABLE} WHERE evidence_id = ANY(%s)", (ids,)
            )


def test_vectors_written_by_evidence_id_can_be_read_back(
    store: PgVectorStore, evidence_ids: list[str]
) -> None:
    vectors = [deterministic_vector(index) for index in range(len(evidence_ids))]

    store.upsert_evidence_vectors(evidence_ids=evidence_ids, vectors=vectors)

    for evidence_id, expected in zip(evidence_ids, vectors, strict=True):
        actual = store.get_evidence_vector(evidence_id)
        assert actual is not None
        assert len(actual) == EMBEDDING_DIM
        assert actual == pytest.approx(expected)
    assert store.get_evidence_vector("it-t6:does-not-exist") is None


def test_dimension_mismatch_fails_explicitly_and_writes_nothing(
    store: PgVectorStore, evidence_ids: list[str]
) -> None:
    wrong_dimension = [0.5] * (EMBEDDING_DIM - 1)

    with pytest.raises(VectorDimensionError) as error:
        store.upsert_evidence_vectors(
            evidence_ids=evidence_ids[:2],
            vectors=[deterministic_vector(0), wrong_dimension],
        )

    assert str(EMBEDDING_DIM) in str(error.value)
    assert str(len(wrong_dimension)) in str(error.value)
    for evidence_id in evidence_ids[:2]:
        assert store.get_evidence_vector(evidence_id) is None


def test_rewriting_the_same_evidence_ids_keeps_one_row_each(
    store: PgVectorStore, evidence_ids: list[str]
) -> None:
    first = [deterministic_vector(0) for _ in evidence_ids]
    second = [deterministic_vector(1) for _ in evidence_ids]

    store.upsert_evidence_vectors(evidence_ids=evidence_ids, vectors=first)
    store.upsert_evidence_vectors(evidence_ids=evidence_ids, vectors=second)

    for evidence_id in evidence_ids:
        actual = store.get_evidence_vector(evidence_id)
        assert actual is not None
        assert actual == pytest.approx(second[0])

    with psycopg.connect(DATABASE_URL) as connection:
        row = connection.execute(
            f"SELECT count(*) FROM {TABLE} WHERE evidence_id = ANY(%s)",
            (evidence_ids,),
        ).fetchone()

    assert row is not None
    assert row[0] == len(evidence_ids)


def test_table_stores_vectors_at_the_fixed_dimension(store: PgVectorStore) -> None:
    store.get_evidence_vector("it-t6:schema-probe")

    with psycopg.connect(DATABASE_URL) as connection:
        row = connection.execute(
            """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = 'embedding'
            """,
            (TABLE,),
        ).fetchone()

    assert row == (f"vector({EMBEDDING_DIM})",)

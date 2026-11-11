"""pgvector 证据段向量存储适配器。

表 ``evidence_vectors`` 以证据段标识为主键，向量列固定 ``vector(1024)``
（ADR-0002）。写入使用 ``ON CONFLICT`` 覆盖，重复处理同一证据段不会产生
重复行；整批写入在一个事务里完成。维度不符在发出 SQL 之前就明确失败，
不截断、不补零。

召回同样走这张表：``<=>`` 是余弦距离，``ORDER BY`` 加 ``LIMIT`` 就是顺序
扫描。**不建 HNSW/IVFFlat**：6806 条向量在毫秒级，近似索引的调参与内存成本
现在无法用评估结果证明收益（ADR-0005），超过十万条再评估。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import psycopg

from idr_core.embedding import EMBEDDING_DIM
from idr_core.vector_store import VectorDimensionError, VectorMatch

TABLE_NAME = "evidence_vectors"

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    evidence_id text PRIMARY KEY,
    embedding vector({EMBEDDING_DIM}) NOT NULL
)
"""


class PgVectorStore:
    """把证据段向量写进 Postgres + pgvector。

    每次调用自建连接：入库是单进程命令，连接池留到有并发消费者的里程碑。
    表在第一次使用时按固定维度创建，职责仅限于本适配器，不需要迁移框架。
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._schema_ready = False

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(evidence_ids) != len(vectors):
            raise ValueError(
                f"证据段标识与向量的数量不一致：{len(evidence_ids)} 个标识，"
                f"{len(vectors)} 个向量"
            )

        rows = [
            (evidence_id, _to_pgvector(vector, label=f"证据段 {evidence_id}"))
            for evidence_id, vector in zip(evidence_ids, vectors, strict=True)
        ]
        if not rows:
            return

        with psycopg.connect(self._database_url) as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {TABLE_NAME} (evidence_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (evidence_id)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    rows,
                )

    def search_similar(
        self, vector: Sequence[float], *, limit: int
    ) -> tuple[VectorMatch, ...]:
        """按余弦距离升序取最近的 ``limit`` 条证据段向量（顺序扫描，不建索引）。"""
        if limit < 1:
            raise ValueError(f"召回条数必须是正整数，收到 {limit}")
        literal = _to_pgvector(vector, label="查询向量")
        with psycopg.connect(self._database_url) as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                f"""
                SELECT evidence_id, embedding <=> %s::vector AS distance
                FROM {TABLE_NAME}
                ORDER BY distance
                LIMIT %s
                """,
                (literal, limit),
            ).fetchall()
        return tuple(
            VectorMatch(evidence_id=str(row[0]), distance=float(row[1]))
            for row in rows
        )

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None:
        with psycopg.connect(self._database_url) as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                f"SELECT embedding::text FROM {TABLE_NAME} WHERE evidence_id = %s",
                (evidence_id,),
            ).fetchone()
        if row is None:
            return None
        return [float(value) for value in json.loads(row[0])]

    def _ensure_schema(self, connection: psycopg.Connection[Any]) -> None:
        if self._schema_ready:
            return
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(_CREATE_TABLE)
        self._schema_ready = True


def _to_pgvector(vector: Sequence[float], *, label: str) -> str:
    """把向量转成 pgvector 字面量；维度不符时明确失败，不截断、不补零。"""
    if len(vector) != EMBEDDING_DIM:
        raise VectorDimensionError(
            f"{label}的向量维度为 {len(vector)}，"
            f"服务端约定为 {EMBEDDING_DIM}；拒绝该向量（不截断、不补零）"
        )
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"

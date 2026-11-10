"""Postgres 文档处理状态适配器。

表 ``document_ingest_states`` 以 ``doc_id``（内容哈希）为主键，一行即一篇
文献的处理状态。写入用 ``ON CONFLICT`` 覆盖，重复处理或断点续跑不会产生
重复行；表在第一次使用时创建，职责仅限于本适配器，不引入迁移框架。
"""

from __future__ import annotations

from typing import Any

import psycopg

from idr_core.state import DocumentProcessingState, DocumentStatus

TABLE_NAME = "document_ingest_states"

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    doc_id text PRIMARY KEY,
    status text NOT NULL,
    evidence_segments integer NOT NULL DEFAULT 0,
    entities integer NOT NULL DEFAULT 0,
    relations integer NOT NULL DEFAULT 0,
    reason text,
    updated_at timestamptz NOT NULL DEFAULT now()
)
"""


class PostgresStateStore:
    """把每篇文献的处理状态写进 Postgres。

    每次调用自建连接：入库是单进程命令，连接池留到有并发消费者的里程碑。
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._schema_ready = False

    def get(self, doc_id: str) -> DocumentProcessingState | None:
        with psycopg.connect(self._database_url) as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                f"""
                SELECT doc_id, status, evidence_segments, entities, relations, reason
                FROM {TABLE_NAME}
                WHERE doc_id = %s
                """,
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return DocumentProcessingState(
            doc_id=row[0],
            status=DocumentStatus(row[1]),
            evidence_segments=row[2],
            entities=row[3],
            relations=row[4],
            reason=row[5],
        )

    def record(self, state: DocumentProcessingState) -> None:
        with psycopg.connect(self._database_url) as connection:
            self._ensure_schema(connection)
            connection.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (doc_id, status, evidence_segments, entities, relations,
                     reason, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (doc_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    evidence_segments = EXCLUDED.evidence_segments,
                    entities = EXCLUDED.entities,
                    relations = EXCLUDED.relations,
                    reason = EXCLUDED.reason,
                    updated_at = now()
                """,
                (
                    state.doc_id,
                    state.status.value,
                    state.evidence_segments,
                    state.entities,
                    state.relations,
                    state.reason,
                ),
            )

    def _ensure_schema(self, connection: psycopg.Connection[Any]) -> None:
        if self._schema_ready:
            return
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_TABLE)
        self._schema_ready = True

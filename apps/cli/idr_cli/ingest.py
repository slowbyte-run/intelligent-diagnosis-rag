"""ingest 命令：跑完入库门面并输出每篇文献与合计的计数。"""

from __future__ import annotations

import sys
from pathlib import Path

from idr_core.config import Config
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.llm.embedding import OpenAICompatibleEmbedder
from idr_core.llm.openai_compat import OpenAICompatibleLLM
from idr_core.parsing.pdf import PdfTextParser
from idr_core.vector_store.pgvector import PgVectorStore
from idr_rotating import (
    IngestDeps,
    IngestError,
    OntologyError,
    PendingKind,
    ingest,
)


def run_ingest(
    source_dir: Path, *, deps: IngestDeps, limit: int | None = None
) -> int:
    try:
        report = ingest(source_dir, deps=deps, limit=limit)
    except OntologyError as error:
        print(f"本体错误：{error}", file=sys.stderr)
        return 1
    except IngestError as error:
        print(f"入库失败：{error}", file=sys.stderr)
        return 1

    for document in report.documents:
        if document.error is not None:
            print(f"{document.path.name}：失败（{document.error}）")
        elif document.quarantine_reason is not None:
            print(f"{document.path.name}：已隔离（{document.quarantine_reason}）")
        else:
            rejected = ""
            if document.rejected_entities or document.rejected_relations:
                rejected = (
                    f"（本体拒绝：实体 {document.rejected_entities}，"
                    f"关系 {document.rejected_relations}）"
                )
            print(
                f"{document.path.name}："
                f"证据段 {document.evidence_segments}，"
                f"实体 {document.entities}，"
                f"关系 {document.relations}{rejected}"
            )

    pending = report.pending_reviews
    if pending:
        print("待审队列（本体之外的抽取，需人工确认）：")
        for review in pending:
            kind = (
                "实体类型"
                if review.kind is PendingKind.ENTITY
                else "关系类型"
            )
            print(
                f"- {kind} {review.value}（出现 {review.frequency} 次；"
                f"原因：{review.reason.value}）"
            )
            for example in review.examples:
                print(
                    f"  例句（{example.doc_id} {example.evidence_id}）："
                    f"{example.text}"
                )

    print(
        f"合计：证据段 {report.evidence_segments}，"
        f"实体 {report.entities}，"
        f"关系 {report.relations}，"
        f"隔离 {report.quarantined}"
    )
    return 0 if report.failed == 0 else 1


def build_deps(config: Config) -> IngestDeps:
    """真实依赖：模型客户端、pdfplumber 解析器、Neo4j、嵌入服务与 pgvector。

    嵌入维度取自 EMBEDDING_DIM；向量列由适配器固定为 ADR-0002 的 1024 维，
    配置漂移会在写入时明确失败，而不是静默截断。
    """
    return IngestDeps(
        llm=OpenAICompatibleLLM(
            base_url=config.deepseek_base_url,
            api_key=config.deepseek_api_key,
            model=config.deepseek_model,
        ),
        parser=PdfTextParser(),
        graph_store=Neo4jGraphStore(
            config.neo4j_uri, config.neo4j_user, config.neo4j_password
        ),
        embedder=OpenAICompatibleEmbedder(
            base_url=config.siliconflow_base_url,
            api_key=config.siliconflow_api_key,
            model=config.embedding_model,
            dimensions=config.embedding_dim,
        ),
        vector_store=PgVectorStore(config.database_url),
    )

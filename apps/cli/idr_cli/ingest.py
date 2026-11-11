"""ingest 命令：跑完入库门面、输出计数并写一份批次报告。"""

from __future__ import annotations

import sys
from pathlib import Path

from idr_core.config import Config
from idr_core.embedding import EMBEDDING_DIM
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.llm.embedding import OpenAICompatibleEmbedder
from idr_core.llm.openai_compat import OpenAICompatibleLLM
from idr_core.parsing.pdf import PdfTextParser
from idr_core.state.postgres import PostgresStateStore
from idr_core.vector_store.pgvector import PgVectorStore
from idr_rotating import (
    DocumentOutcome,
    IngestDeps,
    IngestError,
    OntologyError,
    PendingKind,
    ingest,
    write_report,
)

DEFAULT_REPORTS_DIR = Path("data/output/reports")


def run_ingest(
    source_dir: Path,
    *,
    deps: IngestDeps,
    limit: int | None = None,
    reports_dir: Path | None = None,
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
        outcome = document.status
        if outcome is DocumentOutcome.SKIPPED:
            print(f"{document.path.name}：已跳过（此前已入库）")
        elif outcome is DocumentOutcome.QUARANTINED and document.skipped:
            # 隔离状态缺原因只可能来自外部注入的状态存储，沿用旧文案。
            if document.quarantine_reason is not None:
                print(
                    f"{document.path.name}：已跳过"
                    f"（此前已隔离：{document.quarantine_reason}）"
                )
            else:
                print(f"{document.path.name}：已跳过（此前已入库）")
        elif outcome is DocumentOutcome.QUARANTINED:
            print(
                f"{document.path.name}：已隔离（{document.quarantine_reason}）"
            )
        elif outcome is DocumentOutcome.FAILED:
            print(f"{document.path.name}：失败（{document.error}）")
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
        for warning in document.warnings:
            print(f"  警告：{warning}")

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
        f"分类：成功 {report.succeeded}，失败 {report.failed}，"
        f"隔离 {report.quarantined}，跳过 {report.skipped}，"
        f"待审计 {report.pending_review_count}"
    )
    print(
        f"合计：证据段 {report.evidence_segments}，"
        f"实体 {report.entities}，"
        f"关系 {report.relations}"
    )

    report_path = write_report(
        report,
        output_dir=DEFAULT_REPORTS_DIR if reports_dir is None else reports_dir,
    )
    print(f"报告：{report_path}")
    return 0 if report.failed == 0 else 1


def build_deps(config: Config) -> IngestDeps:
    """真实依赖：模型客户端、pdfplumber 解析器、Neo4j、嵌入服务与 pgvector。

    嵌入维度是 ADR-0002 的共享常量，加载配置时已校验 ``EMBEDDING_DIM`` 必须与
    之相等；这里只传常量，代码里不再出现第二份 1024。
    """
    return IngestDeps(
        llm=OpenAICompatibleLLM(
            base_url=config.deepseek_base_url,
            api_key=config.deepseek_api_key,
            model=config.deepseek_model,
            extra_headers=config.llm_extra_headers,
        ),
        parser=PdfTextParser(),
        graph_store=Neo4jGraphStore(
            config.neo4j_uri, config.neo4j_user, config.neo4j_password
        ),
        embedder=OpenAICompatibleEmbedder(
            base_url=config.siliconflow_base_url,
            api_key=config.siliconflow_api_key,
            model=config.embedding_model,
            dimensions=EMBEDDING_DIM,
        ),
        vector_store=PgVectorStore(config.database_url),
        state_store=PostgresStateStore(config.database_url),
    )

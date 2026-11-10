"""ingest 命令：跑完入库门面并输出每篇文献与合计的计数。"""

from __future__ import annotations

import sys
from pathlib import Path

from idr_core.config import Config
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.llm.openai_compat import OpenAICompatibleLLM
from idr_core.parsing.pdf import PdfTextParser
from idr_rotating import IngestDeps, IngestError, ingest


def run_ingest(
    source_dir: Path, *, deps: IngestDeps, limit: int | None = None
) -> int:
    try:
        report = ingest(source_dir, deps=deps, limit=limit)
    except IngestError as error:
        print(f"入库失败：{error}", file=sys.stderr)
        return 1

    for document in report.documents:
        if document.error is not None:
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

    print(
        f"合计：证据段 {report.evidence_segments}，"
        f"实体 {report.entities}，"
        f"关系 {report.relations}"
    )
    return 0 if report.failed == 0 else 1


def build_deps(config: Config) -> IngestDeps:
    """真实依赖：DeepSeek 兼容客户端、pdfplumber 解析器、Neo4j 图存储。

    嵌入与向量存储留给 T6；本票真的不写向量，因此显式不注入。
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
    )

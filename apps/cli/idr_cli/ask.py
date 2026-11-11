"""ask 命令：问一个问题，打印带引用的结构化答案。

人类可读输出按「问题 → 检索路径 → 命中概念 → 结论 → 每条结论的出处」排列；
向量兜底的答案没有图上的结论，这一栏换成按重排相关度排序的文献证据。
``--json`` 输出同一份答案的机器形态，供评估与后续服务使用。

文献名不在图谱里，图谱只存 ``doc_id``。这里按语料目录逐篇重算内容哈希，
把标识对回文件名：与入库时的标识规则同源，因此不需要额外登记，也不会因为
报告文件被清理而丢失引用里的文献名。

真实依赖在 ``build_ask_deps`` 里一次接齐：Neo4j 图、pgvector 向量表、SiliconFlow
的嵌入与重排。三者都只在真正被调用时联网，``--json`` 与单元测试用假依赖。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from idr_core.config import Config
from idr_core.documents import document_id
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.llm.embedding import OpenAICompatibleEmbedder
from idr_core.llm.rerank import OpenAICompatibleReranker
from idr_core.vector_store.pgvector import PgVectorStore
from idr_rotating import (
    Answer,
    AnswerPath,
    AskDeps,
    ConceptMatch,
    LinkKind,
    OntologyError,
    RetrievalSettingsError,
    ask,
)
from idr_rotating.answer import answer_to_dict
from idr_rotating.ontology import Ontology, load_ontology

DEFAULT_SOURCE_DIR = Path("data/input")

_PATH_LABELS = {
    AnswerPath.GRAPH: "图命中",
    AnswerPath.VECTOR: "向量兜底",
    AnswerPath.REFUSAL: "拒答",
}


class CorpusDocumentNames:
    """按内容哈希把 ``doc_id`` 对回语料目录里的文件名；查不到返回 ``None``。"""

    def __init__(self, source_dir: Path) -> None:
        self._source_dir = source_dir
        self._names: dict[str, str] | None = None

    def name_for(self, doc_id: str) -> str | None:
        if self._names is None:
            self._names = _index_source_dir(self._source_dir)
        return self._names.get(doc_id)


def _index_source_dir(source_dir: Path) -> dict[str, str]:
    """逐个文件算标识；损坏或已删除的文件跳过，不影响其余引用。"""
    names: dict[str, str] = {}
    for path in sorted(source_dir.glob("*.pdf")):
        try:
            names[document_id(path.read_bytes())] = path.name
        except OSError:
            continue
    return names


def build_ask_deps(config: Config, source_dir: Path | None = None) -> AskDeps:
    """真实依赖：Neo4j 图、pgvector 向量表、嵌入与重排，加上文献名目录。"""
    return AskDeps(
        graph_store=Neo4jGraphStore(
            config.neo4j_uri, config.neo4j_user, config.neo4j_password
        ),
        document_names=CorpusDocumentNames(
            DEFAULT_SOURCE_DIR if source_dir is None else source_dir
        ),
        embedder=OpenAICompatibleEmbedder(
            base_url=config.siliconflow_base_url,
            api_key=config.siliconflow_api_key,
            model=config.embedding_model,
        ),
        vector_store=PgVectorStore(config.database_url),
        reranker=OpenAICompatibleReranker(
            base_url=config.siliconflow_base_url,
            api_key=config.siliconflow_api_key,
            model=config.rerank_model,
        ),
    )


def run_ask(
    question: str,
    *,
    deps: AskDeps,
    as_json: bool = False,
) -> int:
    try:
        answer = ask(question, deps=deps)
        output = (
            json.dumps(answer_to_dict(answer), ensure_ascii=False, indent=2)
            if as_json
            else render_answer(answer)
        )
    except OntologyError as error:
        print(f"本体错误：{error}", file=sys.stderr)
        return 1
    except RetrievalSettingsError as error:
        print(f"检索配置错误：{error}", file=sys.stderr)
        return 1

    print(output)
    # 拒答是正常结果而不是命令失败：空答案已经把「为什么没答」写清楚了。
    return 0


def render_answer(answer: Answer) -> str:
    """人类可读输出；关系与类型用本体里的中文标签，不暴露内部标识。"""
    ontology = load_ontology()
    lines = [f"问题：{answer.question}", f"检索路径：{_PATH_LABELS[answer.path]}"]

    others = [
        match
        for match in answer.matches
        if answer.focus is None or match.concept.id != answer.focus.concept.id
    ]
    if answer.focus is not None:
        lines.append(f"命中概念：{_match_text(answer.focus, ontology)}")
    elif others:
        lines.append(
            "命中概念：" + "；".join(_match_text(match, ontology) for match in others)
        )
    if answer.focus is not None and others:
        lines.append(
            "同时命中：" + "；".join(_match_text(match, ontology) for match in others)
        )

    if answer.path is AnswerPath.VECTOR and answer.citations:
        lines.extend(["", "文献证据（图路径未命中，按重排相关度排序）："])
        for index, citation in enumerate(answer.citations, start=1):
            document = (
                citation.document
                if citation.document is not None
                else f"文献 {citation.doc_id}"
            )
            score = (
                f"（相关度 {citation.score:.2f}）"
                if citation.score is not None
                else ""
            )
            lines.append(
                f"{index}. {document} 第 {citation.page} 页{score}："
                f"{_one_line(citation.text)}"
            )
        return "\n".join(lines)

    if not answer.facts:
        detail = f"：{answer.detail}" if answer.detail else "。"
        lines.extend(["", f"没有可回答的依据{detail}"])
        return "\n".join(lines)

    citations = {citation.evidence_id: citation for citation in answer.citations}
    lines.extend(["", "结论："])
    for index, fact in enumerate(answer.facts, start=1):
        relation = ontology.relation_types.get(fact.relation)
        relation_label = relation.label if relation is not None else fact.relation
        lines.append(
            f"{index}. {fact.head.name} —{relation_label}→ {fact.tail.name}"
            f"（置信度 {fact.confidence:.2f}）"
        )
        for evidence_id in fact.evidence_ids:
            citation = citations[evidence_id]
            document = (
                citation.document
                if citation.document is not None
                else f"文献 {citation.doc_id}"
            )
            lines.append(
                f"   依据：{document} 第 {citation.page} 页："
                f"{_one_line(citation.text)}"
            )
    return "\n".join(lines)


def _match_text(match: ConceptMatch, ontology: Ontology) -> str:
    type_definition = ontology.entity_types.get(match.concept.type)
    type_label = (
        type_definition.label
        if type_definition is not None
        else match.concept.type
    )
    kind_label = "整句命中" if match.kind is LinkKind.EXACT else "子串命中"
    return (
        f"{match.concept.name}（{type_label}，"
        f"别名「{match.alias}」{kind_label}）"
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())

"""文档级判别：论文还是专利，以及是否属于没有文字层的扫描件。

这两件事都必须自动完成——语料目录里论文与专利混放，调用方不会预先标注；
扫描件也不能靠人工挑出来，否则空证据段会污染图谱。
"""

from __future__ import annotations

import re
from enum import StrEnum

from idr_core.parsing import ParsedDocument

# 平均每页有效字符低于此值时判定为扫描件；真实文字层远超此值，
# 纯扫描件为 0，只有页码或水印的扫描件也很低。
MIN_EFFECTIVE_CHARS_PER_PAGE = 10
# 单页文档的兜底：整篇有效字符过少同样视为没有文字层。
MIN_EFFECTIVE_CHARS = 20

_PATENT_PARAGRAPH = re.compile(r"\[\d{4,5}\]")

# 专利文献特有的标记。刻意不放「摘要」「实施例」等两类文献都可能出现的词，
# 避免把论文误判成专利；判型只求在混放语料里可靠，不求覆盖每一种版式。
_PATENT_LABELS = (
    "(19)",
    "(12)",
    "(10)",
    "(21)",
    "(22)",
    "(54)",
    "(57)",
    "(71)",
    "(72)",
    "中华人民共和国国家知识产权局",
    "发明专利申请",
    "实用新型专利",
    "专利申请公布",
    "权利要求书",
    "说明书",
    "技术领域",
    "背景技术",
    "发明内容",
    "具体实施方式",
    "附图说明",
    "united states patent",
    "patent application publication",
    "what is claimed",
    "we claim",
    "date of patent",
    "int. cl.",
    "u.s. cl.",
    "field of search",
    "patent no.",
)

# 论文特有的标记：中文期刊的固定体例、英文论文的常用章节与参考文献格式。
_PAPER_LABELS = (
    "摘要",
    "关键词",
    "中图分类号",
    "文献标识码",
    "文章编号",
    "收稿日期",
    "基金项目",
    "作者简介",
    "参考文献",
    "引言",
    "前言",
    "绪论",
    "abstract",
    "keywords",
    "introduction",
    "references",
    "doi",
    "issn",
    "[j]",
    "[m]",
    "[d]",
    "et al.",
)


class DocumentType(StrEnum):
    """切分规则所需的文档类型。"""

    PAPER = "paper"
    PATENT = "patent"


def detect_document_type(parsed: ParsedDocument) -> DocumentType:
    """按结构标记自动判别论文与专利，歧义时按论文处理。"""
    text = "\n".join(page.text for page in parsed.pages).lower()

    paragraph_marks = len(_PATENT_PARAGRAPH.findall(text))
    patent_labels = _count_labels(text, _PATENT_LABELS)
    paper_labels = _count_labels(text, _PAPER_LABELS)

    if paragraph_marks >= 2:
        return DocumentType.PATENT
    if paragraph_marks >= 1 and patent_labels >= 1:
        return DocumentType.PATENT
    if patent_labels >= 2 and patent_labels > paper_labels:
        return DocumentType.PATENT
    return DocumentType.PAPER


def scanned_document_reason(parsed: ParsedDocument) -> str | None:
    """没有文字层时返回可写入报告的原因，否则返回 None。"""
    if not parsed.pages:
        return "扫描件：解析结果没有任何页面"

    effective_chars = sum(
        sum(1 for character in page.text if not character.isspace())
        for page in parsed.pages
    )
    pages = len(parsed.pages)
    average = effective_chars / pages
    if effective_chars >= MIN_EFFECTIVE_CHARS and average >= MIN_EFFECTIVE_CHARS_PER_PAGE:
        return None
    return (
        "扫描件：未检测到可用的文字层"
        f"（有效字符 {effective_chars}，页数 {pages}，平均每页 {average:.1f}；"
        f"阈值：总字符 ≥ {MIN_EFFECTIVE_CHARS} 且平均每页 ≥ "
        f"{MIN_EFFECTIVE_CHARS_PER_PAGE}）"
    )


def _count_labels(text: str, labels: tuple[str, ...]) -> int:
    return sum(1 for label in labels if label in text)

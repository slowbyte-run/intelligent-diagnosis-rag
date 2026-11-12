"""短锚点上下文合并的集成契约：合并后的引文必须能通过回源逐字校验。

单测（``test_ask_anchor_context.py``）用真实切分器证明了「合并文本是页面文字的
连续切片」；这里再走最终口径——拿 ``docs/m2-baseline.md`` 记录的那张公式页
（外圈/内圈故障频率被切成 9–52 字的碎片），把最短的碎片当成锚点，合并同页相邻
证据段，然后交给评估用的 ``SourcePdfVerifier`` 按源 PDF 逐字核对。只要合并在
段间塞了任何非原文的字符，这条测试就会失败。

需要 Neo4j 已入库语料与本地 ``data/input``；两者缺一都跳过。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from idr_core.anchors import EvidenceAnchor
from idr_core.documents import document_id
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.parsing.pdf import page_text
from idr_rotating.citation_check import SourcePdfVerifier
from idr_rotating.citation_context import SHORT_ANCHOR_CHARS, display_text

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

REPO_ROOT = Path(__file__).resolve().parents[3]

# docs/m2-baseline.md 里 P-01 的出处：同一页上外圈/内圈故障频率的公式碎片。
FORMULA_DOC = "4a8dc5af5de2cfe4"
FORMULA_PAGE = 20

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not NEO4J_PASSWORD,
        reason="需要 NEO4J_PASSWORD（先 source .env）",
    ),
]


def test_merged_context_still_verifies_against_the_source_pdf() -> None:
    corpus_dir = REPO_ROOT / "data" / "input"
    if not corpus_dir.is_dir():
        pytest.skip("data/input 不入库；本机有语料时才校验合并文本的回源逐字一致")

    documents = {
        document_id(path.read_bytes()): path
        for path in sorted(corpus_dir.glob("*.pdf"))
    }
    source = documents.get(FORMULA_DOC)
    assert source is not None, "语料快照里找不到 P-01 公式页所属的文献"

    graph_store = Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        records = graph_store.evidence_by_page(FORMULA_DOC, FORMULA_PAGE)
    finally:
        graph_store.close()

    assert records, "线上图谱里没有 P-01 公式页的证据段"
    # P-01 的处境：外圈/内圈两条公式各是一段，中间夹着一小块被切断的表达式。
    outer_formula = next(
        (record for record in records if "Outer race defect" in record.text), None
    )
    inner_formula = next(
        (record for record in records if "Inner race defect" in record.text), None
    )
    assert outer_formula is not None and inner_formula is not None
    fragment = next(
        record
        for record in records
        if record.charspan.start == outer_formula.charspan.end + 1
        and record.charspan.end == inner_formula.charspan.start - 1
    )
    assert len(fragment.text) < SHORT_ANCHOR_CHARS
    anchor = EvidenceAnchor(
        evidence_id=fragment.id, start=0, end=len(fragment.text)
    )
    merged = display_text(fragment, anchor, lambda: records)

    # 锚点仍是原来的碎片，合并只扩大展示文本；上限内把整块公式都带了进来，
    # 这正是「公式碎片可读」要的效果。
    assert fragment.text in merged
    assert len(merged) > len(fragment.text)
    assert len(merged) <= 240
    for label in (
        "Outer race defect",
        "Inner race defect",
        "Rolling Element defect",
        "Cage Defect Frequency",
    ):
        assert label in merged

    # 图谱里的字符区间对得上源 PDF：碎片在它所标的页、所标的偏移上逐字一致。
    source_page = page_text(source, FORMULA_PAGE)
    assert (
        source_page[
            fragment.charspan.start : fragment.charspan.start + len(fragment.text)
        ]
        == fragment.text
    )
    # 最终口径：碎片与合并后的展示文本都能在源 PDF 该页逐字找到。
    verifier = SourcePdfVerifier(corpus_dir)
    curated = verifier.verify(
        {"doc_id": FORMULA_DOC, "page": FORMULA_PAGE, "text": fragment.text}
    )
    assert curated.ok, curated.reason
    checked = verifier.verify(
        {"doc_id": FORMULA_DOC, "page": FORMULA_PAGE, "text": merged}
    )
    assert checked.ok, checked.reason

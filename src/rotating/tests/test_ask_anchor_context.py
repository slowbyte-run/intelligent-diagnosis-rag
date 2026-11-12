"""ask 门面：短锚点的展示上下文（票据 #26）。

公式区被版面切分成 9–52 字的碎片，锚点落在碎片上时引用只有半句话。展示时
并入同页、相邻、连续的证据段作为上下文，锚点区间本身不变；合并有上限，不跨页，
也不跳过不连续的段——展示文本必须仍是页面文字的连续切片，否则回源校验
（``citation_check.SourcePdfVerifier``）会在源 PDF 上找不到它。

测试只走 ``ask`` 门面与内存假实现，不连 Neo4j 或 Postgres。
"""

from __future__ import annotations

from collections.abc import Sequence

from fakes import FakeGraphStore, parsed_page
from idr_core.graph import (
    ConceptNode,
    ConceptRelation,
    EvidenceAnchor,
    EvidenceRecord,
    GraphBatch,
    Provenance,
)
from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_core.parsing import ParsedDocument
from idr_rotating import AnswerPath, AskDeps, DocumentType, ask
from idr_rotating.segments import build_evidence_segments

OUTER_RACE = "concept:bearing_outer_race_fatigue_spalling"
BPFO = "concept:bpfo_harmonics"
DOC = "doc-a"
QUESTION = "轴承外圈疲劳剥落有什么征兆"

# 一页公式区被切成三段：前后是说明，中间是短碎片。段间恰好一个换行，
# 与 PDF 文字层把行连接起来的方式一致（密集证据段实测 6497 对全是 1 个字符）。
FORMULA_TEXTS = (
    "公式给出外圈与内圈故障频率。",
    "外圈剥落 = fr{1− cos α}",
    "内圈剥落 = fr{1+ cos α}",
)


class RecordingGraphStore(FakeGraphStore):
    """记录引用层按页取过哪些证据段；用来确认合并只读当前页。"""

    def __init__(self) -> None:
        super().__init__()
        self.pages_read: list[tuple[str, int]] = []

    def evidence_by_page(self, doc_id: str, page: int) -> tuple[EvidenceRecord, ...]:
        self.pages_read.append((doc_id, page))
        return super().evidence_by_page(doc_id, page)


def concept(instance_id: str, name: str, type_: str) -> ConceptNode:
    """本体规范实例对应的共享概念节点。"""
    return ConceptNode(
        id=f"concept:{instance_id}", name=name, type=type_, doc_id=None
    )


def page_segment(doc_id: str, page: int, start: int, text: str) -> EvidenceSegment:
    """页面文字 [start, start + len(text)) 上的一段证据。"""
    return EvidenceSegment(
        id=f"ev:{doc_id}:{page}:{start}-{start + len(text)}",
        doc_id=doc_id,
        page=page,
        bbox=BBox(0.0, 0.0, 595.0, 842.0),
        charspan=CharSpan(start, start + len(text)),
        text=text,
    )


def layered_page(
    texts: Sequence[str], *, page: int = 4
) -> tuple[str, tuple[EvidenceSegment, ...]]:
    """按阅读顺序把若干段文字排成一页，返回页面文字与证据段。

    段与段之间恰好一个换行：这正是切分器 ``_tighten`` 之后的真实形态，也是
    合并展示时唯一允许出现的分隔符。
    """
    segments: list[EvidenceSegment] = []
    cursor = 0
    for text in texts:
        segments.append(page_segment(DOC, page, cursor, text))
        cursor += len(text) + 1
    return "\n".join(texts), tuple(segments)


def deps_with_anchor(
    segments: Sequence[EvidenceSegment],
    *,
    anchor_index: int,
    anchor_span: tuple[int, int] | None = None,
    graph: FakeGraphStore | None = None,
) -> AskDeps:
    """一条 ``indicated_by`` 边，锚点落在第 ``anchor_index`` 段上。"""
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    segment = segments[anchor_index]
    start, end = (
        (0, len(segment.text)) if anchor_span is None else anchor_span
    )
    store = FakeGraphStore() if graph is None else graph
    store.write_document(
        GraphBatch(
            doc_id=DOC,
            concepts=(outer, bpfo),
            evidence=tuple(segments),
            relations=(
                ConceptRelation(
                    head_id=outer.id,
                    type="indicated_by",
                    tail_id=bpfo.id,
                    provenance=Provenance(
                        doc_id=DOC,
                        evidence_ids=(segment.id,),
                        confidence=0.8,
                        source="extracted",
                        anchors=(
                            EvidenceAnchor(
                                evidence_id=segment.id, start=start, end=end
                            ),
                        ),
                    ),
                ),
            ),
            evidence_links=(),
        )
    )
    return AskDeps(graph_store=store)


def test_short_anchor_shows_adjacent_segments_as_context() -> None:
    """短于阈值的锚点：展示并入同页前后相邻的证据段，锚点区间本身不变。"""
    page_text, segments = layered_page(FORMULA_TEXTS)

    answer = ask(QUESTION, deps=deps_with_anchor(segments, anchor_index=1))

    assert answer.path is AnswerPath.GRAPH
    (citation,) = answer.citations
    # 展示文本是整页文字的一段连续切片，三段都并入（远未触到上限）。
    assert citation.text == page_text
    assert citation.text == page_text[
        segments[0].charspan.start : segments[-1].charspan.end
    ]
    # 锚点仍是原来的碎片区间（相对所属证据段），不随展示文本扩大。
    assert citation.anchor is not None
    assert citation.anchor == EvidenceAnchor(
        evidence_id=segments[1].id,
        start=0,
        end=len(FORMULA_TEXTS[1]),
    )
    assert (
        segments[1].text[citation.anchor.start : citation.anchor.end]
        == FORMULA_TEXTS[1]
    )


# 阈值边界：低于 SHORT_ANCHOR_CHARS 才并入上下文，恰好等于阈值不再扩。
_THRESHOLD_BASE = "外圈剥落的特征频率出现在频谱上，能量集中在高频段，"
LONG_QUOTE = (_THRESHOLD_BASE * 4)[:60]
LONG_TEXTS = ("上一段说明公式的适用条件。", LONG_QUOTE, "下一段给出符号含义。")


def test_anchor_at_the_threshold_is_shown_as_is() -> None:
    """锚点长度到阈值就不再并入上下文：它已经是可以独立阅读的一句。"""
    assert len(LONG_QUOTE) == 60
    page_text, segments = layered_page(LONG_TEXTS)

    answer = ask(QUESTION, deps=deps_with_anchor(segments, anchor_index=1))

    (citation,) = answer.citations
    assert citation.text == LONG_QUOTE
    assert citation.text != page_text
    assert citation.anchor is not None
    assert (
        segments[1].text[citation.anchor.start : citation.anchor.end] == LONG_QUOTE
    )


def test_anchor_below_the_threshold_is_widened() -> None:
    """恰好比阈值短一个字仍算短锚点：边界两侧的行为必须一致可预期。"""
    short_quote = (_THRESHOLD_BASE * 4)[:59]
    assert len(short_quote) == 59
    page_text, segments = layered_page(
        ("上一段说明公式的适用条件。", short_quote, "下一段给出符号含义。")
    )

    answer = ask(QUESTION, deps=deps_with_anchor(segments, anchor_index=1))

    (citation,) = answer.citations
    assert citation.text == page_text
    assert len(citation.text) > len(short_quote)


def repeated(base: str, length: int) -> str:
    """重复一句中文并截到指定长度，用来构造刚好触到展示上限的段落。"""
    return (base * (length // len(base) + 1))[:length]


def test_merge_stops_before_exceeding_the_display_limit() -> None:
    """合并触到上限就停：不把最后一段切一半塞进引文，也不继续往下扩。"""
    base = repeated("外圈剥落的特征频率由几何参数决定，", 40)
    first_neighbour = repeated("内圈剥落的特征频率同样由几何参数决定，", 100)
    second_neighbour = repeated("保持架频率的表达式与之不同，", 100)
    page_text, segments = layered_page((base, first_neighbour, second_neighbour))

    answer = ask(QUESTION, deps=deps_with_anchor(segments, anchor_index=0))

    (citation,) = answer.citations
    expected = base + "\n" + first_neighbour
    # 再并下一段会到 242 字，超过 240 的展示上限，所以停在上一段。
    assert len(expected) == 141
    assert len(expected + "\n" + second_neighbour) == 242
    assert citation.text == expected
    assert len(citation.text) <= 240
    assert second_neighbour not in citation.text
    assert citation.text in page_text


def test_merge_does_not_cross_pages() -> None:
    """上下文只在同一页内找：另一页上紧挨着的证据段不并入，页码不变。"""
    anchor_text = "外圈剥落 = fr{1− cos α}"
    anchor_segment = page_segment(DOC, 4, 0, anchor_text)
    next_page = page_segment(DOC, 5, len(anchor_text) + 1, "内圈剥落 = fr{1+ cos α}")

    graph = RecordingGraphStore()
    answer = ask(
        QUESTION,
        deps=deps_with_anchor(
            (anchor_segment, next_page), anchor_index=0, graph=graph
        ),
    )

    (citation,) = answer.citations
    assert citation.page == 4
    assert citation.text == anchor_text
    assert next_page.text not in citation.text
    # 引用层只按当前页取邻居：另一页的证据段根本没进合并逻辑。
    assert graph.pages_read == [(DOC, 4)]


def test_merge_stops_at_a_non_contiguous_gap() -> None:
    """与下一段之间隔着未知长度的文字时不合并：跳过它去拼更远的段会切断页面原文。"""
    first = page_segment(DOC, 4, 0, "外圈剥落 = fr{1− cos α}")
    # 间隔 5 个字符：不是行连接符，无法在不读原文的情况下重建。
    second = page_segment(DOC, 4, len(first.text) + 5, "内圈剥落 = fr{1+ cos α}")

    answer = ask(
        QUESTION, deps=deps_with_anchor((first, second), anchor_index=0)
    )

    (citation,) = answer.citations
    assert citation.text == first.text
    assert second.text not in citation.text


def test_short_anchor_in_a_segment_over_the_limit_shows_only_the_quote() -> None:
    """锚点所在证据段本身超过上限时只展示片段：截断整段会在句子中间下刀。"""
    huge = repeated("外圈剥落的特征频率由几何参数决定，", 300)
    neighbour = page_segment(DOC, 4, len(huge) + 1, "内圈剥落 = fr{1+ cos α}")
    anchor_start, anchor_end = 20, 40

    answer = ask(
        QUESTION,
        deps=deps_with_anchor(
            (page_segment(DOC, 4, 0, huge), neighbour),
            anchor_index=0,
            anchor_span=(anchor_start, anchor_end),
        ),
    )

    (citation,) = answer.citations
    assert citation.text == huge[anchor_start:anchor_end]
    assert neighbour.text not in citation.text


def test_merged_text_is_a_contiguous_slice_of_the_page_text() -> None:
    """用真实切分器切出的碎片验证：合并后的展示文本仍是页面文字的连续切片。

    证据段由 ``build_evidence_segments`` 从一页文字切出来，段与段之间只隔一个
    换行；合并后的字符串必须能在这页文字里原样找到——否则回源 PDF 的逐字
    校验会失败，引用准确率会掉。
    """
    lines = (
        repeated("外圈剥落的特征频率由几何参数决定，", 200),
        "外圈剥落 = fr{1− cos α}",
        "内圈剥落 = fr{1+ cos α}",
        repeated("式中各符号的含义见前文。", 100),
    )
    page_text = "\n".join(lines)
    page = parsed_page(
        4,
        page_text,
        line_boxes=[
            BBox(72.0, 100.0 + 60 * index, 500.0, 112.0 + 60 * index)
            for index in range(len(lines))
        ],
    )
    segments = build_evidence_segments(
        DOC, ParsedDocument(pages=(page,)), DocumentType.PAPER
    )
    # 每行一段：行距远大于行高，切分器把它们判成四个段落。
    assert [segment.text for segment in segments] == list(lines)
    fragment = segments[1]

    answer = ask(QUESTION, deps=deps_with_anchor(segments, anchor_index=1))

    (citation,) = answer.citations
    assert citation.text in page_text
    assert citation.text != page_text
    assert len(citation.text) <= 240
    assert citation.anchor is not None
    assert citation.anchor.evidence_id == fragment.id
    quote = fragment.text[citation.anchor.start : citation.anchor.end]
    assert quote == fragment.text
    assert quote in citation.text


# 与 #25（无锚点按断言选句）的组合边界：选句窗口已经落在句子边界上，
# 引用层不再拿它当短锚点去并入上下文，否则刚收窄的引用又会摊回整段。
def test_selected_window_for_an_anchorless_assertion_is_not_rewidened() -> None:
    """无锚点断言走选句兜底时，展示文本就是选出的句子，不并入相邻证据段。"""
    paragraph = (
        "外圈剥落会在频谱上出现 BPFO 及其倍频。"
        "内圈剥落会产生 BPFI 特征频率，与转速同步。"
    )
    page_text, segments = layered_page(
        ("上一段说明测点布置。", paragraph, "下一段讨论包络解调。")
    )
    outer = concept(OUTER_RACE.removeprefix("concept:"), "轴承外圈疲劳剥落", "fault_mode")
    bpfo = concept(BPFO.removeprefix("concept:"), "外圈故障频率及其倍频", "symptom")
    graph = FakeGraphStore()
    graph.write_document(
        GraphBatch(
            doc_id=DOC,
            concepts=(outer, bpfo),
            evidence=segments,
            relations=(
                ConceptRelation(
                    head_id=outer.id,
                    type="indicated_by",
                    tail_id=bpfo.id,
                    provenance=Provenance(
                        doc_id=DOC,
                        evidence_ids=(segments[1].id,),
                        confidence=0.8,
                        source="extracted",
                    ),
                ),
            ),
            evidence_links=(),
        )
    )

    answer = ask(QUESTION, deps=AskDeps(graph_store=graph))

    (citation,) = answer.citations
    assert citation.text == "外圈剥落会在频谱上出现 BPFO 及其倍频。"
    assert citation.text in page_text
    assert citation.anchor is not None
    assert "上一段说明测点布置。" not in citation.text
    assert "下一段讨论包络解调。" not in citation.text

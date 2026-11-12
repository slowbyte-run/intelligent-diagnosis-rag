"""短锚点的展示上下文：把同页相邻证据段并入引文，锚点区间本身不变。

公式区被版面切分器切成 9–52 字的碎片，锚点落在碎片上时，引用只有半句话。
展示时从锚点所在的证据段向两侧扩展：只并入同一页、在页面文字上连续、且并入后
不超过展示上限的证据段。``Citation.anchor`` 始终是原来的碎片区间——扩大的只是
展示文本，回源校验（``citation_check``）仍按原区间逐字核对。

**不猜分隔符。** 页面文字里两段证据之间的间隔只允许是 0 个字符或 1 个换行：
后者是文字层抽取（``pdfplumber.extract_text``）连接行的方式，证据段边界都在
行首，切分时的 ``_tighten`` 只把两侧空白裁掉。间隔更长或为负（越界/重叠）时
停止扩展——跳过一段未知文字去拼下一段，拼出来的就不再是页面文字的连续切片，
回源校验会失败，引用准确率会掉。这条规则在当前语料上逐页核对过：309 页的
证据段按「单换行连接」重建，与源 PDF 的页面文字逐字一致。

上限是**展示文本长度**而不是并入段数：一段就写了半页的证据不该因为「只多并了
一段」而出现在引用里。超过上限即停止，宁可少给上下文，也不让一条引用退化成
整页。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from idr_core.anchors import EvidenceAnchor
from idr_core.graph import EvidenceRecord

# 短于这个长度的锚点当成碎片，展示时并入上下文。当前语料里公式区的碎片是
# 9–52 字；60 字以上通常已是完整的一句（中文）或一个从句（英文），不需要再补。
SHORT_ANCHOR_CHARS = 60

# 展示文本的上限：约 2–3 句，足以还原一个公式块；语料里页面普遍上千字，
# 因此引用不会退化成整页。上限按字符数而不是证据段数，理由见模块说明。
MAX_CONTEXT_CHARS = 240


def _needs_context(anchor: EvidenceAnchor) -> bool:
    """锚点是否短到需要并入上下文。"""
    return anchor.end - anchor.start < SHORT_ANCHOR_CHARS


def display_text(
    segment: EvidenceRecord,
    anchor: EvidenceAnchor,
    page_segments: Callable[[], Sequence[EvidenceRecord]],
) -> str:
    """引用的展示文本：短锚点并入同页相邻证据段，长锚点原样返回。

    ``page_segments`` 是取同页证据段的惰性回调：长锚点不需要上下文，调用方
    因此不必为每条引用多查一次图。锚点区间相对 ``segment.text``，合并只在
    页面文字坐标系里进行，返回的字符串始终是这一页文字的连续切片。

    扩展顺序由近及远、左右交替（同距离先左后右）：结果与字典序、适配器返回
    顺序无关，同一个锚点每次得到同样的上下文。
    """
    quote = segment.text[anchor.start : anchor.end]
    if not _needs_context(anchor):
        return quote
    if len(segment.text) > MAX_CONTEXT_CHARS:
        # 锚点所在证据段自己就超过展示上限：并任何邻居都必然超限。截断整段
        # 会在句子中间下刀，不如只给锚点片段。
        return quote

    ordered = _ordered_page(segment, page_segments())
    index = next(
        (position for position, item in enumerate(ordered) if item.id == segment.id),
        None,
    )
    if index is None:
        # 读模型里找不到锚点所属证据段（数据不一致）：退回片段，不猜邻居。
        return quote

    text = segment.text
    start = segment.charspan.start
    end = segment.charspan.end
    left = index - 1
    right = index + 1
    while left >= 0 or right < len(ordered):
        progressed = False
        if left >= 0:
            extended = _extend_left(text, start, ordered[left])
            if extended is None:
                # 不连续或超出上限：更远的段只会更远，这一侧到此为止。
                left = -1
            else:
                text, start = extended
                left -= 1
                progressed = True
        if right < len(ordered):
            extended = _extend_right(text, end, ordered[right])
            if extended is None:
                right = len(ordered)
            else:
                text, end = extended
                right += 1
                progressed = True
        if not progressed:
            break
    return text


def _ordered_page(
    segment: EvidenceRecord, page_segments: Sequence[EvidenceRecord]
) -> tuple[EvidenceRecord, ...]:
    """同一篇文献同一页的证据段，按页面字符区间升序、按标识去重。

    端口已按区间排序，这里仍自行排序与去重：合并的正确性依赖顺序，不能把
    它寄托在适配器的实现细节上。
    """
    unique = {
        item.id: item
        for item in page_segments
        if item.doc_id == segment.doc_id and item.page == segment.page
    }
    return tuple(sorted(unique.values(), key=lambda item: item.charspan.start))


def _extend_left(
    text: str, start: int, neighbour: EvidenceRecord
) -> tuple[str, int] | None:
    """把左侧相邻证据段并入；不连续或超过上限时返回 ``None``。"""
    separator = _separator(start - neighbour.charspan.end)
    if separator is None:
        return None
    candidate = neighbour.text + separator + text
    if len(candidate) > MAX_CONTEXT_CHARS:
        return None
    return candidate, neighbour.charspan.start


def _extend_right(
    text: str, end: int, neighbour: EvidenceRecord
) -> tuple[str, int] | None:
    """把右侧相邻证据段并入；不连续或超过上限时返回 ``None``。"""
    separator = _separator(neighbour.charspan.start - end)
    if separator is None:
        return None
    candidate = text + separator + neighbour.text
    if len(candidate) > MAX_CONTEXT_CHARS:
        return None
    return candidate, neighbour.charspan.end


def _separator(gap: int) -> str | None:
    """两段之间的页面文字间隔。

    0 个字符表示两段直接相接，拼接时什么都不加；1 个字符是行连接用的换行；
    其余长度无法在不看原文的情况下重建，一律视为不连续。
    """
    if gap == 0:
        return ""
    if gap == 1:
        return "\n"
    return None

"""跨领域共享的值对象。

这里只允许出现与具体领域无关的流水线概念（见 AGENTS.md 第 2 条）。
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """版面坐标，单位为 PDF 点，原点在页面左上角。"""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class CharSpan:
    """证据段在所属页面文字中的字符区间，左闭右开。"""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class EvidenceSegment:
    """一条保留原文位置的连续文字，是引用回溯的最小单位。"""

    id: str
    doc_id: str
    page: int
    bbox: BBox
    charspan: CharSpan
    text: str
    # 定位降级的事实（例如 bbox 退回整页），随证据段一起留存。
    warnings: tuple[str, ...] = ()

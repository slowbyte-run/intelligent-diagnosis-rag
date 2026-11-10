"""PDF 解析端口与解析结果模型。

解析只负责把文件变成带页码与版面坐标的纯文字；如何切成证据段由领域包决定。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from idr_core.models import BBox, CharSpan


@dataclass(frozen=True, slots=True)
class ParsedLine:
    """一行文字及其在页面文字中的字符区间与版面坐标。

    证据段由若干行组成，段的版面坐标取这些行的并集，才能定位到原文
    而不是整页。行是解析器能稳定给出的最小版面单位。
    """

    text: str
    bbox: BBox
    charspan: CharSpan


@dataclass(frozen=True, slots=True)
class ParsedPage:
    number: int
    text: str
    bbox: BBox
    lines: tuple[ParsedLine, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    pages: tuple[ParsedPage, ...]


class Parser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...

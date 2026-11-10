"""PDF 解析端口与解析结果模型。

解析只负责把文件变成带页码与版面坐标的纯文字；如何切成证据段由领域包决定。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from idr_core.models import BBox


@dataclass(frozen=True, slots=True)
class ParsedPage:
    number: int
    text: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    pages: tuple[ParsedPage, ...]


class Parser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...

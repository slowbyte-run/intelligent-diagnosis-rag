"""pdfplumber 适配器：只取文字层，不做 OCR。"""

from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedDocument, ParsedLine, ParsedPage


def page_text(path: Path, number: int) -> str:
    """读取源 PDF 第 ``number`` 页的文字层，页码从 1 开始。

    回源校验要的是「文献现在这一页写着什么」，不是解析时留下的副本：调用方
    拿它逐字核对引文，因此这里不读缓存、不读图谱。页码越界抛 ``IndexError``；
    扫描件没有文字层时返回空字符串，是否算失败由调用方判定。
    """
    with pdfplumber.open(path) as pdf:
        if number < 1 or number > len(pdf.pages):
            raise IndexError(
                f"{path.name} 只有 {len(pdf.pages)} 页，调用方声称第 {number} 页"
            )
        return pdf.pages[number - 1].extract_text() or ""


class PdfTextParser:
    """把 PDF 解析成按页的文字、行级版面坐标与字符区间。

    扫描件没有文字层，解析结果会是空字符串；是否隔离由领域包决定。
    """

    def parse(self, path: Path) -> ParsedDocument:
        with pdfplumber.open(path) as pdf:
            pages = tuple(
                self._parse_page(number, page)
                for number, page in enumerate(pdf.pages, start=1)
            )
        return ParsedDocument(pages=pages)

    @staticmethod
    def _parse_page(number: int, page: Page) -> ParsedPage:
        text = page.extract_text() or ""
        lines: list[ParsedLine] = []
        dropped_lines = 0
        cursor = 0
        for raw in page.extract_text_lines(strip=True, return_chars=False):
            line_text = raw["text"]
            start = text.find(line_text, cursor)
            if start < 0:  # 行文字与整页文字对不上时宁可不给坐标，也不编造
                dropped_lines += 1
                continue
            end = start + len(line_text)
            lines.append(
                ParsedLine(
                    text=line_text,
                    bbox=BBox(
                        x0=float(raw["x0"]),
                        y0=float(raw["top"]),
                        x1=float(raw["x1"]),
                        y1=float(raw["bottom"]),
                    ),
                    charspan=CharSpan(start=start, end=end),
                )
            )
            cursor = end
        warnings: tuple[str, ...] = ()
        if dropped_lines:
            warnings = (
                f"第 {number} 页有 {dropped_lines} 行文字无法在整页文字中定位，"
                "已丢弃这些行的坐标",
            )
        return ParsedPage(
            number=number,
            text=text,
            bbox=BBox(*page.bbox),
            lines=tuple(lines),
            warnings=warnings,
        )

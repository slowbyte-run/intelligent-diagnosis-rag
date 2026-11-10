"""pdfplumber 适配器：只取文字层，不做 OCR。"""

from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedDocument, ParsedLine, ParsedPage


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
        cursor = 0
        for raw in page.extract_text_lines(strip=True, return_chars=False):
            line_text = raw["text"]
            start = text.find(line_text, cursor)
            if start < 0:  # 行文字与整页文字对不上时宁可不给坐标，也不编造
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
        return ParsedPage(
            number=number,
            text=text,
            bbox=BBox(*page.bbox),
            lines=tuple(lines),
        )

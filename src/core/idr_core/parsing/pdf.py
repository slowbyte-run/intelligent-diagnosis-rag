"""pdfplumber 适配器：只取文字层，不做 OCR。"""

from pathlib import Path

import pdfplumber

from idr_core.models import BBox
from idr_core.parsing import ParsedDocument, ParsedPage


class PdfTextParser:
    """把 PDF 解析成按页的文字与版面坐标。

    扫描件没有文字层，解析结果会是空字符串；是否隔离由领域包决定。
    """

    def parse(self, path: Path) -> ParsedDocument:
        with pdfplumber.open(path) as pdf:
            pages = tuple(
                ParsedPage(
                    number=number,
                    text=page.extract_text() or "",
                    bbox=BBox(*page.bbox),
                )
                for number, page in enumerate(pdf.pages, start=1)
            )
        return ParsedDocument(pages=pages)

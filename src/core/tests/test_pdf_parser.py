"""真实 PDF 文字层解析器的契约测试（不联网、不连数据库）。"""

from __future__ import annotations

from pathlib import Path

from idr_core.parsing.pdf import PdfTextParser


def one_page_pdf(text: str) -> bytes:
    """生成一页、一段文字的合法最小 PDF，避免为测试引入写 PDF 的依赖。"""

    return one_page_pdf_with_lines([text])


def one_page_pdf_with_lines(lines: list[str]) -> bytes:
    """生成一页多行文字的合法最小 PDF，行距固定 14 点。"""

    parts = ["BT /F1 12 Tf 72 720 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            parts.append("0 -14 Td")
        parts.append(f"({escaped}) Tj")
    parts.append("ET")
    stream = " ".join(parts).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    document = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_offset = len(document)
    document += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    document += b"0000000000 65535 f \n"
    for offset in offsets:
        document += f"{offset:010d} 00000 n \n".encode("ascii")
    document += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(document)


def test_parser_returns_text_page_with_bbox(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pdf"
    path.write_bytes(one_page_pdf("Sample line of text"))

    parsed = PdfTextParser().parse(path)

    assert len(parsed.pages) == 1
    page = parsed.pages[0]
    assert page.number == 1
    assert "Sample line of text" in page.text
    assert (page.bbox.x0, page.bbox.y0, page.bbox.x1, page.bbox.y1) == (
        0.0,
        0.0,
        612.0,
        792.0,
    )


def test_parser_exposes_line_geometry_for_segment_localization(
    tmp_path: Path,
) -> None:
    """证据段的版面坐标要靠行级坐标算出，不能退化成整页坐标。"""

    path = tmp_path / "lines.pdf"
    path.write_bytes(
        one_page_pdf_with_lines(
            ["First line of text occurs", "at the measured frequency."]
        )
    )

    parsed = PdfTextParser().parse(path)

    page = parsed.pages[0]
    assert page.text == "First line of text occurs\nat the measured frequency."
    assert [line.text for line in page.lines] == [
        "First line of text occurs",
        "at the measured frequency.",
    ]

    first, second = page.lines
    assert page.text[first.charspan.start : first.charspan.end] == first.text
    assert page.text[second.charspan.start : second.charspan.end] == second.text
    assert first.charspan.end < second.charspan.start
    assert first.bbox.y1 < second.bbox.y0  # 按阅读顺序自上而下
    assert page.bbox.x0 <= first.bbox.x0 < first.bbox.x1 <= page.bbox.x1
    assert page.bbox.y0 <= first.bbox.y0 < first.bbox.y1 <= page.bbox.y1

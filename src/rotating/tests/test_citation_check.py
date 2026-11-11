"""回源 PDF 的引用校验：用真实语料 PDF 验证逐字比对路径。

`data/` 不入库，因此语料不在本机时跳过真实 PDF 的用例；入参缺失、文献未知
与页码越界仍用不需要语料的用例覆盖。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from idr_core.documents import document_id
from idr_core.parsing.pdf import page_text
from idr_rotating.citation_check import SourcePdfVerifier

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "data" / "input"


def smallest_text_pdf() -> tuple[Path, str]:
    """取语料里最小的、第一页有文字层的 PDF，避免测试解析大文献。"""
    if not CORPUS_DIR.is_dir():
        pytest.skip(f"语料不在本机：{CORPUS_DIR}")
    for path in sorted(CORPUS_DIR.glob("*.pdf"), key=lambda item: item.stat().st_size):
        text = page_text(path, 1)
        if len(text) > 80:
            return path, text
    pytest.skip(f"{CORPUS_DIR} 里没有带文字层的 PDF")


def citation_for(path: Path, quote: str, *, page: int = 1) -> dict[str, object]:
    return {
        "evidence_id": f"ev:{document_id(path.read_bytes())}:{page}:0-{len(quote)}",
        "doc_id": document_id(path.read_bytes()),
        "document": path.name,
        "page": page,
        "text": quote,
    }


def test_verifier_accepts_a_quote_that_exists_verbatim_on_the_cited_page() -> None:
    path, text = smallest_text_pdf()
    quote = text[:60]

    result = SourcePdfVerifier(CORPUS_DIR).verify(citation_for(path, quote))

    assert result.ok is True
    assert result.offset == text.find(quote)


def test_verifier_rejects_a_quote_that_is_not_verbatim_on_the_page() -> None:
    path, text = smallest_text_pdf()
    tampered = "篡改" + text[:60]

    result = SourcePdfVerifier(CORPUS_DIR).verify(citation_for(path, tampered))

    assert result.ok is False
    assert "逐字" in result.reason


def test_verifier_rejects_an_unknown_document(tmp_path: Path) -> None:
    citation = {
        "evidence_id": "ev:0000000000000000:1:0-7",
        "doc_id": "0000000000000000",
        "document": None,
        "page": 1,
        "text": "任意一段引文",
    }

    result = SourcePdfVerifier(tmp_path).verify(citation)

    assert result.ok is False
    assert "源 PDF" in result.reason


def test_verifier_rejects_a_page_number_outside_the_document() -> None:
    path, text = smallest_text_pdf()

    result = SourcePdfVerifier(CORPUS_DIR).verify(
        citation_for(path, text[:20], page=9999)
    )

    assert result.ok is False
    assert "页" in result.reason


def test_verifier_rejects_an_empty_quote(tmp_path: Path) -> None:
    citation = {
        "evidence_id": "ev:0000000000000000:1:0-0",
        "doc_id": "0000000000000000",
        "document": None,
        "page": 1,
        "text": "",
    }

    result = SourcePdfVerifier(tmp_path).verify(citation)

    assert result.ok is False
    assert "空" in result.reason


def test_verifier_requires_an_existing_corpus_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="语料目录"):
        SourcePdfVerifier(tmp_path / "missing")

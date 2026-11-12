"""引用回源校验：打开源 PDF 的第 N 页，核对引文是否逐字出现在这一页。

不信任图谱里存的证据文本，也不信任页码字段：文献按内容哈希对回 ``data/input``
里的文件（与入库时的 ``doc_id`` 规则同源），再从 PDF 的文字层重新取该页文字。
因此引用准确率的分母是**实际输出的引用条数**，分子是能在源 PDF 上原样找到的
条数——它不是裁判模型的打分。

页面文字按 (文件, 页码) 缓存在内存里，一次评估只解析真正被引用到的页，而不是
整篇文献。

引文可能只是证据段里的一个片段（引用锚点收窄后的结果）：片段仍然是整段原文
的子串，所以校验器不需要理解锚点——它只回答「这段文字是否逐字出现在所标页
上」。反过来把校验改成「必须覆盖整段」会改变引用准确率的口径，让历史数字
不可比。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from idr_core.documents import document_id
from idr_core.parsing.pdf import page_text


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """一条引用在源 PDF 上的核对结果。``offset`` 是引文在整页文字里的起点。"""

    ok: bool
    reason: str
    offset: int | None = None


class SourcePdfVerifier:
    """按语料目录校验引用；文献名不参与判断，只按内容哈希对文件。"""

    def __init__(self, source_dir: Path) -> None:
        if not source_dir.is_dir():
            raise ValueError(f"语料目录不存在或不是目录：{source_dir}")
        self._source_dir = source_dir
        self._paths: dict[str, Path] | None = None
        self._pages: dict[tuple[Path, int], str] = {}

    def verify(self, citation: Mapping[str, object]) -> CitationCheck:
        """核对一条引用；任何一步对不上都返回失败原因，不抛异常打断整轮评估。"""
        doc_id = str(citation.get("doc_id", ""))
        text = str(citation.get("text", ""))
        if not doc_id:
            return CitationCheck(ok=False, reason="引用缺少文献标识")
        if not text:
            return CitationCheck(ok=False, reason="引文为空，无法回源核对")

        path = self._document_paths().get(doc_id)
        if path is None:
            return CitationCheck(
                ok=False,
                reason=f"语料目录里找不到内容哈希为 {doc_id} 的源 PDF",
            )

        page_value = citation.get("page")
        if not isinstance(page_value, (int, str)):
            return CitationCheck(ok=False, reason="引用缺少可用的页码")
        try:
            page = int(page_value)
        except ValueError:
            return CitationCheck(ok=False, reason="引用缺少可用的页码")

        try:
            text_on_page = self._page_text(path, page)
        except IndexError as error:
            return CitationCheck(ok=False, reason=str(error))
        except Exception as error:  # 损坏的 PDF 不能让整轮评估崩掉
            return CitationCheck(ok=False, reason=f"读取源 PDF 失败：{error}")

        offset = text_on_page.find(text)
        if offset < 0:
            return CitationCheck(
                ok=False,
                reason=f"引文未在源 PDF 第 {page} 页逐字出现（页码错误、跨页或解析差异）",
            )
        return CitationCheck(ok=True, reason="引文与源 PDF 该页逐字一致", offset=offset)

    def _page_text(self, path: Path, page: int) -> str:
        key = (path, page)
        cached = self._pages.get(key)
        if cached is None:
            cached = page_text(path, page)
            self._pages[key] = cached
        return cached

    def _document_paths(self) -> dict[str, Path]:
        """内容哈希 → PDF 路径；同一份内容出现两次时后出现的文件覆盖前者。"""
        if self._paths is None:
            paths: dict[str, Path] = {}
            for path in sorted(self._source_dir.glob("*.pdf")):
                try:
                    paths[document_id(path.read_bytes())] = path
                except OSError:
                    continue
            self._paths = paths
        return self._paths

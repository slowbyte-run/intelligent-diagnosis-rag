"""内存假实现：单元测试不碰 Neo4j、Postgres 与网络。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from idr_core.graph import GraphBatch
from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedDocument, ParsedLine, ParsedPage

DEFAULT_PAGE_BBOX = BBox(0.0, 0.0, 595.0, 842.0)

EMPTY_EXTRACTION = '{"entities": [], "relations": []}'


def parsed_page(
    number: int,
    text: str,
    *,
    line_boxes: Sequence[BBox] | None = None,
    bbox: BBox = DEFAULT_PAGE_BBOX,
) -> ParsedPage:
    """构造一页解析结果；给了行坐标就同时给出可定位的行几何。"""
    if line_boxes is None:
        return ParsedPage(number=number, text=text, bbox=bbox)

    texts = text.split("\n")
    if len(texts) != len(line_boxes):
        raise ValueError(
            f"第 {number} 页有 {len(texts)} 行文字，却给了 {len(line_boxes)} 个行坐标"
        )

    lines: list[ParsedLine] = []
    cursor = 0
    for line_text, line_bbox in zip(texts, line_boxes, strict=True):
        lines.append(
            ParsedLine(
                text=line_text,
                bbox=line_bbox,
                charspan=CharSpan(start=cursor, end=cursor + len(line_text)),
            )
        )
        cursor += len(line_text) + 1
    return ParsedPage(number=number, text=text, bbox=bbox, lines=tuple(lines))


def stacked_boxes(
    count: int,
    *,
    x0: float = 72.0,
    top: float = 72.0,
    width: float = 400.0,
    height: float = 12.0,
    gap: float = 2.0,
) -> list[BBox]:
    """按阅读顺序生成 count 个自上而下的行坐标。"""
    return [
        BBox(
            x0=x0,
            y0=top + index * (height + gap),
            x1=x0 + width,
            y1=top + index * (height + gap) + height,
        )
        for index in range(count)
    ]



class FakeParser:
    """按页码顺序返回固定页面，不读取真实 PDF。"""

    def __init__(self, pages: Sequence[ParsedPage]) -> None:
        self._pages = tuple(pages)

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(pages=self._pages)


class FakeLLM:
    """返回固定响应，并记录收到的提示词。"""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


class FakeEmbedder:
    """占位实现：本票不写向量。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class FakeVectorStore:
    """占位实现：本票不写向量。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, list[float]]] = []

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        self.rows.extend(
            (evidence_id, list(vector))
            for evidence_id, vector in zip(evidence_ids, vectors, strict=True)
        )


class FakeGraphStore:
    """把写入批次留在内存里，供测试断言图谱内容。"""

    def __init__(self) -> None:
        self.batches: list[GraphBatch] = []

    def write_document(self, batch: GraphBatch) -> None:
        self.batches.append(batch)

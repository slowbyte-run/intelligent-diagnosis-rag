"""把解析结果切成带原文定位的证据段。

专利按 ``[0001]`` 形式的段号切分，一段一条证据段；论文按标题层级切分，
证据段不跨章节。两类文档都以页为界——证据段只有一个页码字段，跨页段落
在页边界处断开，各自保留页码与字符区间。

切分规则属于设备领域包：段落号、章节体例与论文/专利判别都是领域知识，
共享核心只提供「页 + 行 + 字符区间 + 版面坐标」这种与领域无关的原语。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_core.parsing import ParsedDocument, ParsedPage
from idr_rotating.document import PATENT_PARAGRAPH, DocumentType

# 标题的三种常见写法：层级编号（1.2 方法）、单级编号（1. 一种… 或 1. Introduction）、
# 中文序号（一、概述）。编号后紧跟数字的写法不算标题（如「3.5 mm 间隙」）。
_HIERARCHICAL_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,3}){1,3}[.、]?\s+\S")
_SIMPLE_HEADING = re.compile(r"^\d{1,2}[.、]\s*(?!\d)\S")
_NUMBER_SPACE_HEADING = re.compile(r"^\d{1,2}\s+(?!\d)\S")
_CN_HEADING = re.compile(r"^[一二三四五六七八九十]+[、.]\s*\S")

_HEADING_WORDS = frozenset(
    {
        "摘要",
        "abstract",
        "关键词",
        "keywords",
        "引言",
        "前言",
        "绪论",
        "introduction",
        "研究背景",
        "related work",
        "技术领域",
        "背景技术",
        "发明内容",
        "具体实施方式",
        "附图说明",
        "实施例",
        "权利要求书",
        "说明书",
        "方法",
        "methods",
        "材料与方法",
        "materials and methods",
        "实验",
        "experiments",
        "结果",
        "results",
        "讨论",
        "discussion",
        "结论",
        "conclusion",
        "conclusions",
        "参考文献",
        "references",
        "致谢",
        "acknowledgements",
        "acknowledgments",
    }
)


@dataclass(frozen=True, slots=True)
class _Line:
    """切分时使用的行视图，兼容解析器给了行坐标与只给了整页文字两种情况。"""

    text: str
    start: int
    end: int
    bbox: BBox


def build_evidence_segments(
    doc_id: str,
    parsed: ParsedDocument,
    document_type: DocumentType,
) -> tuple[EvidenceSegment, ...]:
    """按文档类型切分，返回按文档顺序排列的证据段。"""
    segments: list[EvidenceSegment] = []
    for page in sorted(parsed.pages, key=lambda item: item.number):
        if document_type is DocumentType.PATENT:
            ranges = _merge_heading_only(page, _patent_ranges(page))
        else:
            ranges = _merge_heading_only(page, _block_ranges(page, 0, len(page.text)))

        for raw_start, raw_end in ranges:
            start, end = _tighten(page.text, raw_start, raw_end)
            if start >= end:
                continue
            segment_id = f"ev:{doc_id}:{page.number}:{start}-{end}"
            bbox, warnings = _bbox_for(page, start, end, segment_id)
            segments.append(
                EvidenceSegment(
                    id=segment_id,
                    doc_id=doc_id,
                    page=page.number,
                    bbox=bbox,
                    charspan=CharSpan(start=start, end=end),
                    text=page.text[start:end],
                    warnings=warnings,
                )
            )
    return tuple(segments)


def _merge_heading_only(
    page: ParsedPage, ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """把只有一行标题的区间并入它引出的下一段，避免产出只有标题的证据段。"""
    merged: list[tuple[int, int]] = []
    index = 0
    while index < len(ranges):
        start, end = ranges[index]
        tight_start, tight_end = _tighten(page.text, start, end)
        has_next = index + 1 < len(ranges)
        if (
            has_next
            and tight_start < tight_end
            and _is_heading_only(page.text[tight_start:tight_end])
        ):
            _, next_end = _tighten(page.text, *ranges[index + 1])
            if next_end > tight_start:
                merged.append((tight_start, next_end))
                index += 2
                continue
        merged.append((start, end))
        index += 1
    return merged


def _is_heading_only(text: str) -> bool:
    stripped = text.strip()
    return "\n" not in stripped and _is_heading(stripped)


def _patent_ranges(page: ParsedPage) -> list[tuple[int, int]]:
    """专利优先按段号切；没有段号时退回按标题与段落间距切。"""
    text = page.text
    markers = [match.start() for match in PATENT_PARAGRAPH.finditer(text)]
    if not markers:
        return _block_ranges(page, 0, len(text))

    ranges: list[tuple[int, int]] = []
    if markers[0] > 0:
        # 段号之前的封面、摘要、权利要求书等没有段号，按通用规则再切一层。
        ranges.extend(_block_ranges(page, 0, markers[0]))
    for index, start in enumerate(markers):
        end = markers[index + 1] if index + 1 < len(markers) else len(text)
        # 段号之间可能出现「背景技术」「具体实施方式」这类无段号的章节标题；
        # 在标题处断开，再由 _merge_heading_only 归到它引出的下一段。
        boundaries = [start, *_heading_starts(page, start, end), end]
        ranges.extend(zip(boundaries, boundaries[1:]))
    return ranges


def _block_ranges(page: ParsedPage, start: int, end: int) -> list[tuple[int, int]]:
    """按标题与段落间距切一个页内区间；标题与其后的正文合为一段。"""
    lines = [
        line for line in _page_lines(page) if line.end > start and line.start < end
    ]
    if not lines:
        return [(start, end)]

    heading_indexes = {
        index for index, line in enumerate(lines) if _is_heading(line.text)
    }
    paragraph_starts = _paragraph_starts(lines)
    # 标题本身已经是边界，紧跟标题的那一行不再单独切，以免产出只有标题的空壳段。
    for index in heading_indexes:
        if index + 1 < len(lines):
            paragraph_starts.discard(lines[index + 1].start)

    boundaries = {start, *(lines[index].start for index in heading_indexes)}
    boundaries.update(paragraph_starts)
    ordered = sorted(boundary for boundary in boundaries if start <= boundary < end)
    return [
        (ordered[index], ordered[index + 1] if index + 1 < len(ordered) else end)
        for index in range(len(ordered))
    ]


def _heading_starts(page: ParsedPage, start: int, end: int) -> list[int]:
    return sorted(
        {
            line.start
            for line in _page_lines(page)
            if start < line.start < end and _is_heading(line.text)
        }
    )


def _page_lines(page: ParsedPage) -> tuple[_Line, ...]:
    if page.lines:
        return tuple(
            _Line(
                text=line.text,
                start=line.charspan.start,
                end=line.charspan.end,
                bbox=line.bbox,
            )
            for line in page.lines
        )

    lines: list[_Line] = []
    cursor = 0
    for text in page.text.split("\n"):
        lines.append(
            _Line(
                text=text,
                start=cursor,
                end=cursor + len(text),
                bbox=page.bbox,
            )
        )
        cursor += len(text) + 1
    return tuple(lines)


def _is_heading(line_text: str) -> bool:
    text = line_text.strip()
    if not text or len(text) > 80:
        return False
    if text.endswith(("。", "；", "，", "！", "？", ";", ",", "!", "?")):
        return False

    normalized = text.rstrip("：:").lower()
    if normalized in _HEADING_WORDS:
        return True
    if _HIERARCHICAL_HEADING.match(text):
        return True
    if len(text) <= 50 and _SIMPLE_HEADING.match(text):
        return True
    if len(text) <= 40 and _NUMBER_SPACE_HEADING.match(text):
        return True
    return bool(_CN_HEADING.match(text))


def _paragraph_starts(lines: list[_Line]) -> set[int]:
    """用行距与首行缩进判断段落起点，不依赖空行（文字层里通常没有空行）。

    缩进必须相对**本栏**的左边距判断：双栏或两端对齐的论文里，右栏每一行的
    x0 都远离全页最小左边距，拿全页左边距当基准会把每一行都判成新段落。
    """
    content = [line for line in lines if line.text.strip()]
    if len(content) < 2:
        return set()

    line_height = median(line.bbox.y1 - line.bbox.y0 for line in content)
    char_widths = [
        (line.bbox.x1 - line.bbox.x0) / len(line.text) for line in content
    ]
    char_width = median(char_widths)

    # 按左边界聚类成栏，取每栏的左边距中位数与右边界。
    def column_key(line: _Line) -> int:
        return round(line.bbox.x0 / 24.0)

    columns: dict[int, list[_Line]] = {}
    for line in content:
        columns.setdefault(column_key(line), []).append(line)
    baseline = {key: median(item.bbox.x0 for item in members) for key, members in columns.items()}
    right_edge = {key: max(item.bbox.x1 for item in members) for key, members in columns.items()}

    starts: set[int] = set()
    for previous, current in zip(lines, lines[1:]):
        if not previous.text.strip():
            starts.add(current.start)
            continue
        if not current.text.strip():
            continue

        if column_key(current) != column_key(previous):
            # 换栏即换段落：右栏第一行不能并进左栏最后一段。
            starts.add(current.start)
            continue

        if line_height > 0 and current.bbox.y0 - previous.bbox.y1 > 0.8 * line_height:
            starts.add(current.start)
            continue

        key = column_key(current)
        left = baseline.get(key, current.bbox.x0)
        edge = right_edge.get(key, current.bbox.x1)
        previous_line_was_full = previous.bbox.x1 >= edge - 2 * char_width
        if (
            char_width > 0
            and previous_line_was_full
            and current.bbox.x0 - left > 1.5 * char_width
        ):
            starts.add(current.start)
    return starts


def _bbox_for(
    page: ParsedPage, start: int, end: int, segment_id: str
) -> tuple[BBox, tuple[str, ...]]:
    """返回证据段的版面坐标；定位降级时同时返回可留存的警告。"""
    boxes = [
        line.bbox
        for line in page.lines
        if line.charspan.start < end and line.charspan.end > start
    ]
    if not boxes:
        return page.bbox, (
            f"证据段 {segment_id} 没有可用的行坐标，bbox 回退为整页",
        )
    bbox = BBox(
        x0=min(box.x0 for box in boxes),
        y0=min(box.y0 for box in boxes),
        x1=max(box.x1 for box in boxes),
        y1=max(box.y1 for box in boxes),
    )
    if _has_uncovered_text(page, start, end):
        return bbox, (
            f"证据段 {segment_id} 的部分文字没有行坐标，"
            "bbox 只覆盖可定位的行",
        )
    return bbox, ()


def _has_uncovered_text(page: ParsedPage, start: int, end: int) -> bool:
    """证据段区间里是否有非空白文字没有被任何行坐标覆盖。"""
    if not page.lines:
        # 整页都没有行几何时由「回退为整页」那条警告说明，不重复报。
        return False
    spans = sorted(
        (line.charspan.start, line.charspan.end)
        for line in page.lines
        if line.charspan.end > start and line.charspan.start < end
    )
    cursor = start
    for span_start, span_end in spans:
        if span_start > cursor and page.text[cursor:span_start].strip():
            return True
        cursor = max(cursor, span_end)
    return cursor < end and bool(page.text[cursor:end].strip())


def _tighten(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end

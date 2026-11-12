"""无锚点引用的选句兜底：按断言内容在证据段里挑出最相关的句子窗口。

重抽取完成之前，入库的边没有段内锚点，引用只能落到整条证据段上。整段往往是
一整页版面切出来的块：同一段里讲了四条故障频率，用户看不出哪一句支撑了哪条
断言。这里按断言的三组关键词——头概念名、关系标签、尾概念名——逐句统计命中，
取覆盖命中最多的连续句子窗口作为引用。

规则是纯词法的，不调模型、不引入分词依赖：

- 中文按相邻两字切词（``外圈剥落`` 给出 ``外圈``、``圈剥``、``剥落``），
  单字段名保留单字；英文与数字按完整词元切分（``BPFO``、``FTF``）；
- 一个窗口的得分是三个字段各自「窗口内单句命中的最大词数」之和。一句话覆盖
  得越全得分越高；窗口里多塞不相干的句子不会加分，同分时更窄的窗口胜出，
  因此不会因为原文长就退化成整段；
- 头、尾概念至少要命中一个：只命中「表现为」这类通用关系词，说明不了这句话
  支撑的是哪条断言，按没有命中处理，退回整段。

确定性来自规则本身：切词与命中都是字符串比较，候选窗口按（得分，宽度，起点）
固定比较，同一断言重复运行必然得到同一个窗口。

选出的窗口只在句子边界上取整句，``text[start:end]`` 逐字来自证据段原文，
不插入省略号或分隔符，回源逐字校验因此仍然成立；选句只影响引用展示，
检索命中的判定单位仍然是整条证据段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 句子边界：中文句读与换行。英文句点不作边界——小数、缩写与公式编号会被
# 误切；英文文献里换行已经足够把句子分开。
_SENTENCE_ENDINGS = frozenset("。！？；!?;\n")

# 句末标点后紧跟的收尾符号归入同一句，引用不会只剩半截引号或括号。
_SENTENCE_CLOSERS = frozenset("”’」』）)】》")

# 英文与数字按完整词元匹配：保留 BPFO、ISO 15243、0.5 这类写法。
_LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._\-/][A-Za-z0-9]+)*")

# 中文按连续汉字段切两字词；单字字段名退化为单字匹配。
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

# 三个字段的固定顺序：头概念、关系标签、尾概念。
_HEAD = 0
_RELATION = 1
_TAIL = 2


@dataclass(frozen=True, slots=True)
class CitationWindow:
    """选中的句子窗口，``[start, end)`` 是证据段内的逐字字符区间。"""

    start: int
    end: int


def select_citation_window(
    text: str,
    *,
    head: str,
    relation: str,
    tail: str,
) -> CitationWindow | None:
    """选出与断言词法重合最多的连续句子窗口；没有命中时返回 ``None``。

    返回的区间一定落在句子边界上，且 ``text[start:end]`` 是原文的连续切片。
    窗口覆盖了整条证据段（除首尾空白）时也返回 ``None``：这时与整段引用没有
    区别，不值得多记一个锚点。
    """
    sentences = _split_sentences(text)
    if not sentences:
        return None

    fields = (_units(head), _units(relation), _units(tail))
    hits = tuple(
        tuple(
            frozenset(unit for unit in units if unit in text[start:end].lower())
            for units in fields
        )
        for start, end in sentences
    )

    best_key: tuple[int, int, int] | None = None
    best: tuple[int, int] | None = None
    for first in range(len(sentences)):
        for last in range(first, len(sentences)):
            per_field = tuple(
                max(len(hits[index][field]) for index in range(first, last + 1))
                for field in (_HEAD, _RELATION, _TAIL)
            )
            # 只命中关系标签不足以定位断言，必须命中头或尾概念名。
            if per_field[_HEAD] + per_field[_TAIL] == 0:
                continue
            start = sentences[first][0]
            width = sentences[last][1] - start
            # 得分高的窗口优先；同分取更窄的；再同分取更早的。
            key = (sum(per_field), -width, -start)
            if best_key is None or key > best_key:
                best_key = key
                best = (start, last)
    if best is None:
        return None

    start, last = best
    end = sentences[last][1]
    # 首尾只剩空白时，选的其实就是整段：等价于没有锚点。
    if not text[:start].strip() and not text[end:].strip():
        return None
    return CitationWindow(start=start, end=end)


def _units(term: str) -> frozenset[str]:
    """把一个字段拆成可匹配的词元：英文词元 + 中文两字词。"""
    units: set[str] = set()
    for token in _LATIN_TOKEN.findall(term):
        units.add(token.lower())
    for run in _CJK_RUN.findall(term):
        if len(run) == 1:
            units.add(run)
        else:
            units.update(run[index : index + 2] for index in range(len(run) - 1))
    return frozenset(units)


def _split_sentences(text: str) -> tuple[tuple[int, int], ...]:
    """按句末标点与换行切出句子区间；空白不进入区间。"""
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _SENTENCE_ENDINGS:
            continue
        end = index + 1
        while end < len(text) and text[end] in _SENTENCE_CLOSERS:
            end += 1
        span = _trim(text, start, end)
        if span is not None:
            spans.append(span)
        start = end
    span = _trim(text, start, len(text))
    if span is not None:
        spans.append(span)
    return tuple(spans)


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """去掉区间首尾空白；区间为空时返回 ``None``。"""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None

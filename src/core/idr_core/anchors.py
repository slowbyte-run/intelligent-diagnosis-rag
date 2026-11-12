"""引用锚点：把支撑一条断言的原文片段定位成证据段内的字符区间。

锚点是叠在证据段之上的东西，不产生新的切分，也不改变证据段标识：它只回答
「这条断言依据的是这一段里的哪一句话」。区间在这里一次算定并固化，查询期
不再重新定位，同一个断言每次得到同一个区间。

定位规则只有一条：片段必须**逐字**出现在证据段文本里。片段出现多次时只取
首次并标记歧义——两处一模一样时无法判断哪一处才是断言的本意，猜一处会把
引用指到另一句话上。

规则变了就改 :data:`ANCHOR_RULE_VERSION`：它参与抽取指纹，已完成的文献因此
会重抽，而不是让新旧两种锚点规则的结果混在同一张图里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# 定位规则版本：逐字匹配、首次出现、歧义标记。改动这条规则必须递增，
# 抽取指纹会随之改变，已入库的文献不会被静默跳过。
ANCHOR_RULE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    """一条断言在某个证据段内的逐字出处，``[start, end)`` 左闭右开。

    ``ambiguous`` 表示同一片段在该证据段里出现了不止一次：区间取首次出现，
    标记留给人复核，不据此推断断言的本意。
    """

    evidence_id: str
    start: int
    end: int
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"锚点起点不能为负：{self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"锚点区间必须非空：[{self.start}, {self.end})"
            )


def locate_anchor(
    quote: str, evidence_ids: Sequence[str], evidence_text: Mapping[str, str]
) -> EvidenceAnchor | None:
    """在断言引用的证据段里定位支撑片段。

    按引用顺序取第一个逐字命中片段的证据段：一条断言只带一个片段，落到哪个
    证据段就锚在哪一段，其余引用照旧整段引用。没有给出片段、或片段在任何一段
    里都不是逐字出现时返回 ``None``，由调用方送进待审队列——编造的引用不得
    入库。
    """
    if not quote:
        return None
    for evidence_id in evidence_ids:
        text = evidence_text.get(evidence_id)
        if text is None:
            continue
        start = text.find(quote)
        if start < 0:
            continue
        return EvidenceAnchor(
            evidence_id=evidence_id,
            start=start,
            end=start + len(quote),
            # 重叠出现的同一片段（如 "aaa" 里的 "aa"）同样算多次：宁可标记
            # 歧义，也不能默认第一个位置就是本意。
            ambiguous=text.find(quote, start + 1) >= 0,
        )
    return None

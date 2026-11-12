"""引用锚点：把支撑一条断言的原文片段定位成证据段内的字符区间。

锚点是叠在证据段之上的东西，不产生新的切分，也不改变证据段标识：它只回答
「这条断言依据的是这一段里的哪一句话」。区间在这里一次算定并固化，查询期
不再重新定位，同一个断言每次得到同一个区间。

定位规则只有一条：片段必须**逐字**出现在证据段文本里。片段出现多次时只取
首次并标记歧义——两处一模一样时无法判断哪一处才是断言的本意，猜一处会把
引用指到另一句话上。

规则变了就改 :data:`ANCHOR_RULE_VERSION`。它**不再参与抽取指纹**：提示词已
不再要求模型产出 quote，锚点只在模型自愿给出时附加，改匹配规则不会改变抽
取结果。此前的做法是
会重抽，而不是让新旧两种锚点规则的结果混在同一张图里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# 定位规则版本：逐字匹配、首次出现、歧义标记。改动这条规则必须递增，
# 抽取指纹会随之改变，已入库的文献不会被静默跳过。
ANCHOR_RULE_VERSION = "4"


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
        anchor = _exact_anchor(evidence_id, text, quote)
        if anchor is not None:
            return anchor
        anchor = _normalized_anchor(evidence_id, text, quote)
        if anchor is not None:
            return anchor
    return None


def _exact_anchor(
    evidence_id: str, text: str, quote: str
) -> EvidenceAnchor | None:
    start = text.find(quote)
    if start < 0:
        return None
    return EvidenceAnchor(
        evidence_id=evidence_id,
        start=start,
        end=start + len(quote),
        # 重叠出现的同一片段（如 "aaa" 里的 "aa"）同样算多次：宁可标记
        # 歧义，也不能默认第一个位置就是本意。
        ambiguous=text.find(quote, start + 1) >= 0,
    )


def _normalized_anchor(
    evidence_id: str, text: str, quote: str
) -> EvidenceAnchor | None:
    """空白不敏感地再找一次。

    文字层抽出来的证据常有怪 spacing（``withahammer``、``Fig.,(4)``、中英
    混排），模型复述时按正常间距写，逐字比对就会全数落空。把连续空白折叠成
    一个空格再比，能救回这一大类；区间仍映射回原文偏移，切片依然是原文的
    连续子串。
    """
    normalized_text, offsets = _fold_whitespace(text)
    normalized_quote, _ = _fold_whitespace(quote)
    if not normalized_quote:
        return None
    start = normalized_text.find(normalized_quote)
    if start < 0:
        return None
    end = start + len(normalized_quote)
    if end >= len(offsets):
        return None
    return EvidenceAnchor(
        evidence_id=evidence_id,
        start=offsets[start],
        end=offsets[end],
        ambiguous=normalized_text.find(normalized_quote, start + 1) >= 0,
    )


def _fold_whitespace(text: str) -> tuple[str, list[int]]:
    """去掉全部空白，并给出每个保留字符在原文里的起始偏移。

    文字层既有 ``withahammer`` 这种粘连，也有把一句话拆成多行留下的空白；
    模型复述时的间距与两者都对不上。直接把空白从两侧都去掉，比只折叠连续空
    白能多救回「粘连 vs 正常间距」这一大类。
    """
    kept: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        kept.append(char)
        offsets.append(index)
    offsets.append(len(text))
    return "".join(kept), offsets

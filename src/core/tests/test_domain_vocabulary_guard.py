"""架构约束：共享核心包内不得出现领域词汇（AGENTS.md 第 2 条）。

这条约束靠人自觉守不住——改了共享代码的人往往不会想到自己正在把领域知识
渗进核心。用测试守住它，比在 review 里反复提醒可靠。

守卫必须覆盖 ``src/core`` 下的全部源码与文档，**包括核心自己的测试**：
测试夹具是最容易无意中引入领域词的地方，而夹具里的领域词会把领域概念
带进共享包。守卫文件自身是唯一例外——它必须写出这些词才能检查它们。
"""

from __future__ import annotations

import re
from pathlib import Path

# 中文领域词按子串匹配即可：这些词不会出现在通用流水线概念里。
DOMAIN_WORDS = (
    # 旋转设备
    "轴承",
    "齿轮",
    "转子",
    "联轴器",
    "叶轮",
    "外圈",
    "内圈",
    "滚动体",
    "保持架",
    "齿面",
    "齿根",
    "绕组",
    "剥落",
    "点蚀",
    "磨损",
    "腐蚀",
    "振动",
    # 肺部疾病
    "肺部",
    "疾病",
    "症状",
    "征兆",
)

# 英文领域标识按整词匹配，避免误伤 decoupling 这类通用词。
DOMAIN_IDENTIFIERS = (
    "bearing",
    "gear",
    "rotor",
    "impeller",
    "coupling",
    "fault_mode",
    "damage_form",
    "symptom",
    "bpfo",
    "bpfi",
)

_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:" + "|".join(DOMAIN_IDENTIFIERS) + r")s?\b", re.IGNORECASE
)

CORE_ROOT = Path(__file__).resolve().parents[1]
GUARD_FILE = Path(__file__).resolve()
SCANNED_SUFFIXES = frozenset({".py", ".md"})


def test_core_tree_contains_no_domain_vocabulary() -> None:
    offenders: list[str] = []

    for source_file in sorted(CORE_ROOT.rglob("*")):
        if not source_file.is_file():
            continue
        # 守卫自身必须列出被禁词；它不参与扫描，其余核心文件一个不放过。
        if source_file == GUARD_FILE:
            continue
        if source_file.suffix not in SCANNED_SUFFIXES:
            continue

        text = source_file.read_text(encoding="utf-8")
        relative = source_file.relative_to(CORE_ROOT.parent)
        for word in DOMAIN_WORDS:
            if word in text:
                offenders.append(f"{relative}: {word}")
        for match in _IDENTIFIER_PATTERN.finditer(text):
            offenders.append(f"{relative}: {match.group(0)}")

    assert not offenders, "共享核心出现领域词汇：" + "；".join(offenders)

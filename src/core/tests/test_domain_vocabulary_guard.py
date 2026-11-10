"""架构约束：共享核心包内不得出现领域词汇（AGENTS.md 第 2 条）。

这条约束靠人自觉守不住——改了共享代码的人往往不会想到自己正在把领域知识
渗进核心。用测试守住它，比在 review 里反复提醒可靠。
"""

from pathlib import Path

FORBIDDEN_WORDS = (
    "轴承",
    "齿轮",
    "转子",
    "联轴器",
    "叶轮",
    "肺部",
    "疾病",
    "症状",
    "征兆",
)

CORE_PACKAGE = Path(__file__).resolve().parents[1] / "idr_core"


def test_core_package_contains_no_domain_vocabulary() -> None:
    offenders: list[str] = []

    for source_file in CORE_PACKAGE.rglob("*.py"):
        text = source_file.read_text(encoding="utf-8")
        for word in FORBIDDEN_WORDS:
            if word in text:
                relative = source_file.relative_to(CORE_PACKAGE.parent)
                offenders.append(f"{relative}: {word}")

    assert not offenders, "共享核心包出现领域词汇：" + "；".join(offenders)

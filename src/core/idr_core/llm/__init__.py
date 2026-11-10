"""模型调用端口。

业务代码只依赖这个协议；具体供应商适配器放在 ``idr_core.llm`` 下（AGENTS.md 第 4 条）。
"""

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...

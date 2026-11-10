"""OpenAI 兼容客户端适配器。

业务代码只依赖 ``idr_core.llm.LLMClient`` 协议；换供应商时只改这一层。
本票的测试从不调用真实模型，只验证接线。
"""

from openai import OpenAI


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_seconds
        )
        self._model = model

    def complete(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("模型返回了空内容")
        return content

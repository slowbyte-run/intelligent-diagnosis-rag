"""OpenAI 兼容客户端适配器。

业务代码只依赖 ``idr_core.llm.LLMClient`` 协议；换供应商时只改这一层。
除文本外，本适配器还报告供应商给出的 token 用量与调用参数，供批次报告
记录成本与复现模型组合。测试从不调用真实模型，只验证接线。
"""

from openai import OpenAI

from idr_core.llm import LLMResponse, ModelDescriptor


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_seconds
        )
        self._base_url = base_url
        self._model = model
        self._temperature = temperature

    def complete(self, prompt: str) -> str:
        return self.complete_with_usage(prompt).text

    def complete_with_usage(self, prompt: str) -> LLMResponse:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self._temperature,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("模型返回了空内容")

        usage = completion.usage
        return LLMResponse(
            text=content,
            model=completion.model or self._model,
            input_tokens=None if usage is None else usage.prompt_tokens,
            output_tokens=None if usage is None else usage.completion_tokens,
        )

    def describe(self) -> ModelDescriptor:
        """复现一次运行所需的模型标识与影响输出的调用参数。"""
        return ModelDescriptor(
            model=self._model,
            parameters={
                "base_url": self._base_url,
                "temperature": self._temperature,
                "response_format": "json_object",
            },
        )

"""模型调用端口：文本、用量与耗时的采集。

报告要记录每次模型调用的模型名、输入输出 token 数与耗时，因此端口除了
``complete(prompt) -> str`` 之外还需要一种可选的、能带上用量的调用方式。
"""

from __future__ import annotations

from idr_core.llm import LLMResponse, ModelDescriptor, call_model, describe_model


class InstrumentedLLMStub:
    """实现可选用量能力的客户端，记录收到的提示词。"""

    def __init__(
        self, response: LLMResponse, parameters: dict[str, str | int | float | bool]
    ) -> None:
        self._response = response
        self._parameters = parameters
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        return self.complete_with_usage(prompt).text

    def complete_with_usage(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        return self._response

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(model="deepseek-chat", parameters=self._parameters)


class PlainLLMStub:
    """只实现必需成员的客户端：不报告用量。"""

    def complete(self, prompt: str) -> str:
        return "回答"


def test_call_model_reports_the_provider_usage_of_the_call() -> None:
    llm = InstrumentedLLMStub(
        LLMResponse(
            text='{"entities": []}',
            model="deepseek-chat",
            input_tokens=120,
            output_tokens=30,
        ),
        parameters={},
    )

    text, call = call_model(llm, "抽取提示词")

    assert text == '{"entities": []}'
    assert (call.model, call.input_tokens, call.output_tokens) == (
        "deepseek-chat",
        120,
        30,
    )
    assert call.elapsed_seconds >= 0.0
    assert llm.prompts == ["抽取提示词"]


def test_call_model_keeps_working_for_clients_that_do_not_report_usage() -> None:
    text, call = call_model(PlainLLMStub(), "提示词")

    assert text == "回答"
    assert call.model == "unknown"
    assert (call.input_tokens, call.output_tokens) == (None, None)
    assert call.elapsed_seconds >= 0.0


def test_describe_model_reports_the_parameters_that_affect_the_output() -> None:
    llm = InstrumentedLLMStub(
        LLMResponse(text="{}"),
        parameters={"temperature": 0.0, "response_format": "json_object"},
    )

    assert describe_model(llm) == ModelDescriptor(
        model="deepseek-chat",
        parameters={"temperature": 0.0, "response_format": "json_object"},
    )


def test_describe_model_returns_none_for_clients_without_the_capability() -> None:
    assert describe_model(PlainLLMStub()) is None

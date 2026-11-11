"""OpenAI 兼容适配器：把供应商返回的用量映射成端口的结果对象。

测试用假的 SDK 客户端顶上真实客户端，不联网、不真调模型；验证的是适配器
对外的行为——文本、模型名、token 数与复现参数。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import idr_core.llm.openai_compat as openai_compat
from idr_core.llm import LLMResponse, ModelDescriptor
from idr_core.llm.openai_compat import OpenAICompatibleLLM


def build_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: str | None,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> OpenAICompatibleLLM:
    completion = SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=(
            None
            if prompt_tokens is None or completion_tokens is None
            else SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        ),
    )

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return completion

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai_compat, "OpenAI", FakeOpenAI)
    return OpenAICompatibleLLM(
        base_url="https://example.invalid",
        api_key="sk-test",
        model="deepseek-chat",
    )


def test_completion_reports_the_model_and_token_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = build_client(
        monkeypatch,
        content='{"entities": []}',
        model="deepseek-chat",
        prompt_tokens=1200,
        completion_tokens=340,
    )

    response = llm.complete_with_usage("抽取提示词")

    assert response == LLMResponse(
        text='{"entities": []}',
        model="deepseek-chat",
        input_tokens=1200,
        output_tokens=340,
    )
    assert llm.complete("抽取提示词") == '{"entities": []}'


def test_missing_usage_is_reported_as_unknown_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = build_client(
        monkeypatch,
        content="{}",
        model="deepseek-chat",
        prompt_tokens=None,
        completion_tokens=None,
    )

    response = llm.complete_with_usage("抽取提示词")

    assert (response.input_tokens, response.output_tokens) == (None, None)


def test_describe_reports_the_model_and_parameters_used_for_reproduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = build_client(
        monkeypatch,
        content="{}",
        model="deepseek-chat",
        prompt_tokens=1,
        completion_tokens=1,
    )

    assert llm.describe() == ModelDescriptor(
        model="deepseek-chat",
        parameters={
            "base_url": "https://example.invalid",
            "temperature": 0.0,
            "response_format": "json_object",
            "extra_headers": "",
        },
    )


def test_sends_configured_extra_headers_and_reports_their_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_compat, "OpenAI", FakeOpenAI)

    llm = OpenAICompatibleLLM(
        base_url="https://example.invalid",
        api_key="sk-test",
        model="test-model",
        extra_headers={"x-opencode-session": "idr"},
    )

    assert captured["default_headers"] == {"x-opencode-session": "idr"}
    assert llm.describe().parameters["extra_headers"] == "x-opencode-session"

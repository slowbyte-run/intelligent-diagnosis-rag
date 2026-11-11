"""重排适配器在真实 SiliconFlow 服务上的契约测试。

这条测试**会真实调用计费的重排模型**（两次短语料，token 量很小），默认不跑：

    set -a; source .env; set +a
    uv run pytest -m integration

它守的是响应解析的契约：相关段必须比无关段分高，用量要能落进成本记录。
供应商改了 ``/rerank`` 的响应形状（字段名、下标、meta.tokens）时，这里会先红。
"""

from __future__ import annotations

import os

import pytest

from idr_core.config import load_config
from idr_core.llm.rerank import OpenAICompatibleReranker, rerank_documents

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY（先 source .env）",
    ),
]

QUERY = "设备出现周期性冲击时应该用什么方法确认"
RELEVANT = (
    "Periodic impacts excite high-frequency resonance, and envelope "
    "demodulation is used to confirm this kind of anomaly."
)
IRRELEVANT = (
    "This paper compares image classification benchmarks and reports top-1 "
    "accuracy on a computer vision dataset."
)


def test_real_rerank_scores_the_relevant_passage_above_the_unrelated_one() -> None:
    config = load_config()
    reranker = OpenAICompatibleReranker(
        base_url=config.siliconflow_base_url,
        api_key=config.siliconflow_api_key,
        model=config.rerank_model,
    )

    response, call = rerank_documents(reranker, QUERY, [RELEVANT, IRRELEVANT])

    scores = {item.index: item.score for item in response.scores}
    assert set(scores) == {0, 1}
    assert scores[0] > scores[1]
    assert scores[0] >= 0.5

    # 成本记录要有数：模型名、输入 token 与耗时都来自这一次真实调用。
    assert call.model
    assert call.input_tokens is not None and call.input_tokens > 0
    assert call.output_tokens == 0
    assert call.elapsed_seconds > 0.0

    print(
        f"\n真实重排：model={call.model} input_tokens={call.input_tokens} "
        f"elapsed={call.elapsed_seconds:.3f}s "
        f"scores={{{', '.join(f'{index}: {score:.6f}' for index, score in sorted(scores.items()))}}}"
    )

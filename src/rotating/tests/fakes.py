"""内存假实现：单元测试不碰 Neo4j、Postgres 与网络。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from idr_core.embedding import EMBEDDING_DIM
from idr_core.graph import GraphBatch
from idr_core.llm import LLMResponse, ModelDescriptor
from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedDocument, ParsedLine, ParsedPage
from idr_core.state import DocumentProcessingState

DEFAULT_PAGE_BBOX = BBox(0.0, 0.0, 595.0, 842.0)

EMPTY_EXTRACTION = '{"entities": [], "relations": []}'


def parsed_page(
    number: int,
    text: str,
    *,
    line_boxes: Sequence[BBox] | None = None,
    bbox: BBox = DEFAULT_PAGE_BBOX,
) -> ParsedPage:
    """构造一页解析结果；给了行坐标就同时给出可定位的行几何。"""
    if line_boxes is None:
        return ParsedPage(number=number, text=text, bbox=bbox)

    texts = text.split("\n")
    if len(texts) != len(line_boxes):
        raise ValueError(
            f"第 {number} 页有 {len(texts)} 行文字，却给了 {len(line_boxes)} 个行坐标"
        )

    lines: list[ParsedLine] = []
    cursor = 0
    for line_text, line_bbox in zip(texts, line_boxes, strict=True):
        lines.append(
            ParsedLine(
                text=line_text,
                bbox=line_bbox,
                charspan=CharSpan(start=cursor, end=cursor + len(line_text)),
            )
        )
        cursor += len(line_text) + 1
    return ParsedPage(number=number, text=text, bbox=bbox, lines=tuple(lines))


def stacked_boxes(
    count: int,
    *,
    x0: float = 72.0,
    top: float = 72.0,
    width: float = 400.0,
    height: float = 12.0,
    gap: float = 2.0,
) -> list[BBox]:
    """按阅读顺序生成 count 个自上而下的行坐标。"""
    return [
        BBox(
            x0=x0,
            y0=top + index * (height + gap),
            x1=x0 + width,
            y1=top + index * (height + gap) + height,
        )
        for index in range(count)
    ]


class FakeParser:
    """按页码顺序返回固定页面，不读取真实 PDF，并记录被解析的路径。"""

    def __init__(self, pages: Sequence[ParsedPage]) -> None:
        self._pages = tuple(pages)
        self.calls: list[Path] = []

    def parse(self, path: Path) -> ParsedDocument:
        self.calls.append(path)
        return ParsedDocument(pages=self._pages)


class FakeLLM:
    """返回固定响应或按调用顺序返回响应，并记录收到的提示词。"""

    def __init__(self, response: str | Sequence[str]) -> None:
        self._responses = (response,) if isinstance(response, str) else tuple(response)
        if not self._responses:
            raise ValueError("FakeLLM 至少需要一个响应")
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return self._responses[index]


_EVIDENCE_LINE = re.compile(r"^\[(ev:[^\]]+)\] (.*)$", re.MULTILINE)


def evidence_lines_from_prompt(prompt: str) -> tuple[tuple[str, str], ...]:
    """从抽取提示词里读出本批证据段的 ``(标识, 原文)``，供假模型按批作答。"""
    return tuple(
        (match.group(1), match.group(2))
        for match in _EVIDENCE_LINE.finditer(prompt)
    )


class BatchwiseFakeLLM:
    """模拟「只看一批证据」的模型：每批返回共有概念与本批独有概念各一个。

    证据段标识从提示词里读，因此每个实体只引用本批证据；共有概念在每批都
    出现，用来观察跨批去重与证据合并。
    """

    def __init__(
        self,
        *,
        shared_name: str = "跨批共现部件",
        unique_prefix: str = "本批故障",
    ) -> None:
        self.prompts: list[str] = []
        self.calls: list[tuple[tuple[str, str], ...]] = []
        self._shared_name = shared_name
        self._unique_prefix = unique_prefix

    def complete(self, prompt: str) -> str:
        evidence = evidence_lines_from_prompt(prompt)
        if not evidence:
            raise AssertionError("抽取提示词里没有任何证据段")
        self.prompts.append(prompt)
        self.calls.append(evidence)

        unique_name = f"{self._unique_prefix}{len(self.calls)}"
        return json.dumps(
            {
                "entities": [
                    {
                        "name": self._shared_name,
                        "type": "component",
                        "evidence_ids": [evidence[0][0]],
                        "confidence": 0.9,
                    },
                    {
                        "name": unique_name,
                        "type": "fault_mode",
                        "evidence_ids": [evidence[-1][0]],
                        "confidence": 0.8,
                    },
                ],
                "relations": [
                    {
                        "head": self._shared_name,
                        "type": "has_fault_mode",
                        "tail": unique_name,
                        "evidence_ids": [evidence[-1][0]],
                        "confidence": 0.7,
                    }
                ],
            },
            ensure_ascii=False,
        )


class InstrumentedFakeLLM(FakeLLM):
    """同时报告用量与调用参数的假模型，让报告能记录成本。"""

    def __init__(
        self,
        response: str,
        *,
        model: str = "fake-model",
        input_tokens: int = 100,
        output_tokens: int = 20,
        parameters: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(response)
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._parameters = {} if parameters is None else parameters

    def complete_with_usage(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return LLMResponse(
            text=self._responses[index],
            model=self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(model=self._model, parameters=self._parameters)


class ScriptedLLM:
    """按调用顺序返回脚本化响应；元素是异常时抛出，用来模拟中断或单篇失败。"""

    def __init__(self, steps: Sequence[str | BaseException]) -> None:
        self._steps = tuple(steps)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        step = self._steps[len(self.prompts) - 1]
        if isinstance(step, BaseException):
            raise step
        return step


class FakeEmbedder:
    """按文本长度生成确定性的定长向量，并记录被嵌入的文本。"""

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        self.dimensions = dimensions
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(text))] * self.dimensions for text in texts]


class FakeVectorStore:
    """按证据段标识覆盖写入，模拟真实适配器的 upsert 语义。"""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        for evidence_id, vector in zip(evidence_ids, vectors, strict=True):
            self.vectors[evidence_id] = list(vector)

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None:
        return self.vectors.get(evidence_id)


class FailingVectorStore:
    """写入必失败的向量存储，用于验证图谱不会被半途写入。"""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def upsert_evidence_vectors(
        self, *, evidence_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        raise self.error

    def get_evidence_vector(self, evidence_id: str) -> list[float] | None:
        return None


class FakeGraphStore:
    """把写入批次留在内存里，供测试断言图谱内容。"""

    def __init__(self) -> None:
        self.batches: list[GraphBatch] = []

    def write_document(self, batch: GraphBatch) -> None:
        self.batches.append(batch)


class FakeStateStore:
    """内存版处理状态：按 doc_id 覆盖写入，模拟真实适配器的 upsert 语义。"""

    def __init__(self) -> None:
        self.states: dict[str, DocumentProcessingState] = {}

    def get(self, doc_id: str) -> DocumentProcessingState | None:
        return self.states.get(doc_id)

    def record(self, state: DocumentProcessingState) -> None:
        self.states[state.doc_id] = state

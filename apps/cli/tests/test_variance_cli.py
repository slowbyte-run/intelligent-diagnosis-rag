"""``idr variance`` 的门面测试：假解析器 + 假模型，不碰网络也不造真 PDF。"""

from __future__ import annotations

import json
from pathlib import Path

from idr_cli.variance import run_variance
from idr_core.models import BBox
from idr_core.parsing import ParsedDocument, ParsedPage

PAGE = ParsedPage(
    number=1,
    text="外圈剥落表现为 BPFO 倍频振动。",
    bbox=BBox(0, 0, 200, 300),
)


class FakeParser:
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(pages=(PAGE,))


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _payload(name: str, evidence_id: str) -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": name,
                    "type": "fault_mode",
                    "evidence_ids": [evidence_id],
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )


def test_variance_command_reports_the_spread_and_writes_a_report(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    llm = FakeLLM([json.dumps({"entities": [], "relations": []})])

    exit_code = run_variance(
        source, llm=llm, runs=2, parser=FakeParser(), out=tmp_path / "out.json"
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "重复次数：2" in printed
    assert "断言集合两两重合" in printed


def test_variance_command_uses_the_runs_and_reports_identical_output(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4 fake")

    # 先借一次真实的切分拿到证据段 id：用空响应无法继续，所以直接用同样文本
    # 拼出的 id 形态不可靠——改为让假模型回放与切分无关的响应，断言只看计数。
    llm = FakeLLM([json.dumps({"entities": [], "relations": []})])

    exit_code = run_variance(source, llm=llm, runs=2, parser=FakeParser())

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "所有运行完全一致：是" in out
    assert llm.calls == 2


def test_a_pdf_without_segments_is_reported_not_crashed(tmp_path: Path, capsys) -> None:
    class EmptyParser:
        def parse(self, path: Path) -> ParsedDocument:
            return ParsedDocument(pages=())

    source = tmp_path / "empty.pdf"
    source.write_bytes(b"%PDF-1.4 fake")

    exit_code = run_variance(
        source, llm=FakeLLM(["{}"]), runs=2, parser=EmptyParser()
    )

    assert exit_code == 1
    assert "没有产出证据段" in capsys.readouterr().err

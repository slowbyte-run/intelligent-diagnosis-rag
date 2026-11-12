"""``idr extract-eval`` 的门面测试：假模型回放固定响应，不碰网络与图谱。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idr_cli.extract_eval import run_extract_eval
from idr_cli.main import main

SEGMENT_ID = "ev:aaaaaaaaaaaaaaaa:1:0-10"
SEGMENT_TEXT = "外圈剥落表现为振动。"


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _response(evidence_id: str, name: str = "振动") -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": name,
                    "type": "symptom",
                    "evidence_ids": [evidence_id],
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )


def _write_annotations(
    tmp_path: Path, *, documents: int = 1
) -> Path:
    segments = []
    entities = []
    doc_ids = ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"]
    for index in range(documents):
        doc_id = doc_ids[index]
        segment_id = f"ev:{doc_id}:1:0-10"
        segments.append({"id": segment_id, "page": 1, "text": SEGMENT_TEXT})
        entities.append(
            {"evidence_id": segment_id, "name": "振动", "type": "symptom"}
        )
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "测试标注集",
                "segments": segments,
                "expected_entities": entities,
                "expected_relations": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_extract_eval_reports_assertion_level_metrics_and_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = _write_annotations(tmp_path)
    report_path = tmp_path / "reports" / "extract-eval.json"

    exit_code = run_extract_eval(
        llm=FakeLLM([_response(SEGMENT_ID)]),
        annotations_path=annotations,
        out=report_path,
        model="test-model",
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "断言级召回率" in printed
    assert "断言级精度" in printed
    assert "漏抽" in printed
    assert "类型不符" in printed
    assert "端点不符" in printed
    payload = json.loads(report_path.read_text("utf-8"))
    assert payload["metrics"]["recall"] == 1.0
    assert payload["metrics"]["precision"] == 1.0
    assert payload["model"] == "test-model"
    assert payload["cases"][0]["status"] == "matched"


def test_extract_eval_limit_only_runs_the_first_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = _write_annotations(tmp_path, documents=2)
    llm = FakeLLM([_response(SEGMENT_ID)])

    exit_code = run_extract_eval(
        llm=llm, annotations_path=annotations, limit=1
    )

    assert exit_code == 0
    assert llm.calls == 1
    assert "文献：1" in capsys.readouterr().out


def test_extract_eval_fails_cleanly_on_a_bad_annotation_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")

    exit_code = run_extract_eval(llm=FakeLLM(["{}"]), annotations_path=path)

    assert exit_code == 1
    assert "标注集错误" in capsys.readouterr().err


def test_main_dispatches_the_extract_eval_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "DEEPSEEK_BASE_URL": "https://example.invalid",
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_MODEL": "test-model",
        "SILICONFLOW_BASE_URL": "https://example.invalid",
        "SILICONFLOW_API_KEY": "sk-test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIM": "1024",
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "DATABASE_URL": "postgresql://idr:secret@127.0.0.1:1/idr",
    }.items():
        monkeypatch.setenv(name, value)

    annotations = _write_annotations(tmp_path)
    report_path = tmp_path / "extract-eval.json"

    exit_code = main(
        [
            "extract-eval",
            "--annotations",
            str(annotations),
            "--out",
            str(report_path),
        ],
        extract_eval_llm_factory=lambda config: FakeLLM([_response(SEGMENT_ID)]),
    )

    assert exit_code == 0
    assert "断言级召回率" in capsys.readouterr().out
    assert report_path.exists()

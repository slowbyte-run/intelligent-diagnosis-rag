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


def test_extract_eval_reports_a_distribution_and_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = _write_annotations(tmp_path)
    report_path = tmp_path / "reports" / "extract-eval.json"
    llm = FakeLLM([_response(SEGMENT_ID)])

    exit_code = run_extract_eval(
        llm=llm,
        annotations_path=annotations,
        out=report_path,
        model="test-model",
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    # 默认重复 3 次：单次运行只是一次抽样。
    assert llm.calls == 3
    assert "重复次数：3" in printed
    assert "每次运行：" in printed
    assert "召回率：最低" in printed
    assert "极差" in printed
    assert "两两重合" in printed

    payload = json.loads(report_path.read_text("utf-8"))
    assert payload["schema_version"] == 2
    assert payload["runs"] == 3
    assert payload["model"] == "test-model"
    assert [detail["metrics"]["recall"] for detail in payload["run_details"]] == [
        1.0,
        1.0,
        1.0,
    ]
    assert payload["distribution"]["recall"]["values"] == [1.0, 1.0, 1.0]
    assert payload["distribution"]["recall"]["range"] == 0.0
    assert payload["distribution"]["pairwise_overlap"] == [1.0, 1.0, 1.0]
    assert payload["distribution"]["identical_hits"] is True
    assert payload["run_details"][0]["cases"][0]["status"] == "matched"


def test_extract_eval_reports_the_spread_when_runs_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = _write_annotations(tmp_path)
    report_path = tmp_path / "extract-eval.json"
    # 第 2 次运行抽到的一个「多出的断言」：召回与精度都掉到 0。
    llm = FakeLLM(
        [
            _response(SEGMENT_ID),
            _response(SEGMENT_ID, name="异常噪声"),
            _response(SEGMENT_ID),
        ]
    )

    exit_code = run_extract_eval(
        llm=llm, annotations_path=annotations, runs=3, out=report_path
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "极差 100.0 个百分点" in printed
    payload = json.loads(report_path.read_text("utf-8"))
    assert [detail["metrics"]["recall"] for detail in payload["run_details"]] == [
        1.0,
        0.0,
        1.0,
    ]
    assert payload["distribution"]["recall"]["values"] == [1.0, 0.0, 1.0]
    assert payload["distribution"]["recall"]["range"] == 1.0
    assert payload["distribution"]["pairwise_overlap"] == [0.0, 1.0, 0.0]
    assert payload["distribution"]["identical_hits"] is False


def test_extract_eval_does_not_invent_a_range_for_an_undefined_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """一次输出都没有时精度没有分母；报告要如实留空，不能拿 0 当极差。"""
    annotations = _write_annotations(tmp_path)
    report_path = tmp_path / "extract-eval.json"
    empty_output = json.dumps({"entities": [], "relations": []})

    exit_code = run_extract_eval(
        llm=FakeLLM([empty_output]),
        annotations_path=annotations,
        runs=2,
        out=report_path,
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "精度：—（有运行的分数没有分母，无法比较极差）" in printed
    payload = json.loads(report_path.read_text("utf-8"))
    assert payload["distribution"]["precision"]["values"] == [None, None]
    assert payload["distribution"]["precision"]["range"] is None
    assert payload["distribution"]["recall"]["values"] == [0.0, 0.0]


def test_extract_eval_limit_only_runs_the_first_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = _write_annotations(tmp_path, documents=2)
    llm = FakeLLM([_response(SEGMENT_ID)])

    exit_code = run_extract_eval(
        llm=llm, annotations_path=annotations, limit=1, runs=2
    )

    assert exit_code == 0
    assert llm.calls == 2
    assert "文献：1" in capsys.readouterr().out


def test_extract_eval_rejects_a_single_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations = _write_annotations(tmp_path)

    exit_code = run_extract_eval(
        llm=FakeLLM([_response(SEGMENT_ID)]), annotations_path=annotations, runs=1
    )

    assert exit_code == 1
    assert "至少为 2" in capsys.readouterr().err


def test_extract_eval_fails_cleanly_on_a_bad_annotation_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")

    exit_code = run_extract_eval(llm=FakeLLM(["{}"]), annotations_path=path)

    assert exit_code == 1
    assert "标注集错误" in capsys.readouterr().err


def test_main_rejects_a_single_run_before_calling_the_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["extract-eval", "--runs", "1"])

    assert "至少为 2" in capsys.readouterr().err


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
    llm = FakeLLM([_response(SEGMENT_ID)])

    exit_code = main(
        [
            "extract-eval",
            "--annotations",
            str(annotations),
            "--runs",
            "2",
            "--out",
            str(report_path),
        ],
        extract_eval_llm_factory=lambda config: llm,
    )

    assert exit_code == 0
    assert llm.calls == 2
    assert "重复次数：2" in capsys.readouterr().out
    payload = json.loads(report_path.read_text("utf-8"))
    assert payload["runs"] == 2
    assert len(payload["distribution"]["pairwise_overlap"]) == 1

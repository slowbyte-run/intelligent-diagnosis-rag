"""eval 子命令的默认路径：只读图谱快照，不加载配置、不连 Neo4j。

真实快照与金标准集都在版本库里，因此这些测试不造数据：它们验证 CLI 的外部
行为——默认走快照、报告显式标注「图谱已固定」、同一快照两次摘要一致、金标准
绑定的校验和不符时失败并说明原因。旧实时图谱路径另见 ``test_eval_cli.py``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from idr_core.config import Config
from idr_cli.eval import run_snapshot_eval
from idr_cli.main import main
from idr_rotating.evaluation import DEFAULT_GOLDEN_PATH

SNAPSHOT_CHECKSUM = "sha256:343227369845f2927e78a16e9a9460d37b4eace9e21869a4a7144bed3bd02e58"


def golden_with_checksum(path: Path, checksum: str) -> Path:
    payload = json.loads(DEFAULT_GOLDEN_PATH.read_text(encoding="utf-8"))
    payload["snapshot"]["checksum"] = checksum
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_main_eval_defaults_to_the_snapshot_without_loading_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "eval-snapshot.json"

    def forbidden_config(config: Config) -> NoReturn:
        raise AssertionError("快照评估不应加载配置")

    def forbidden_ask_deps(config: Config, source_dir: Path) -> NoReturn:
        raise AssertionError("快照评估不应构建真实依赖")

    def forbidden_snapshot_graph(config: Config) -> NoReturn:
        raise AssertionError("快照评估不应连接图数据库")

    exit_code = main(
        ["eval", "--out", str(report_path)],
        deps_factory=forbidden_config,
        ask_deps_factory=forbidden_ask_deps,
        snapshot_graph_factory=forbidden_snapshot_graph,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "图谱已固定" in output
    assert "引用文本自洽率" in output
    assert "结果摘要：sha256:" in output

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "retrieval-snapshot-eval"
    assert payload["graph_frozen"] is True
    assert payload["graph"]["checksum"] == SNAPSHOT_CHECKSUM
    assert payload["golden"]["checksum"] == SNAPSHOT_CHECKSUM


def test_two_consecutive_cli_runs_produce_the_same_digest(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert run_snapshot_eval(out_path=first) == 0
    assert run_snapshot_eval(out_path=second) == 0

    left = json.loads(first.read_text(encoding="utf-8"))
    right = json.loads(second.read_text(encoding="utf-8"))

    assert left["result_digest"] == right["result_digest"]
    assert left["metrics"] == right["metrics"]
    assert left["cases"] == right["cases"]


def test_checksum_mismatch_exits_nonzero_and_names_both_checksums(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    golden = golden_with_checksum(
        tmp_path / "golden.json", "sha256:" + "0" * 64
    )
    report_path = tmp_path / "report.json"

    exit_code = run_snapshot_eval(golden_path=golden, out_path=report_path)

    message = capsys.readouterr().err
    assert exit_code == 1
    assert "不是同一次导出" in message
    assert SNAPSHOT_CHECKSUM in message
    assert "sha256:" + "0" * 64 in message
    assert not report_path.exists()


def test_pdf_verification_is_only_marked_as_checked_when_corpus_is_given(
    tmp_path: Path,
) -> None:
    without_corpus = tmp_path / "without-pdf.json"
    with_corpus = tmp_path / "with-pdf.json"
    (tmp_path / "corpus").mkdir()

    assert run_snapshot_eval(out_path=without_corpus) == 0
    assert (
        run_snapshot_eval(
            out_path=with_corpus, corpus_dir=tmp_path / "corpus"
        )
        == 0
    )

    left = json.loads(without_corpus.read_text(encoding="utf-8"))
    right = json.loads(with_corpus.read_text(encoding="utf-8"))

    assert left["citation_pdf_verifiable"]["checked"] is False
    assert left["citation_pdf_verifiable"]["citation_pdf_accuracy"] is None

    assert right["citation_pdf_verifiable"]["checked"] is True
    assert right["citation_pdf_verifiable"]["unavailable_reason"] is None
    # 空语料目录里没有 PDF：回源失败，但快照口径的文本自洽率不因此改变。
    assert right["citation_pdf_verifiable"]["citation_pdf_accuracy"] == 0.0
    assert (
        right["metrics"]["citation_text_consistency"]
        == left["metrics"]["citation_text_consistency"]
    )

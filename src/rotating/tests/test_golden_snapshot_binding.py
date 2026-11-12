"""金标准集与图谱快照的绑定：评的是哪张图，必须能机械核对。

金标准集记录它绑定的快照内容校验和；加载期只校验格式，评估入口再核对文件。
两者不符时评估直接失败并说明原因——悄悄换一张图会让「这次改动变好还是变坏」
重新变成噪声，这正是这轮要消灭的东西。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from idr_core.graph.snapshot import GraphSnapshot, load_snapshot
from idr_rotating.evaluation import (
    GoldenSet,
    GoldenSetError,
    SnapshotBindingError,
    check_snapshot_binding,
    load_golden_set,
)
from idr_rotating.snapshot_export import DEFAULT_SNAPSHOT_PATH

SNAPSHOT_CHECKSUM = "sha256:" + "0" * 64


def golden_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "snapshot": {"checksum": SNAPSHOT_CHECKSUM},
        "questions": [
            {
                "id": "T-01",
                "class": "factual",
                "question": "阿尔法有什么征兆？",
                "expected_concepts": ["concept:alpha"],
                "expected_evidence_ids": [
                    "ev:0123456789abcdef:1:0-5",
                ],
            }
        ],
    }
    payload.update(overrides)
    return payload


def write_golden(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_shipped_golden_set_binds_the_committed_snapshot() -> None:
    """提交进版本库的题与图必须互相对应；快照重导后这一条会先红。"""
    golden = load_golden_set()
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)

    check_snapshot_binding(
        golden, snapshot, snapshot_path=DEFAULT_SNAPSHOT_PATH
    )

    assert golden.snapshot_checksum == snapshot.checksum
    assert golden.path.name == "golden_questions.json"
    assert len(golden.questions) == 50


def test_load_golden_set_returns_the_bound_checksum(tmp_path: Path) -> None:
    path = write_golden(tmp_path / "golden.json", golden_payload())

    golden = load_golden_set(path)

    assert golden.snapshot_checksum == SNAPSHOT_CHECKSUM
    assert golden.path == path
    assert golden.questions[0].id == "T-01"


def test_golden_set_without_a_snapshot_binding_is_rejected(tmp_path: Path) -> None:
    payload = golden_payload()
    del payload["snapshot"]

    path = write_golden(tmp_path / "golden.json", payload)

    with pytest.raises(GoldenSetError, match="快照"):
        load_golden_set(path)


def test_golden_set_rejects_a_malformed_checksum(tmp_path: Path) -> None:
    path = write_golden(
        tmp_path / "golden.json",
        golden_payload(snapshot={"checksum": "sha256:not-a-digest"}),
    )

    with pytest.raises(GoldenSetError, match="校验和"):
        load_golden_set(path)


def test_golden_set_rejects_unknown_snapshot_binding_fields(tmp_path: Path) -> None:
    path = write_golden(
        tmp_path / "golden.json",
        golden_payload(snapshot={"checksum": SNAPSHOT_CHECKSUM, "path": "elsewhere"}),
    )

    with pytest.raises(GoldenSetError, match="未知字段"):
        load_golden_set(path)


def test_old_golden_schema_without_a_binding_is_rejected_with_guidance(
    tmp_path: Path,
) -> None:
    payload = golden_payload()
    payload["schema_version"] = 1
    del payload["snapshot"]
    path = write_golden(tmp_path / "golden.json", payload)

    with pytest.raises(GoldenSetError, match="snapshot.checksum"):
        load_golden_set(path)


def test_checksum_mismatch_fails_loudly_with_both_values_and_next_step() -> None:
    golden = load_golden_set()
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    mismatched = GoldenSet(
        path=golden.path,
        snapshot_checksum="sha256:" + "f" * 64,
        questions=golden.questions,
    )

    with pytest.raises(SnapshotBindingError) as caught:
        check_snapshot_binding(
            mismatched, snapshot, snapshot_path=DEFAULT_SNAPSHOT_PATH
        )

    message = str(caught.value)
    assert "sha256:" + "f" * 64 in message
    assert snapshot.checksum in message
    assert str(golden.path) in message
    assert "snapshot.checksum" in message


def test_matching_binding_is_accepted() -> None:
    golden = load_golden_set()
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)

    check_snapshot_binding(golden, snapshot, snapshot_path=DEFAULT_SNAPSHOT_PATH)


def test_binding_check_uses_the_snapshot_checksum_not_the_file_path() -> None:
    """同一内容换个路径仍算同一张图；绑定的是内容校验和，不是文件名。"""
    golden = load_golden_set()
    snapshot: GraphSnapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)

    check_snapshot_binding(
        golden, snapshot, snapshot_path=Path("/tmp/同一份内容的副本.json")
    )

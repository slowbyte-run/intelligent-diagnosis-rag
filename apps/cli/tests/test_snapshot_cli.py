"""snapshot 子命令：一条命令把线上图谱的评分闭包导出成快照文件。

用内存假图跑完整链路，不依赖 Neo4j；真实图谱上的导出由集成测试覆盖。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from idr_core.anchors import EvidenceAnchor
from idr_core.graph.read import Concept, EvidenceRecord, FactEdge
from idr_core.graph.snapshot import load_snapshot
from idr_core.models import CharSpan

from idr_cli.main import main
from idr_cli.snapshot import run_snapshot

DOC = "0123456789abcdef"
ALPHA = "concept:alpha"
BETA = "concept:beta"
EVIDENCE_ID = f"ev:{DOC}:1:0-4"
OTHER_EVIDENCE_ID = f"ev:{DOC}:2:10-14"


class FakeGraph:
    def __init__(self) -> None:
        self._concepts = (
            Concept(id=ALPHA, name="节点甲", type="kind"),
            Concept(id=BETA, name="节点乙", type="kind"),
        )
        self._evidence = (
            EvidenceRecord(
                id=EVIDENCE_ID,
                doc_id=DOC,
                page=1,
                text="证据原文",
                charspan=CharSpan(0, 4),
            ),
            EvidenceRecord(
                id=OTHER_EVIDENCE_ID,
                doc_id=DOC,
                page=2,
                text="另一段原文",
                charspan=CharSpan(10, 14),
            ),
        )

    def facts_from(
        self, concept_id: str, relation_types: Sequence[str]
    ) -> tuple[FactEdge, ...]:
        if concept_id != ALPHA or "has_component" not in relation_types:
            return ()
        return (
            FactEdge(
                head_id=ALPHA,
                relation="has_component",
                tail_id=BETA,
                evidence_ids=(EVIDENCE_ID,),
                confidence=0.8,
                doc_id=DOC,
                anchors=(EvidenceAnchor(evidence_id=EVIDENCE_ID, start=0, end=2),),
            ),
        )

    def concepts_by_ids(self, concept_ids: Sequence[str]) -> tuple[Concept, ...]:
        wanted = set(concept_ids)
        return tuple(item for item in self._concepts if item.id in wanted)

    def evidence_by_ids(
        self, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceRecord, ...]:
        wanted = set(evidence_ids)
        return tuple(item for item in self._evidence if item.id in wanted)

    def evidence_by_page(self, doc_id: str, page: int) -> tuple[EvidenceRecord, ...]:
        return tuple(
            item for item in self._evidence if (item.doc_id, item.page) == (doc_id, page)
        )


def write_golden(
    path: Path,
    *,
    concept_ids: tuple[str, ...] = (ALPHA,),
    evidence_ids: tuple[str, ...] = (EVIDENCE_ID,),
) -> Path:
    payload = {
        "schema_version": 2,
        "snapshot": {"checksum": "sha256:" + "0" * 64},
        "questions": [
            {
                "id": "T-01",
                "class": "factual",
                "question": "这是什么",
                "expected_concepts": list(concept_ids),
                "expected_evidence_ids": list(evidence_ids),
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_run_snapshot_writes_a_verifiable_file_and_prints_its_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json")
    target = tmp_path / "snapshot.json"

    exit_code = run_snapshot(FakeGraph(), golden_path=golden, out_path=target)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "校验和：sha256:" in output
    assert "1 题" in output
    snapshot = load_snapshot(target)
    assert snapshot.checksum in output
    assert [concept.id for concept in snapshot.concepts] == [ALPHA, BETA]
    assert [edge.tail_id for edge in snapshot.edges] == [BETA]


def test_run_snapshot_surfaces_golden_evidence_that_no_relation_edge_cites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """金标准绑定、但不在任何关系边上的证据要显式报告，不能悄悄埋着。"""
    golden = write_golden(
        tmp_path / "golden.json",
        concept_ids=(ALPHA,),
        evidence_ids=(EVIDENCE_ID, OTHER_EVIDENCE_ID),
    )

    exit_code = run_snapshot(
        FakeGraph(), golden_path=golden, out_path=tmp_path / "snapshot.json"
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "不在任何关系边上" in output
    snapshot = load_snapshot(tmp_path / "snapshot.json")
    assert {item.id for item in snapshot.evidence} == {
        EVIDENCE_ID,
        OTHER_EVIDENCE_ID,
    }


def test_run_snapshot_fails_cleanly_when_a_golden_binding_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = write_golden(tmp_path / "golden.json", concept_ids=("concept:ghost",))
    target = tmp_path / "snapshot.json"

    exit_code = run_snapshot(FakeGraph(), golden_path=golden, out_path=target)

    assert exit_code == 1
    assert "concept:ghost" in capsys.readouterr().err
    assert not target.exists()


def test_main_dispatches_snapshot_subcommand(
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

    golden = write_golden(tmp_path / "golden.json")
    target = tmp_path / "snapshot.json"

    exit_code = main(
        ["snapshot", "--golden", str(golden), "--out", str(target)],
        snapshot_graph_factory=lambda config: FakeGraph(),
    )

    assert exit_code == 0
    assert target.exists()
    assert "校验和" in capsys.readouterr().out

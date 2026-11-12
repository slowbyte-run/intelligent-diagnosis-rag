"""snapshot 子命令：一条命令把线上图谱的评分闭包导出成版本化快照。

命令只读图谱：走 ``GraphReader`` 读端口，不写任何节点或边，也不碰语料。
输出是一份 JSON（格式见 ``idr_core.graph.snapshot``）与它的内容校验和；文件
默认落在领域包内、与金标准集同级，进版本库后可复核「这次评分用的是哪版图谱」。

导出会报告金标准绑定、但不在任何关系边上的证据段。这类期望在当前图路径下
无法被引用（证据只挂在概念→证据内部链路上），快照仍带上正文以便加载期校验；
报告把它们显式列出来，避免评分结果里出现无法归因的失分。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from idr_core.config import Config
from idr_core.graph.neo4j import Neo4jGraphStore
from idr_core.graph.read import GraphReader
from idr_core.graph.snapshot import GraphSnapshot, SnapshotError
from idr_rotating.evaluation import GoldenQuestion, GoldenSetError, load_golden_set
from idr_rotating.snapshot_export import DEFAULT_SNAPSHOT_PATH, write_export


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    """命令输出所需的数字；与渲染分开，测试不必解析排版。"""

    questions: int
    concepts: int
    edges: int
    evidence: int
    uncited_expected_evidence: tuple[str, ...]
    checksum: str
    path: Path
    size: int


def build_snapshot_graph(config: Config) -> Neo4jGraphStore:
    """快照只依赖图读取：不接嵌入、不接向量表，导出期间不联网。"""
    return Neo4jGraphStore(
        config.neo4j_uri, config.neo4j_user, config.neo4j_password
    )


def run_snapshot(
    graph: GraphReader,
    *,
    golden_path: Path | None = None,
    out_path: Path | None = None,
) -> int:
    """导出快照并打印摘要；配置或闭包不完整时返回 1。"""
    target = DEFAULT_SNAPSHOT_PATH if out_path is None else Path(out_path)
    try:
        questions = load_golden_set(golden_path).questions
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = write_export(graph, questions, path=target)
    except GoldenSetError as error:
        print(f"金标准集错误：{error}", file=sys.stderr)
        return 1
    except SnapshotError as error:
        print(f"快照导出失败：{error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"无法写快照 {target}：{error}", file=sys.stderr)
        return 1

    print(render_summary(snapshot_summary(questions, snapshot, target)))
    return 0


def snapshot_summary(
    questions: Sequence[GoldenQuestion], snapshot: GraphSnapshot, target: Path
) -> SnapshotSummary:
    """命令输出所需的数字；与渲染分开，测试不必解析排版。"""
    cited = {
        evidence_id for edge in snapshot.edges for evidence_id in edge.evidence_ids
    }
    expected = {
        evidence_id
        for question in questions
        for evidence_id in question.expected_evidence_ids
    }
    return SnapshotSummary(
        questions=len(questions),
        concepts=len(snapshot.concepts),
        edges=len(snapshot.edges),
        evidence=len(snapshot.evidence),
        uncited_expected_evidence=tuple(sorted(expected - cited)),
        checksum=snapshot.checksum,
        path=target,
        size=target.stat().st_size,
    )


def render_summary(summary: SnapshotSummary) -> str:
    lines = [
        f"图谱快照：{summary.questions} 题 / {summary.concepts} 个概念 / "
        f"{summary.edges} 条关系边 / {summary.evidence} 段证据"
    ]
    if summary.uncited_expected_evidence:
        shown = "、".join(summary.uncited_expected_evidence[:3])
        suffix = (
            f" 等 {len(summary.uncited_expected_evidence)} 条"
            if len(summary.uncited_expected_evidence) > 3
            else ""
        )
        lines.append(
            f"注意：金标准绑定的 {len(summary.uncited_expected_evidence)} 条证据"
            f"不在任何关系边上（{shown}{suffix}）；"
            "正文已包含，但图路径无法引用它们"
        )
    lines.append(f"文件：{summary.path}（{_human_size(summary.size)}）")
    lines.append(f"校验和：{summary.checksum}")
    return "\n".join(lines)


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"

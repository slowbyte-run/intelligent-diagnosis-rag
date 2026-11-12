"""图谱快照：把评分需要的图闭包冻成一份可版本化、可校验的文件。

评估要回答「这次改动让检索变好还是变坏」，就不能让图谱本身在两次评估之间
变化。快照把一次导出时刻的闭包写成文件：概念节点、关系边（含文献、证据段、
置信度与锚点）以及这些边引用的证据段原文。检索评估只读快照，因此不依赖图
数据库，也不需要重新抽取。

**确定性是格式的一部分。** 同一份输入导出的字节必须完全一致：所有有排序自由
的集合都在 :meth:`GraphSnapshot.build` 里定序，文件里不写时间戳、主机名或
绝对路径。校验和是对**内容**（去掉 ``checksum`` 字段后的规范化 JSON）算的
SHA-256，不是对排版算的：重新缩进文件不会改变校验和，改一个字的原文会。

**快照不是图数据库的备份。** 它只含导出方声明的闭包（见 ``closure`` 段），
不会替调用方补齐闭包之外的节点；读侧必须按闭包语义使用它。文件格式自描述：
``format`` 标识格式，``schema_version`` 标识代际，``closure`` 说明这份快照
覆盖了什么范围。读快照的图端口适配器不在本模块——这里只负责格式、定序、
校验和与加载。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idr_core.anchors import EvidenceAnchor
from idr_core.graph.read import Concept, EvidenceRecord, FactEdge
from idr_core.models import CharSpan

# 文件头里的格式标识：读侧据此拒绝不是快照的 JSON。
SNAPSHOT_FORMAT = "idr-graph-snapshot"

# 格式代际：字段增删、语义变化时递增；读侧遇到不认识的版本直接报错。
SNAPSHOT_SCHEMA_VERSION = 1

_CHECKSUM_PREFIX = "sha256:"

_CLOSURE_FIELDS = frozenset(
    {"seed_concept_ids", "relation_types", "structural_relations", "max_hops"}
)
_CONCEPT_FIELDS = frozenset({"id", "name", "type"})
_EDGE_FIELDS = frozenset(
    {"head_id", "relation", "tail_id", "doc_id", "evidence_ids", "confidence", "anchors"}
)
_ANCHOR_FIELDS = frozenset({"evidence_id", "start", "end", "ambiguous"})
_EVIDENCE_FIELDS = frozenset({"id", "doc_id", "page", "charspan", "text"})
_TOP_LEVEL_FIELDS = frozenset(
    {"format", "schema_version", "checksum", "closure", "concepts", "edges", "evidence"}
)


class SnapshotError(ValueError):
    """快照格式或闭包不完整：错误必须让导出/加载失败，而不是交出一份假数据。"""


@dataclass(frozen=True, slots=True)
class SnapshotClosure:
    """这份快照覆盖的图范围；它随文件留档，让「评分用的是哪张图」可核对。

    ``seed_concept_ids`` 是闭包起点，``relation_types`` 是导出纳入的关系类型，
    ``structural_relations`` 是可以两跳中转的关系子集，``max_hops`` 是遍历
    上界。它们由导出方决定并在导出时定序；格式层不解释领域语义。
    """

    seed_concept_ids: tuple[str, ...]
    relation_types: tuple[str, ...]
    structural_relations: tuple[str, ...]
    max_hops: int

    def __post_init__(self) -> None:
        if not self.seed_concept_ids:
            raise SnapshotError("快照闭包至少要有一个起点概念")
        if not self.relation_types:
            raise SnapshotError("快照闭包必须声明纳入的关系类型")
        if self.max_hops < 1:
            raise SnapshotError(
                f"快照闭包的 max_hops 必须是正整数，收到 {self.max_hops}"
            )
        unknown = set(self.structural_relations) - set(self.relation_types)
        if unknown:
            raise SnapshotError(
                "快照闭包的结构关系中转类型必须属于纳入的关系类型："
                f"{sorted(unknown)}"
            )

    def normalised(self) -> SnapshotClosure:
        """按标识排序后的副本；集合语义的字段不保留输入顺序。"""
        return SnapshotClosure(
            seed_concept_ids=tuple(sorted(set(self.seed_concept_ids))),
            relation_types=tuple(sorted(set(self.relation_types))),
            structural_relations=tuple(sorted(set(self.structural_relations))),
            max_hops=self.max_hops,
        )


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """一份图谱快照：闭包说明 + 概念 + 关系边 + 证据段原文。

    字段直接复用图读取端口的数据类，快照因此就是「一次冻结的读模型」：
    边保留每篇文献一条并行边的原貌（不聚合），锚点与证据段平行，证据段带
    页面文字里的字符区间。``build`` 负责定序与完整性校验，构造后不可变。
    """

    closure: SnapshotClosure
    concepts: tuple[Concept, ...]
    edges: tuple[FactEdge, ...]
    evidence: tuple[EvidenceRecord, ...]

    @classmethod
    def build(
        cls,
        *,
        closure: SnapshotClosure,
        concepts: Sequence[Concept],
        edges: Sequence[FactEdge],
        evidence: Sequence[EvidenceRecord],
    ) -> GraphSnapshot:
        """定序并校验后构造；任何引用不完整都失败，不生成半份快照。"""
        sorted_concepts = _unique_by_id(concepts, what="概念")
        sorted_evidence = _unique_by_id(evidence, what="证据段")
        evidence_ids = set(sorted_evidence)
        concept_ids = set(sorted_concepts)

        normalised_edges = tuple(
            _normalise_edge(edge, concept_ids=concept_ids, evidence_ids=evidence_ids)
            for edge in edges
        )
        ordered_edges = tuple(
            sorted(
                normalised_edges,
                key=lambda edge: (
                    edge.head_id,
                    edge.relation,
                    edge.tail_id,
                    edge.doc_id,
                ),
            )
        )
        _reject_duplicate_edges(ordered_edges)

        return cls(
            closure=closure.normalised(),
            concepts=tuple(sorted_concepts.values()),
            edges=ordered_edges,
            evidence=tuple(sorted_evidence.values()),
        )

    @property
    def checksum(self) -> str:
        """内容校验和，与文件里 ``checksum`` 字段同源。"""
        return snapshot_checksum(self)


def snapshot_checksum(snapshot: GraphSnapshot) -> str:
    """对快照内容算 SHA-256；排版、缩进与键顺序都不参与。"""
    return _checksum(_content_payload(snapshot))


def serialize_snapshot(snapshot: GraphSnapshot) -> str:
    """把快照写成文件文本；同内容永远得到同一串字节。"""
    content = _content_payload(snapshot)
    # 校验和覆盖 format 与 schema_version，换版本不能悄悄沿用旧哈希。
    checksum = _checksum(content)
    document = {
        "format": content["format"],
        "schema_version": content["schema_version"],
        "checksum": checksum,
        "closure": content["closure"],
        "concepts": content["concepts"],
        "edges": content["edges"],
        "evidence": content["evidence"],
    }
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def write_snapshot(path: Path, snapshot: GraphSnapshot) -> str:
    """写文件并返回内容校验和；目录不存在由调用方负责。"""
    Path(path).write_text(serialize_snapshot(snapshot), encoding="utf-8")
    return snapshot.checksum


def load_snapshot(path: Path) -> GraphSnapshot:
    """读取并校验一份快照：先核对内容校验和，再解析字段。

    校验和用文件里的原始内容核对，因此手改过任何一个字都会失败；校验通过后
    才重建数据类，字段缺失、拼错或版本不认识同样报错，不退回默认值。
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        raise SnapshotError(f"无法读取快照 {target}：{error}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise SnapshotError(f"快照不是合法 JSON：{error}") from error
    return snapshot_from_document(document)


def snapshot_from_document(document: object) -> GraphSnapshot:
    """从已解析的 JSON 对象重建快照；也供测试与下游复用。"""
    if not isinstance(document, Mapping):
        raise SnapshotError("快照顶层必须是 JSON 对象")
    unknown = set(document) - _TOP_LEVEL_FIELDS
    if unknown:
        raise SnapshotError(f"快照顶层有未知字段：{sorted(unknown)}")

    if document.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError(
            f"不是图谱快照：format 应为 {SNAPSHOT_FORMAT!r}，"
            f"收到 {document.get('format')!r}"
        )
    version = document.get("schema_version")
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(
            f"快照 schema_version 不支持：{version!r}"
            f"（本工具支持 {SNAPSHOT_SCHEMA_VERSION}）"
        )

    declared = document.get("checksum")
    if not isinstance(declared, str) or not declared:
        raise SnapshotError("快照缺少 checksum 字段")

    # 先解析再核对哈希：字段拼错、版本不认识这类结构错误，比「被改过」更
    # 具体地指出问题；结构正确但内容对不上才是改动的信号。
    parsed = (
        _parse_closure(document.get("closure")),
        _parse_concepts(document.get("concepts")),
        _parse_edges(document.get("edges")),
        _parse_evidence(document.get("evidence")),
    )
    actual = _document_checksum(document)
    if declared != actual:
        raise SnapshotError(
            f"快照校验和不符：文件声明 {declared}，按内容重算得到 {actual}；"
            "文件被改动过，或导出不完整"
        )

    return GraphSnapshot.build(
        closure=parsed[0],
        concepts=parsed[1],
        edges=parsed[2],
        evidence=parsed[3],
    )


def _content_payload(snapshot: GraphSnapshot) -> dict[str, Any]:
    """校验和与文件正文共用的内容视图，键序固定、集合已定序。"""
    return {
        "format": SNAPSHOT_FORMAT,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "closure": {
            "seed_concept_ids": list(snapshot.closure.seed_concept_ids),
            "relation_types": list(snapshot.closure.relation_types),
            "structural_relations": list(snapshot.closure.structural_relations),
            "max_hops": snapshot.closure.max_hops,
        },
        "concepts": [
            {"id": concept.id, "name": concept.name, "type": concept.type}
            for concept in snapshot.concepts
        ],
        "edges": [
            {
                "head_id": edge.head_id,
                "relation": edge.relation,
                "tail_id": edge.tail_id,
                "doc_id": edge.doc_id,
                "evidence_ids": list(edge.evidence_ids),
                "confidence": edge.confidence,
                "anchors": [
                    {
                        "evidence_id": anchor.evidence_id,
                        "start": anchor.start,
                        "end": anchor.end,
                        "ambiguous": anchor.ambiguous,
                    }
                    for anchor in edge.anchors
                ],
            }
            for edge in snapshot.edges
        ],
        "evidence": [
            {
                "id": record.id,
                "doc_id": record.doc_id,
                "page": record.page,
                "charspan": [record.charspan.start, record.charspan.end],
                "text": record.text,
            }
            for record in snapshot.evidence
        ],
    }


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """规范化 JSON：键排序、无多余空白、不转义非 ASCII。

    ``ensure_ascii=False`` 让中文原样参与哈希；读侧用同一套参数才能复算。
    """
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _checksum(payload: Mapping[str, Any]) -> str:
    return _CHECKSUM_PREFIX + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _document_checksum(document: Mapping[str, Any]) -> str:
    content = {key: value for key, value in document.items() if key != "checksum"}
    return _checksum(content)


def _unique_by_id(
    records: Sequence[Concept] | Sequence[EvidenceRecord], *, what: str
) -> dict[str, Any]:
    """按标识去重并排序；同一标识出现两次是数据错误，不猜保留哪一个。"""
    indexed: dict[str, Any] = {}
    for record in records:
        if record.id in indexed:
            raise SnapshotError(f"快照里的{what}标识重复：{record.id}")
        indexed[record.id] = record
    return {key: indexed[key] for key in sorted(indexed)}


def _normalise_edge(
    edge: FactEdge, *, concept_ids: set[str], evidence_ids: set[str]
) -> FactEdge:
    """一条边的证据清单与锚点定序，并核对它引用的节点与原文都在快照里。"""
    for endpoint in (edge.head_id, edge.tail_id):
        if endpoint not in concept_ids:
            raise SnapshotError(
                f"边引用了快照里没有的概念：{endpoint}"
                f"（{edge.head_id} -{edge.relation}-> {edge.tail_id}）"
            )
    if not edge.evidence_ids:
        raise SnapshotError(
            "没有证据的关系边不能进快照：聚合层本来就会丢弃它"
            f"（{edge.head_id} -{edge.relation}-> {edge.tail_id}）"
        )
    missing = [item for item in edge.evidence_ids if item not in evidence_ids]
    if missing:
        raise SnapshotError(
            f"边引用的证据段不在快照里：{missing}"
            f"（{edge.head_id} -{edge.relation}-> {edge.tail_id}）"
        )
    anchors = _normalise_anchors(edge, evidence_ids=set(edge.evidence_ids))
    return FactEdge(
        head_id=edge.head_id,
        relation=edge.relation,
        tail_id=edge.tail_id,
        evidence_ids=tuple(sorted(set(edge.evidence_ids))),
        # 统一成 float：JSON 里的 1 与 1.0 解析回来是同一种内容，校验和才稳定。
        confidence=float(edge.confidence),
        doc_id=edge.doc_id,
        anchors=anchors,
    )


def _normalise_anchors(
    edge: FactEdge, *, evidence_ids: set[str]
) -> tuple[EvidenceAnchor, ...]:
    by_evidence: dict[str, EvidenceAnchor] = {}
    for anchor in edge.anchors:
        if anchor.evidence_id not in evidence_ids:
            raise SnapshotError(
                f"锚点指向该边没有引用的证据段：{anchor.evidence_id}"
                f"（{edge.head_id} -{edge.relation}-> {edge.tail_id}）"
            )
        previous = by_evidence.get(anchor.evidence_id)
        if previous is not None and previous != anchor:
            raise SnapshotError(
                f"同一条边对同一证据段给出了多个锚点：{anchor.evidence_id}"
                f"（{edge.head_id} -{edge.relation}-> {edge.tail_id}）"
            )
        by_evidence[anchor.evidence_id] = anchor
    return tuple(by_evidence[key] for key in sorted(by_evidence))


def _reject_duplicate_edges(edges: Sequence[FactEdge]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        key = (edge.head_id, edge.relation, edge.tail_id, edge.doc_id)
        if key in seen:
            raise SnapshotError(
                "快照里的关系边重复（同一对概念、同一关系、同一文献）："
                f"{key[0]} -{key[1]}-> {key[2]}（{key[3]}）"
            )
        seen.add(key)


def _parse_closure(value: object) -> SnapshotClosure:
    document = _expect_mapping(value, "closure")
    _reject_unknown(document, _CLOSURE_FIELDS, "closure")
    max_hops = document.get("max_hops")
    if isinstance(max_hops, bool) or not isinstance(max_hops, int):
        raise SnapshotError(f"closure.max_hops 必须是整数，收到 {max_hops!r}")
    return SnapshotClosure(
        seed_concept_ids=_string_list(document.get("seed_concept_ids"), "closure.seed_concept_ids"),
        relation_types=_string_list(document.get("relation_types"), "closure.relation_types"),
        structural_relations=_string_list(
            document.get("structural_relations"), "closure.structural_relations"
        ),
        max_hops=max_hops,
    )


def _parse_concepts(value: object) -> tuple[Concept, ...]:
    entries = _expect_list(value, "concepts")
    concepts: list[Concept] = []
    for index, entry in enumerate(entries):
        where = f"concepts[{index}]"
        document = _expect_mapping(entry, where)
        _reject_unknown(document, _CONCEPT_FIELDS, where)
        concepts.append(
            Concept(
                id=_string(document.get("id"), f"{where}.id"),
                name=_string(document.get("name"), f"{where}.name"),
                type=_string(document.get("type"), f"{where}.type"),
            )
        )
    return tuple(concepts)


def _parse_edges(value: object) -> tuple[FactEdge, ...]:
    entries = _expect_list(value, "edges")
    edges: list[FactEdge] = []
    for index, entry in enumerate(entries):
        where = f"edges[{index}]"
        document = _expect_mapping(entry, where)
        _reject_unknown(document, _EDGE_FIELDS, where)
        confidence = document.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise SnapshotError(
                f"{where}.confidence 必须是数字，收到 {confidence!r}"
            )
        edges.append(
            FactEdge(
                head_id=_string(document.get("head_id"), f"{where}.head_id"),
                relation=_string(document.get("relation"), f"{where}.relation"),
                tail_id=_string(document.get("tail_id"), f"{where}.tail_id"),
                doc_id=_string(document.get("doc_id"), f"{where}.doc_id"),
                evidence_ids=_string_list(
                    document.get("evidence_ids"), f"{where}.evidence_ids"
                ),
                confidence=float(confidence),
                anchors=_parse_anchors(document.get("anchors"), where),
            )
        )
    return tuple(edges)


def _parse_anchors(value: object, where: str) -> tuple[EvidenceAnchor, ...]:
    entries = _expect_list(value, f"{where}.anchors")
    anchors: list[EvidenceAnchor] = []
    for index, entry in enumerate(entries):
        position = f"{where}.anchors[{index}]"
        document = _expect_mapping(entry, position)
        _reject_unknown(document, _ANCHOR_FIELDS, position)
        start = _integer(document.get("start"), f"{position}.start")
        end = _integer(document.get("end"), f"{position}.end")
        ambiguous = document.get("ambiguous")
        if not isinstance(ambiguous, bool):
            raise SnapshotError(f"{position}.ambiguous 必须是布尔值")
        try:
            anchors.append(
                EvidenceAnchor(
                    evidence_id=_string(
                        document.get("evidence_id"), f"{position}.evidence_id"
                    ),
                    start=start,
                    end=end,
                    ambiguous=ambiguous,
                )
            )
        except ValueError as error:
            raise SnapshotError(f"{position} 非法：{error}") from error
    return tuple(anchors)


def _parse_evidence(value: object) -> tuple[EvidenceRecord, ...]:
    entries = _expect_list(value, "evidence")
    records: list[EvidenceRecord] = []
    for index, entry in enumerate(entries):
        where = f"evidence[{index}]"
        document = _expect_mapping(entry, where)
        _reject_unknown(document, _EVIDENCE_FIELDS, where)
        span = _expect_list(document.get("charspan"), f"{where}.charspan")
        if len(span) != 2:
            raise SnapshotError(f"{where}.charspan 必须是 [start, end] 两个整数")
        records.append(
            EvidenceRecord(
                id=_string(document.get("id"), f"{where}.id"),
                doc_id=_string(document.get("doc_id"), f"{where}.doc_id"),
                page=_integer(document.get("page"), f"{where}.page"),
                text=_string(document.get("text"), f"{where}.text"),
                charspan=CharSpan(
                    start=_integer(span[0], f"{where}.charspan[0]"),
                    end=_integer(span[1], f"{where}.charspan[1]"),
                ),
            )
        )
    return tuple(records)


def _expect_mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotError(f"{where} 必须是 JSON 对象")
    return value


def _expect_list(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SnapshotError(f"{where} 必须是 JSON 数组")
    return value


def _reject_unknown(
    document: Mapping[str, Any], allowed: frozenset[str], where: str
) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise SnapshotError(f"{where} 有未知字段：{sorted(unknown)}")


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotError(f"{where} 必须是非空字符串，收到 {value!r}")
    return value


def _string_list(value: object, where: str) -> tuple[str, ...]:
    entries = _expect_list(value, where)
    return tuple(_string(entry, f"{where}[{index}]") for index, entry in enumerate(entries))


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotError(f"{where} 必须是整数，收到 {value!r}")
    return value

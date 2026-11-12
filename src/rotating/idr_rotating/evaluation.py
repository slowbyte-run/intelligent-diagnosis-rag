"""金标准集与三项指标：把「检索策略变好还是变坏」变成可重复的数字。

金标准集是领域资产（``golden_questions.json``）。schema_version 2 起，它记录
自己绑定的图谱快照内容校验和（``snapshot.checksum``）：检索评估在打分前核对
实际快照，不符即失败——题与图错配时继续跑只会悄悄错分。评分本身仍是机械的：

- **检索命中率**（``retrieval_hit_rate``）：可答题（``expect_refusal=false``）中，
  至少命中一个期望证据段的比例；跨文献题还要求每个期望文献都至少出现一条引用。
  分母是可答题数，拒答对照题不参与。期望概念覆盖作为独立诊断项保留，不参与
  命中判定——概念是某一次抽取的产物，抽取有采样方差。
- **引用准确率**（``citation_accuracy``）：答案实际输出的全部引用中，回源 PDF
  后能在所标页逐字找到的比例（micro，分母是引用条数）。回源由
  :class:`SourcePdfVerifier` 完成，不用裁判模型；没有输出引用时为 ``None``。
  快照路径没有 PDF，改用 :class:`SnapshotCitationVerifier` 核对「引文是否为
  快照原文的逐字片段」，报告里叫 ``citation_text_consistency``，不与本项混用
  （见 ``snapshot_eval`` 与 ``docs/retrieval-eval.md``）。
- **拒答正确率**（``refusal_correctness``）：全部题中，答案路径是否为拒答与
  ``expect_refusal`` 一致的比例。它同时惩罚「该答却拒」和「该拒却答」。

答案只按 ``answer_to_dict`` 的 dict 形态读取，不依赖 ``Answer`` 数据类。评估器
只校验自己真正要用的字段（``path`` / ``focus`` / ``facts[].head|tail`` /
``comparison`` 的两个锚点与各分区 / ``citations[]``），对**新增**字段（例如
#15 的 ``Fact.path``、``truncated``）保持兼容；删除或改名这些字段才会失败。
对比型答案的结论在 ``comparison`` 下、``facts`` 为空，概念集合必须把两侧锚点
与分区里的事实都算进来，否则每一道对比题都会缺概念。评估过程只读图谱、只读
语料，唯一写出的是报告文件。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idr_core.graph.snapshot import GraphSnapshot
from idr_rotating.citation_check import CitationCheck

# 金标准集的格式代际：2 起记录它绑定的图谱快照校验和。旧版（1）没有绑定，
# 无法判断这批评分对应哪张图，加载期直接拒绝并提示补哪一项。
GOLDEN_SCHEMA_VERSION = 2
EVALUATION_SCHEMA_VERSION = 1
QUESTION_CLASSES = ("factual", "comparative", "cross_document")
REFUSAL_REASONS = ("no_concept", "no_evidence")
DEFAULT_GOLDEN_PATH = Path(__file__).with_name("golden_questions.json")

METRIC_DEFINITIONS = {
    "retrieval_hit_rate": (
        "可答题中，至少命中一个期望证据段、且跨文献题每个期望文献至少一条引用"
        "的比例；分母是可答题数。概念覆盖单独报告，不参与命中判定。"
    ),
    "citation_accuracy": (
        "答案实际输出的全部引用中，能在源 PDF 所标页逐字找到的比例（micro）；"
        "没有输出引用时为 null。回源由 PDF 重新抽取，不用裁判模型。这是实时"
        "图谱路径（--live-graph）的口径；快照路径另见 citation_text_consistency。"
    ),
    "refusal_correctness": (
        "全部题中，答案路径是否为拒答与 expect_refusal 一致的比例；"
        "同时惩罚该答却拒与该拒却答。"
    ),
}

_EVIDENCE_ID_PATTERN = re.compile(r"^ev:([0-9a-f]{16}):")
_CONCEPT_ID_PATTERN = re.compile(r"^concept:[a-z][a-z0-9_]*$")
_SNAPSHOT_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SNAPSHOT_BINDING_FIELDS = frozenset({"checksum"})
_KNOWN_QUESTION_FIELDS = frozenset(
    {
        "id",
        "class",
        "question",
        "expected_concepts",
        "expected_evidence_ids",
        "expect_refusal",
        "expected_refusal_reason",
        "note",
    }
)


class GoldenSetError(ValueError):
    """金标准集格式错误：让出题错误在加载阶段失败，而不是评出一组假数字。"""


class SnapshotBindingError(GoldenSetError):
    """金标准集与图谱快照不是同一次导出。

    评分的前提是题目与图对应；两者不匹配时继续跑只会悄悄错分，因此评估入口
    直接失败，并说明是哪两个校验和对不上、下一步该改哪一项。
    """


class EvaluationError(ValueError):
    """答案形态不符合评估契约（例如 ``schema_version`` 不匹配）。"""


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """一份金标准集：题目 + 它绑定的图谱快照内容校验和。

    校验和是快照**内容**的 SHA-256（见 :mod:`idr_core.graph.snapshot`），
    换排版不变、改一个字的证据原文就变。评估入口拿实际快照校验和与它比对，
    不符即失败；加载本身不比对文件，因为导出快照时金标准集可能暂时落后。
    """

    path: Path
    snapshot_checksum: str
    questions: tuple[GoldenQuestion, ...]


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """一道金标准题及其机械评分所需的全部绑定。"""

    id: str
    question_class: str
    question: str
    expected_concepts: tuple[str, ...]
    expected_evidence_ids: tuple[str, ...]
    expect_refusal: bool = False
    expected_refusal_reason: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class CaseResult:
    """一道题的判分细节；报告里逐条留档，便于定位是哪一环坏了。"""

    id: str
    question_class: str
    question: str
    path: str
    detail: str | None
    expect_refusal: bool
    refusal_correct: bool
    hit: bool | None
    concept_ok: bool
    expected_concepts: tuple[str, ...]
    missing_concepts: tuple[str, ...]
    evidence_ok: bool
    expected_evidence_ids: tuple[str, ...]
    matched_evidence_ids: tuple[str, ...]
    document_ok: bool
    expected_documents: tuple[str, ...]
    missing_documents: tuple[str, ...]
    citations_total: int
    citations_verified: int
    citation_failures: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """按题型分组的三项指标中的命中率部分；引用与拒答仍按全表统计。"""

    total: int
    answerable: int
    retrieval_hits: int
    retrieval_hit_rate: float | None


@dataclass(frozen=True, slots=True)
class Metrics:
    question_count: int
    answerable_count: int
    retrieval_hits: int
    retrieval_hit_rate: float | None
    citations_total: int
    citations_verified: int
    citation_accuracy: float | None
    refusal_correct: int
    refusal_correctness: float | None
    path_counts: dict[str, int]
    by_class: dict[str, ClassMetrics]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    generated_at: str
    metrics: Metrics
    cases: tuple[CaseResult, ...]


def document_of_evidence(evidence_id: str) -> str:
    """从证据段 ID 取出文献标识；ID 由入库层按 ``ev:<doc_id>:<页>:<区间>`` 生成。"""
    match = _EVIDENCE_ID_PATTERN.match(evidence_id)
    if match is None:
        raise GoldenSetError(
            f"证据段 ID 格式不对（应为 ev:<16 位文献标识>:<页>:<区间>）：{evidence_id}"
        )
    return match.group(1)


def load_golden_set(path: Path | None = None) -> GoldenSet:
    """读取并校验金标准集；任何结构错误都在这里失败，不进入评分。

    返回的 :class:`GoldenSet` 同时带上它绑定的快照校验和；两者是否相符由
    评估入口用 :func:`check_snapshot_binding` 核对。
    """
    target = DEFAULT_GOLDEN_PATH if path is None else Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as error:
        raise GoldenSetError(f"无法读取金标准集 {target}：{error}") from error
    except json.JSONDecodeError as error:
        raise GoldenSetError(f"金标准集不是合法 JSON：{error}") from error

    if not isinstance(payload, dict):
        raise GoldenSetError("金标准集顶层必须是 JSON 对象")
    version = payload.get("schema_version")
    if version != GOLDEN_SCHEMA_VERSION:
        raise GoldenSetError(
            f"金标准集 schema_version 不支持：{version!r}"
            f"（本工具支持 {GOLDEN_SCHEMA_VERSION}）；"
            "schema_version 1 的旧格式没有记录绑定的快照校验和，"
            "请在顶层补上 snapshot.checksum 后升级为 2"
        )
    snapshot_checksum = _snapshot_checksum(payload.get("snapshot"))
    entries = payload.get("questions")
    if not isinstance(entries, list) or not entries:
        raise GoldenSetError("金标准集缺少非空的 questions 数组")

    questions = tuple(
        _question_from_entry(entry, index) for index, entry in enumerate(entries)
    )
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise GoldenSetError(f"金标准集里题目 ID 重复：{question.id}")
        seen.add(question.id)
    return GoldenSet(
        path=target,
        snapshot_checksum=snapshot_checksum,
        questions=questions,
    )


def _snapshot_checksum(value: object) -> str:
    """解析顶部 ``snapshot`` 绑定；缺少或格式不对都在加载期失败。"""
    if not isinstance(value, Mapping):
        raise GoldenSetError(
            "金标准集缺少 snapshot 对象；每份金标准集必须记录它评的是哪一份"
            "图谱快照（snapshot.checksum），否则分数无法归因"
        )
    unknown = set(value) - _SNAPSHOT_BINDING_FIELDS
    if unknown:
        raise GoldenSetError(f"金标准集的 snapshot 有未知字段：{sorted(unknown)}")
    checksum = value.get("checksum")
    if not isinstance(checksum, str) or not _SNAPSHOT_CHECKSUM_PATTERN.fullmatch(
        checksum
    ):
        raise GoldenSetError(
            "金标准集的 snapshot.checksum（图谱快照内容校验和）必须是 "
            f"'sha256:<64 位小写十六进制>'，收到 {checksum!r}"
        )
    return checksum


def check_snapshot_binding(
    golden: GoldenSet, snapshot: GraphSnapshot, *, snapshot_path: Path
) -> None:
    """核对快照内容校验和与金标准集的绑定；不符时失败并说明原因。

    比较的是快照内容校验和，与文件路径、排版无关：同一份图复制到哪里都算
    同一张图，重新导出后哪怕文件名不变也不再是同一张图。
    """
    actual = snapshot.checksum
    if actual == golden.snapshot_checksum:
        return
    raise SnapshotBindingError(
        "金标准集与图谱快照不是同一次导出："
        f"{golden.path} 绑定 {golden.snapshot_checksum}，"
        f"而 {snapshot_path} 的内容校验和是 {actual}。"
        "这通常意味着图谱重新导出过，或者 --snapshot 指错了文件；"
        "拿一套题去评另一张图，分数无法归因，因此评估不继续。"
        "确认要换图时，重新导出快照并把金标准集的 snapshot.checksum "
        f"更新为 {actual}，作为一次显式的评估资产改动提交。"
    )


def _question_from_entry(entry: object, index: int) -> GoldenQuestion:
    if not isinstance(entry, Mapping):
        raise GoldenSetError(f"第 {index + 1} 道题不是 JSON 对象")
    unknown = set(entry) - _KNOWN_QUESTION_FIELDS
    if unknown:
        raise GoldenSetError(f"第 {index + 1} 道题有未知字段：{sorted(unknown)}")

    question_id = entry.get("id")
    if not isinstance(question_id, str) or not question_id:
        raise GoldenSetError(f"第 {index + 1} 道题缺少非空 id")
    question_class = entry.get("class")
    if question_class not in QUESTION_CLASSES:
        raise GoldenSetError(
            f"{question_id} 的 class 必须是 {QUESTION_CLASSES} 之一，收到 {question_class!r}"
        )
    question_text = entry.get("question")
    if not isinstance(question_text, str) or not question_text.strip():
        raise GoldenSetError(f"{question_id} 的 question 不能为空")

    concepts = _string_list(entry.get("expected_concepts"), "expected_concepts", question_id)
    evidence_ids = _string_list(
        entry.get("expected_evidence_ids"), "expected_evidence_ids", question_id
    )
    for concept_id in concepts:
        if _CONCEPT_ID_PATTERN.fullmatch(concept_id) is None:
            raise GoldenSetError(
                f"{question_id} 绑定了非规范概念节点 ID：{concept_id}；"
                "金标准集只能引用本体规范实例（concept:<实例 ID>）"
            )
    for evidence_id in evidence_ids:
        document_of_evidence(evidence_id)

    expect_refusal = entry.get("expect_refusal", False)
    if not isinstance(expect_refusal, bool):
        raise GoldenSetError(f"{question_id} 的 expect_refusal 必须是布尔值")
    reason_value = entry.get("expected_refusal_reason")
    note_value = entry.get("note", "")
    if not isinstance(note_value, str):
        raise GoldenSetError(f"{question_id} 的 note 必须是字符串")

    if expect_refusal:
        if concepts or evidence_ids:
            raise GoldenSetError(
                f"{question_id} 标记为拒答题，却绑定了期望概念或期望证据；"
                "拒答题只能期望空答案"
            )
        if reason_value not in REFUSAL_REASONS:
            raise GoldenSetError(
                f"{question_id} 的 expected_refusal_reason 必须是 "
                f"{REFUSAL_REASONS} 之一，收到 {reason_value!r}"
            )
        reason: str | None = str(reason_value)
    else:
        if reason_value is not None:
            raise GoldenSetError(
                f"{question_id} 不是拒答题，不能写 expected_refusal_reason"
            )
        if not concepts:
            raise GoldenSetError(f"{question_id} 的 expected_concepts 不能为空")
        if not evidence_ids:
            raise GoldenSetError(f"{question_id} 的 expected_evidence_ids 不能为空")
        reason = None

    if question_class == "cross_document":
        documents = {document_of_evidence(item) for item in evidence_ids}
        if len(documents) < 2:
            raise GoldenSetError(
                f"{question_id} 是跨文献题，但期望证据只来自 {len(documents)} 篇文献；"
                "跨文献题至少绑定两篇文献的证据段"
            )

    return GoldenQuestion(
        id=question_id,
        question_class=str(question_class),
        question=question_text,
        expected_concepts=concepts,
        expected_evidence_ids=evidence_ids,
        expect_refusal=expect_refusal,
        expected_refusal_reason=reason,
        note=note_value,
    )


def _string_list(value: object, field: str, question_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise GoldenSetError(f"{question_id} 的 {field} 必须是字符串数组")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise GoldenSetError(f"{question_id} 的 {field} 含空值或非字符串")
        entries.append(item)
    if len(set(entries)) != len(entries):
        raise GoldenSetError(f"{question_id} 的 {field} 有重复项")
    return tuple(entries)


def evaluate_golden(
    questions: Sequence[GoldenQuestion],
    *,
    answer_for: Callable[[str], Mapping[str, Any]],
    verify_citation: Callable[[dict[str, Any]], CitationCheck],
    generated_at: str | None = None,
) -> EvaluationReport:
    """逐题调用答案函数并判分；``answer_for`` 返回 ``answer_to_dict`` 的形态。"""
    if not questions:
        raise GoldenSetError("没有可评估的题目")

    cases = tuple(
        _score_case(question, answer_for(question.question), verify_citation)
        for question in questions
    )
    return EvaluationReport(
        generated_at=generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        metrics=_metrics(cases),
        cases=cases,
    )


def _score_case(
    question: GoldenQuestion,
    answer: Mapping[str, Any],
    verify_citation: Callable[[dict[str, Any]], CitationCheck],
) -> CaseResult:
    path, detail, concepts, citations = _read_answer(answer)

    citation_failures: list[tuple[str, str]] = []
    for citation in citations:
        check = verify_citation(citation)
        if not check.ok:
            citation_failures.append((str(citation.get("evidence_id", "")), check.reason))

    cited_evidence = tuple(
        dict.fromkeys(str(citation.get("evidence_id", "")) for citation in citations)
    )
    cited_documents = tuple(
        dict.fromkeys(str(citation.get("doc_id", "")) for citation in citations)
    )
    matched_evidence = tuple(
        evidence_id
        for evidence_id in question.expected_evidence_ids
        if evidence_id in cited_evidence
    )
    missing_concepts = tuple(
        concept_id
        for concept_id in question.expected_concepts
        if concept_id not in concepts
    )
    concept_ok = not missing_concepts
    evidence_ok = bool(matched_evidence)

    expected_documents: tuple[str, ...] = ()
    missing_documents: tuple[str, ...] = ()
    document_ok = True
    if question.question_class == "cross_document" and not question.expect_refusal:
        expected_documents = tuple(
            dict.fromkeys(
                document_of_evidence(evidence_id)
                for evidence_id in question.expected_evidence_ids
            )
        )
        missing_documents = tuple(
            document
            for document in expected_documents
            if document not in cited_documents
        )
        document_ok = not missing_documents

    refusal_correct = (path == "refusal") == question.expect_refusal
    # 命中只看证据：证据段由固定的切分产生，重抽语料也不会变；而期望概念是
    # 某一次抽取的产物，抽取本身有采样方差（同一篇同提示词两次重抽，实体
    # 9→13、关系 7→11），把概念并进命中判据会让分数随抽样浮动几十个点，测
    # 到的是运气而不是能力。概念覆盖作为独立诊断项保留在报告里。
    hit = (
        None
        if question.expect_refusal
        else evidence_ok and document_ok
    )

    return CaseResult(
        id=question.id,
        question_class=question.question_class,
        question=question.question,
        path=path,
        detail=detail,
        expect_refusal=question.expect_refusal,
        refusal_correct=refusal_correct,
        hit=hit,
        concept_ok=concept_ok,
        expected_concepts=question.expected_concepts,
        missing_concepts=missing_concepts,
        evidence_ok=evidence_ok,
        expected_evidence_ids=question.expected_evidence_ids,
        matched_evidence_ids=matched_evidence,
        document_ok=document_ok,
        expected_documents=expected_documents,
        missing_documents=missing_documents,
        citations_total=len(citations),
        citations_verified=len(citations) - len(citation_failures),
        citation_failures=tuple(citation_failures),
    )


def _read_answer(
    answer: Mapping[str, Any],
) -> tuple[str, str | None, tuple[str, ...], tuple[dict[str, Any], ...]]:
    """只读 dict 契约：路径、概念集合与引用；字段缺失就报错而不是猜。"""
    version = answer.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise EvaluationError(f"答案缺少合法的 schema_version：{version!r}")
    path = answer.get("path")
    if not isinstance(path, str) or not path:
        raise EvaluationError("答案缺少 path")
    detail_value = answer.get("detail")
    detail = detail_value if isinstance(detail_value, str) else None

    concepts: list[str] = []
    focus = answer.get("focus")
    if isinstance(focus, Mapping):
        focus_id = _nested_concept_id(focus)
        if focus_id is not None:
            concepts.append(focus_id)

    comparison = answer.get("comparison")
    if comparison is not None:
        if not isinstance(comparison, Mapping):
            raise EvaluationError("答案的 comparison 不是对象")
        concepts.extend(_comparison_concept_ids(comparison))

    facts = answer.get("facts")
    if not isinstance(facts, list):
        raise EvaluationError("答案的 facts 不是数组")
    for fact in facts:
        if not isinstance(fact, Mapping):
            raise EvaluationError("答案的 facts 含非对象元素")
        concepts.extend(_fact_concept_ids(fact))

    citations = answer.get("citations")
    if not isinstance(citations, list):
        raise EvaluationError("答案的 citations 不是数组")
    citation_rows: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise EvaluationError("答案的 citations 含非对象元素")
        citation_rows.append(dict(citation))

    return path, detail, tuple(dict.fromkeys(concepts)), tuple(citation_rows)


def _comparison_concept_ids(comparison: Mapping[str, Any]) -> tuple[str, ...]:
    """对比答案提到的全部概念：两个锚点、共同项与各分区的事实。

    对比答案的 ``facts`` 为空（结论在 ``comparison`` 下），不读这里会让每一道
    对比题都缺概念、被误判成未命中。
    """
    concept_ids: list[str] = []
    for side in ("left", "right"):
        match = comparison.get(side)
        if not isinstance(match, Mapping):
            raise EvaluationError(f"答案的 comparison 缺少 {side}")
        concept_id = _nested_concept_id(match)
        if concept_id is None:
            raise EvaluationError(f"答案的 comparison.{side} 缺少 concept.id")
        concept_ids.append(concept_id)

    shared = comparison.get("shared")
    if not isinstance(shared, list):
        raise EvaluationError("答案的 comparison.shared 不是数组")
    for item in shared:
        if not isinstance(item, Mapping):
            raise EvaluationError("答案的 comparison.shared 含非对象元素")
        tail = item.get("tail")
        if not isinstance(tail, Mapping) or not isinstance(tail.get("id"), str):
            raise EvaluationError("答案的 comparison.shared 缺少 tail.id")
        concept_ids.append(str(tail["id"]))
        for side in ("left", "right"):
            facts = item.get(side)
            if not isinstance(facts, list):
                raise EvaluationError(f"答案的 comparison.shared 缺少 {side} 数组")
            for fact in facts:
                if not isinstance(fact, Mapping):
                    raise EvaluationError("答案的 comparison.shared 含非对象事实")
                concept_ids.extend(_fact_concept_ids(fact))

    for partition in ("only_left", "only_right", "discriminating"):
        facts = comparison.get(partition)
        if not isinstance(facts, list):
            raise EvaluationError(f"答案的 comparison.{partition} 不是数组")
        for fact in facts:
            if not isinstance(fact, Mapping):
                raise EvaluationError(
                    f"答案的 comparison.{partition} 含非对象元素"
                )
            concept_ids.extend(_fact_concept_ids(fact))
    return tuple(concept_ids)


def _fact_concept_ids(fact: Mapping[str, Any]) -> tuple[str, ...]:
    """一条事实提到的全部概念：两端 + （#15 起新增的）路径中间节点。"""
    concept_ids: list[str] = []
    for side in ("head", "tail"):
        concept = fact.get(side)
        if not isinstance(concept, Mapping) or not isinstance(concept.get("id"), str):
            raise EvaluationError(f"答案的事实缺少 {side}.id")
        concept_ids.append(str(concept["id"]))
    steps = fact.get("path")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping):
                raise EvaluationError("答案的事实 path 含非对象步骤")
            for side in ("head", "tail"):
                concept = step.get(side)
                if isinstance(concept, Mapping) and isinstance(concept.get("id"), str):
                    concept_ids.append(str(concept["id"]))
    return tuple(concept_ids)


def _nested_concept_id(match: Mapping[str, Any]) -> str | None:
    concept = match.get("concept")
    if isinstance(concept, Mapping) and isinstance(concept.get("id"), str):
        return str(concept["id"])
    return None


def _metrics(cases: Sequence[CaseResult]) -> Metrics:
    answerable = [case for case in cases if not case.expect_refusal]
    hits = sum(1 for case in answerable if case.hit)
    citations_total = sum(case.citations_total for case in cases)
    citations_verified = sum(case.citations_verified for case in cases)
    refusal_correct = sum(1 for case in cases if case.refusal_correct)

    path_counts: dict[str, int] = {}
    for case in cases:
        path_counts[case.path] = path_counts.get(case.path, 0) + 1

    by_class: dict[str, ClassMetrics] = {}
    for question_class in QUESTION_CLASSES:
        group = [case for case in cases if case.question_class == question_class]
        group_answerable = [case for case in group if not case.expect_refusal]
        group_hits = sum(1 for case in group_answerable if case.hit)
        by_class[question_class] = ClassMetrics(
            total=len(group),
            answerable=len(group_answerable),
            retrieval_hits=group_hits,
            retrieval_hit_rate=_ratio(group_hits, len(group_answerable)),
        )

    return Metrics(
        question_count=len(cases),
        answerable_count=len(answerable),
        retrieval_hits=hits,
        retrieval_hit_rate=_ratio(hits, len(answerable)),
        citations_total=citations_total,
        citations_verified=citations_verified,
        citation_accuracy=_ratio(citations_verified, citations_total),
        refusal_correct=refusal_correct,
        refusal_correctness=_ratio(refusal_correct, len(cases)),
        path_counts=path_counts,
        by_class=by_class,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """报告的 JSON 形态；字段名是留档与后续对比的契约。"""
    metrics = report.metrics
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "generated_at": report.generated_at,
        "metric_definitions": METRIC_DEFINITIONS,
        "metrics": {
            "question_count": metrics.question_count,
            "answerable_count": metrics.answerable_count,
            "retrieval_hits": metrics.retrieval_hits,
            "retrieval_hit_rate": _rounded(metrics.retrieval_hit_rate),
            "citations_total": metrics.citations_total,
            "citations_verified": metrics.citations_verified,
            "citation_accuracy": _rounded(metrics.citation_accuracy),
            "refusal_correct": metrics.refusal_correct,
            "refusal_correctness": _rounded(metrics.refusal_correctness),
            "path_counts": metrics.path_counts,
            "by_class": {
                name: {
                    "total": group.total,
                    "answerable": group.answerable,
                    "retrieval_hits": group.retrieval_hits,
                    "retrieval_hit_rate": _rounded(group.retrieval_hit_rate),
                }
                for name, group in metrics.by_class.items()
            },
        },
        "cases": [_case_to_dict(case) for case in report.cases],
    }


def _case_to_dict(case: CaseResult) -> dict[str, Any]:
    return {
        "id": case.id,
        "class": case.question_class,
        "question": case.question,
        "path": case.path,
        "detail": case.detail,
        "expect_refusal": case.expect_refusal,
        "refusal_correct": case.refusal_correct,
        "hit": case.hit,
        "concept_ok": case.concept_ok,
        "expected_concepts": list(case.expected_concepts),
        "missing_concepts": list(case.missing_concepts),
        "evidence_ok": case.evidence_ok,
        "expected_evidence_ids": list(case.expected_evidence_ids),
        "matched_evidence_ids": list(case.matched_evidence_ids),
        "document_ok": case.document_ok,
        "expected_documents": list(case.expected_documents),
        "missing_documents": list(case.missing_documents),
        "citations": {
            "total": case.citations_total,
            "verified": case.citations_verified,
            "failures": [
                {"evidence_id": evidence_id, "reason": reason}
                for evidence_id, reason in case.citation_failures
            ],
        },
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)

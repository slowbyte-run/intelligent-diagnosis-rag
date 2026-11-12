"""抽取方差度量：同一输入重复抽取，看产出有多不稳定。

存在的理由：抽取端点不接受确定性保证——``temperature=0`` 下连调三次得三份
不同输出，``seed`` 参数被接受却不生效。金标准分数是单次抽样的结果，把它当
能力估计会得出错误结论（见 ``docs/extraction-variance.md``）。

这个模块只做一件事：跑 N 次同一提示词，报告计数与断言集合的重合度。它不改
抽取行为，也不写图。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.llm import LLMClient
from idr_core.models import EvidenceSegment
from idr_rotating.aliases import AliasResolver
from idr_rotating.extraction import (
    Extraction,
    ValidatedExtraction,
    parse_extraction,
    validate_extraction,
)
from idr_rotating.ontology import Ontology
from idr_rotating.prompts import build_extraction_prompt


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """一次抽取的产出摘要。"""

    run: int
    entities: int
    relations: int
    assertion_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class VarianceReport:
    """N 次抽取的分布，而不是单点。"""

    runs: tuple[RunSnapshot, ...]
    entity_counts: tuple[int, ...]
    relation_counts: tuple[int, ...]
    #: 每对运行之间断言集合的 Jaccard 系数，反映「两次抽到的知识有多像」。
    pairwise_overlap: tuple[float, ...]

    @property
    def identical(self) -> bool:
        """所有运行的断言集合是否完全一致。"""
        return len({snapshot.assertion_keys for snapshot in self.runs}) == 1

    @property
    def mean_overlap(self) -> float | None:
        """两两重合度均值；少于两次运行时无意义。"""
        if not self.pairwise_overlap:
            return None
        return sum(self.pairwise_overlap) / len(self.pairwise_overlap)


def measure_extraction_variance(
    llm: LLMClient,
    ontology: Ontology,
    segments: tuple[EvidenceSegment, ...],
    *,
    runs: int = 3,
    aliases: AliasResolver | None = None,
) -> VarianceReport:
    """对同一批证据段重复抽取 ``runs`` 次，返回分布。"""
    if runs < 2:  # noqa: PLR2004 - 两次以下没有「方差」可言
        raise ValueError("runs 至少为 2，否则无法比较")

    prompt = build_extraction_prompt(ontology, segments)
    evidence_text = {segment.id: segment.text for segment in segments}
    doc_id = segments[0].doc_id if segments else ""

    snapshots: list[RunSnapshot] = []
    for index in range(runs):
        extraction = validate_extraction(
            parse_extraction(llm.complete(prompt)),
            ontology,
            doc_id=doc_id,
            evidence_text=evidence_text,
            aliases=aliases,
        )
        snapshots.append(
            RunSnapshot(
                run=index + 1,
                entities=len(extraction.entities),
                relations=len(extraction.relations),
                assertion_keys=_assertion_keys(extraction),
            )
        )

    return VarianceReport(
        runs=tuple(snapshots),
        entity_counts=tuple(snapshot.entities for snapshot in snapshots),
        relation_counts=tuple(snapshot.relations for snapshot in snapshots),
        pairwise_overlap=_pairwise_overlap([s.assertion_keys for s in snapshots]),
    )


def _assertion_keys(extraction: ValidatedExtraction) -> frozenset[str]:
    """把一次抽取化成可比较的键集合。"""
    keys: set[str] = set()
    for entity in extraction.entities:
        keys.add(f"entity:{entity.name}")
    for relation in extraction.relations:
        keys.add(f"relation:{relation.head}|{relation.type}|{relation.tail}")
    return frozenset(keys)


def _pairwise_overlap(sets: list[frozenset[str]]) -> tuple[float, ...]:
    overlap: list[float] = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            first, second = sets[left], sets[right]
            union = first | second
            overlap.append(1.0 if not union else len(first & second) / len(union))
    return tuple(overlap)

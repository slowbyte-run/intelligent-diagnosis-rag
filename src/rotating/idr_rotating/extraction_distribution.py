"""抽取评估的重复运行与分布。

单次抽取评估只是一次抽样：同一输入重复抽取，产出会变，召回率与精度也跟着变
（见 ``docs/extraction-variance.md``）。这个模块在 ``extract_annotated`` 外面
重复 N 次，每次把结果交给纯函数 ``score_extractions``，再把结果摊成分布：

- 每次运行的召回率、精度与命中计数；
- 各次运行「命中了哪些标注断言」的集合，以及两两 Jaccard 重合度；
- 召回率与精度各自的极差（最大值 − 最小值）。

评分本身不在这里，也不改。重合度与极差的表达复用 ``variance.py``，避免为
「重复运行 + 分布」发明第二套词汇。
"""

from __future__ import annotations

from dataclasses import dataclass

from idr_core.llm import LLMClient
from idr_rotating.aliases import AliasResolver
from idr_rotating.extraction_evaluation import (
    AnnotationSet,
    ExtractionScore,
    extract_annotated,
    score_extractions,
)
from idr_rotating.ontology import Ontology
from idr_rotating.variance import MetricSpread, pairwise_overlap


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """一次完整评估：逐篇抽取、评分，以及这次命中的标注断言集合。"""

    run: int
    score: ExtractionScore
    #: 这次运行命中的期望断言键，用于跨运行比较「命中了哪些」。
    hit_keys: frozenset[str]

    @property
    def recall(self) -> float | None:
        return self.score.recall

    @property
    def precision(self) -> float | None:
        return self.score.precision


@dataclass(frozen=True, slots=True)
class ExtractionDistribution:
    """N 次抽取评估的分布，而不是单点。"""

    runs: tuple[EvaluationRun, ...]
    #: 每对运行之间「命中的标注断言集合」的 Jaccard 系数。
    pairwise_overlap: tuple[float, ...]
    #: 召回率与精度的极差；单次运行时没有分布可言，因此调用方至少传 2。
    recall: MetricSpread
    precision: MetricSpread

    @property
    def mean_overlap(self) -> float | None:
        if not self.pairwise_overlap:
            return None
        return sum(self.pairwise_overlap) / len(self.pairwise_overlap)

    @property
    def identical_hits(self) -> bool:
        """所有运行命中的标注断言集合是否完全一致。"""
        return len({run.hit_keys for run in self.runs}) == 1

    @property
    def consistent_hits(self) -> frozenset[str]:
        """每次运行都命中的标注断言：不是抽样运气，是稳定抽到。"""
        if not self.runs:
            return frozenset()
        consistent = self.runs[0].hit_keys
        for run in self.runs[1:]:
            consistent &= run.hit_keys
        return consistent

    @property
    def ever_hits(self) -> frozenset[str]:
        """至少一次命中的标注断言：其中不属于 ``consistent_hits`` 的部分是抖动。"""
        return frozenset().union(*(run.hit_keys for run in self.runs))

    @property
    def flaky_hits(self) -> frozenset[str]:
        """只在部分运行里命中的标注断言。"""
        return self.ever_hits - self.consistent_hits


def measure_extraction_distribution(
    llm: LLMClient,
    ontology: Ontology,
    annotations: AnnotationSet,
    *,
    runs: int = 3,
    aliases: AliasResolver | None = None,
) -> ExtractionDistribution:
    """对同一标注集重复评估 ``runs`` 次，返回每次分数与整体分布。"""
    if runs < 2:
        raise ValueError("runs 至少为 2，否则无法比较分布")

    evaluations: list[EvaluationRun] = []
    for index in range(runs):
        score = score_extractions(
            annotations,
            extract_annotated(llm, ontology, annotations, aliases=aliases),
        )
        evaluations.append(
            EvaluationRun(
                run=index + 1,
                score=score,
                hit_keys=_hit_keys(score),
            )
        )

    return ExtractionDistribution(
        runs=tuple(evaluations),
        pairwise_overlap=pairwise_overlap(
            [evaluation.hit_keys for evaluation in evaluations]
        ),
        recall=MetricSpread(
            tuple(evaluation.recall for evaluation in evaluations)
        ),
        precision=MetricSpread(
            tuple(evaluation.precision for evaluation in evaluations)
        ),
    )


def _hit_keys(score: ExtractionScore) -> frozenset[str]:
    """把一次评估化成「命中了哪些标注断言」的键集合。

    键里带证据段标识，因此不同证据段里的同名断言不会互相顶替。
    """
    return frozenset(
        f"{outcome.kind}:{outcome.evidence_id}:{outcome.label}"
        for outcome in score.outcomes
        if outcome.status == "matched"
    )

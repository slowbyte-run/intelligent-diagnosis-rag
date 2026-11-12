"""``idr variance``：同一篇文献重复抽取，报告产出分布。

用途是回答「这个分数能不能当作能力估计」。抽取端点不保证确定性，单次抽样
的数字会骗人；这个命令把分布摊开。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from idr_core.config import Config
from idr_core.llm import LLMClient
from idr_core.parsing import Parser
from idr_core.documents import document_id
from idr_core.parsing.pdf import PdfTextParser
from idr_rotating.document import detect_document_type
from idr_rotating.ontology import load_ontology
from idr_rotating.segments import build_evidence_segments
from idr_rotating.variance import VarianceReport, measure_extraction_variance


def run_variance(
    source: Path,
    *,
    llm: LLMClient,
    runs: int,
    out: Path | None = None,
    parser: Parser | None = None,
) -> int:
    """对单篇文献的整篇证据段重复抽取，打印并可选落盘分布。

    ``parser`` 可注入，测试用假解析器就不必造真 PDF。
    """
    try:
        parsed = (parser or PdfTextParser()).parse(source)
    except Exception as error:  # noqa: BLE001 - 单篇解析失败要给人话
        print(f"解析失败：{source.name}：{error}", file=sys.stderr)
        return 1

    segments = build_evidence_segments(
        document_id(source.read_bytes()), parsed, detect_document_type(parsed)
    )
    if not segments:
        print(f"该文献没有产出证据段：{source.name}", file=sys.stderr)
        return 1

    report = measure_extraction_variance(
        llm,
        load_ontology(),
        tuple(segments),
        runs=runs,
    )
    _print_report(source, report, runs)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_to_dict(source, report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"报告：{out}")
    return 0


def _print_report(source: Path, report: VarianceReport, runs: int) -> None:
    print(f"文献：{source.name}")
    print(f"重复次数：{runs}")
    print()
    print("每次产出：")
    for snapshot in report.runs:
        print(
            f"  #{snapshot.run}  实体 {snapshot.entities:>3}  "
            f"关系 {snapshot.relations:>3}  断言 {len(snapshot.assertion_keys):>3}"
        )
    print()
    print(f"实体数：{list(report.entity_counts)}")
    print(f"关系数：{list(report.relation_counts)}")
    overlap = report.mean_overlap
    print(f"断言集合两两重合（Jaccard 均值）：{'—' if overlap is None else f'{overlap:.3f}'}")
    print(f"所有运行完全一致：{'是' if report.identical else '否'}")


def _to_dict(source: Path, report: VarianceReport) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": source.name,
        "runs": [
            {
                "run": snapshot.run,
                "entities": snapshot.entities,
                "relations": snapshot.relations,
                "assertion_keys": sorted(snapshot.assertion_keys),
            }
            for snapshot in report.runs
        ],
        "entity_counts": list(report.entity_counts),
        "relation_counts": list(report.relation_counts),
        "pairwise_overlap": list(report.pairwise_overlap),
        "mean_overlap": report.mean_overlap,
        "identical": report.identical,
    }


def build_variance_llm(config: Config) -> LLMClient:
    """真实模型客户端；与入库用的是同一套参数。"""
    from idr_core.llm.openai_compat import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
        model=config.deepseek_model,
        extra_headers=config.llm_extra_headers,
    )

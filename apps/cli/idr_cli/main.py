"""命令行入口。"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from idr_core.config import Config, ConfigError, load_config
from idr_cli.ask import DEFAULT_SOURCE_DIR, build_ask_deps, run_ask
from idr_cli.calibrate import run_calibrate
from idr_cli.doctor import run_doctor
from idr_cli.eval import run_eval
from idr_cli.ingest import build_deps, run_ingest
from idr_cli.recall import DEFAULT_VARIANTS, run_recall
from idr_rotating import AskDeps, IngestDeps


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idr",
        description="知识图谱 RAG 工具：把文献转成可溯源的知识图谱，并回答带引用的问题。",
    )
    commands = parser.add_subparsers(dest="command", metavar="<命令>")
    commands.add_parser("doctor", help="检查配置与本地服务连通性")
    ingest_command = commands.add_parser(
        "ingest", help="把目录中的 PDF 入库，输出证据段、实体与关系计数"
    )
    ingest_command.add_argument("source_dir", help="存放 PDF 的目录")
    ingest_command.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="批次报告输出目录，默认 data/output/reports",
    )
    ingest_command.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="只处理按文件名排序的前 N 篇，用于扩大语料前快速验证",
    )
    ask_command = commands.add_parser(
        "ask", help="用一句自然语言提问，输出带文献名、页码与原文的答案"
    )
    ask_command.add_argument("question", help="问题文本")
    ask_command.add_argument(
        "--json",
        action="store_true",
        help="输出结构化 JSON，供评估与下游服务使用",
    )
    ask_command.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"语料目录，用于把文档标识对回文献名，默认 {DEFAULT_SOURCE_DIR}",
    )
    eval_command = commands.add_parser(
        "eval", help="跑金标准集，输出检索命中率、引用准确率与拒答正确率"
    )
    eval_command.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="金标准集 JSON，默认使用领域包内的 golden_questions.json",
    )
    eval_command.add_argument(
        "--out",
        type=Path,
        default=None,
        help="评估报告输出文件，默认 data/output/reports/eval-<时间戳>.json",
    )
    eval_command.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"源 PDF 目录，用于引用回源校验，默认 {DEFAULT_SOURCE_DIR}",
    )
    calibrate_command = commands.add_parser(
        "calibrate",
        help="用金标准集校准向量兜底阈值，输出推荐值并可选写回配置",
    )
    calibrate_command.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="金标准集 JSON，默认使用领域包内的 golden_questions.json",
    )
    calibrate_command.add_argument(
        "--config",
        type=Path,
        default=None,
        help="检索配置路径，默认使用领域包内的 config/retrieval.yaml",
    )
    calibrate_command.add_argument(
        "--out",
        type=Path,
        default=None,
        help="校准报告输出文件，默认 data/output/reports/calibration-<时间戳>.json",
    )
    calibrate_command.add_argument(
        "--write",
        action="store_true",
        help="把推荐阈值与校准标记写回检索配置（保留注释）",
    )
    recall_command = commands.add_parser(
        "recall",
        help="只量向量召回（嵌入 + pgvector，不重排），报告 recall@k 与逐题名次",
    )
    recall_command.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="金标准集 JSON，默认使用领域包内的 golden_questions.json",
    )
    recall_command.add_argument(
        "--out",
        type=Path,
        default=None,
        help="召回报告输出文件，默认 data/output/reports/recall-<时间戳>.json",
    )
    recall_command.add_argument(
        "--top-k",
        type=_positive_int,
        default=100,
        metavar="N",
        help="向量检索取多少条候选用于诊断，默认 100（要覆盖 recall@100）",
    )
    recall_command.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        metavar="LIST",
        help=(
            "逗号分隔的查询变体：raw、enriched、configured、"
            "instruct:<预设>、enriched+instruct:<预设>；"
            f"默认 {DEFAULT_VARIANTS}"
        ),
    )
    recall_command.add_argument(
        "--lexical",
        action="store_true",
        help="把词面候选按 RRF 与向量候选融合后再统计（与线上 lexical_recall 同口径）",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    deps_factory: Callable[[Config], IngestDeps] = build_deps,
    ask_deps_factory: Callable[[Config, Path], AskDeps] = build_ask_deps,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        parser.print_help()
        return 0

    if arguments.command == "doctor":
        return run_doctor()

    if arguments.command == "ingest":
        try:
            config = load_config()
            deps = deps_factory(config)
        except ConfigError as error:
            print(f"配置错误：{error}", file=sys.stderr)
            return 1
        return run_ingest(
            Path(arguments.source_dir),
            deps=deps,
            limit=arguments.limit,
            reports_dir=arguments.reports_dir,
        )

    if arguments.command == "ask":
        try:
            config = load_config()
            ask_deps = ask_deps_factory(config, Path(arguments.corpus_dir))
        except ConfigError as error:
            print(f"配置错误：{error}", file=sys.stderr)
            return 1
        return run_ask(
            arguments.question,
            deps=ask_deps,
            as_json=arguments.json,
        )

    if arguments.command == "eval":
        try:
            config = load_config()
            eval_deps = ask_deps_factory(config, Path(arguments.corpus_dir))
        except ConfigError as error:
            print(f"配置错误：{error}", file=sys.stderr)
            return 1
        return run_eval(
            deps=eval_deps,
            corpus_dir=Path(arguments.corpus_dir),
            golden_path=arguments.golden,
            out_path=arguments.out,
        )

    if arguments.command == "calibrate":
        try:
            config = load_config()
            calibrate_deps = ask_deps_factory(config, DEFAULT_SOURCE_DIR)
        except ConfigError as error:
            print(f"配置错误：{error}", file=sys.stderr)
            return 1
        return run_calibrate(
            deps=calibrate_deps,
            golden_path=arguments.golden,
            out_path=arguments.out,
            settings_path=arguments.config,
            write_config=arguments.write,
        )

    if arguments.command == "recall":
        try:
            config = load_config()
            recall_deps = ask_deps_factory(config, DEFAULT_SOURCE_DIR)
        except ConfigError as error:
            print(f"配置错误：{error}", file=sys.stderr)
            return 1
        return run_recall(
            deps=recall_deps,
            golden_path=arguments.golden,
            out_path=arguments.out,
            top_k=arguments.top_k,
            variants=arguments.variants,
            lexical=arguments.lexical,
        )

    raise AssertionError(f"未实现的命令：{arguments.command}")


if __name__ == "__main__":
    sys.exit(main())

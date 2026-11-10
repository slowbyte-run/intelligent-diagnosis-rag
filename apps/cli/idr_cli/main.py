"""命令行入口。"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from idr_core.config import Config, ConfigError, load_config
from idr_cli.doctor import run_doctor
from idr_cli.ingest import build_deps, run_ingest
from idr_rotating import IngestDeps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idr",
        description="知识图谱 RAG 工具：把文献转成可溯源的知识图谱。",
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
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    deps_factory: Callable[[Config], IngestDeps] = build_deps,
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
            reports_dir=arguments.reports_dir,
        )

    raise AssertionError(f"未实现的命令：{arguments.command}")


if __name__ == "__main__":
    sys.exit(main())

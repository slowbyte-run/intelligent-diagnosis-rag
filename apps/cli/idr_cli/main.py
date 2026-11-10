"""命令行入口。"""

import argparse
import sys
from collections.abc import Sequence

from idr_cli.doctor import run_doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idr",
        description="知识图谱 RAG 工具：把文献转成可溯源的知识图谱。",
    )
    commands = parser.add_subparsers(dest="command", metavar="<命令>")
    commands.add_parser("doctor", help="检查配置与本地服务连通性")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        parser.print_help()
        return 0

    if arguments.command == "doctor":
        return run_doctor()

    raise AssertionError(f"未实现的命令：{arguments.command}")


if __name__ == "__main__":
    sys.exit(main())

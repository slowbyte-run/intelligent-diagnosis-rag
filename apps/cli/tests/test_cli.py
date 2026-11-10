"""命令行边界测试：只断言进程的退出码与输出流。"""

import os
import subprocess
import sys


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "idr_cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def bare_env(**overrides: str) -> dict[str, str]:
    """只保留 PATH 的最小环境，用于模拟「什么都没配」的机器。"""
    return {"PATH": os.environ.get("PATH", ""), **overrides}


def test_help_lists_available_commands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "doctor" in result.stdout


def test_doctor_names_every_missing_environment_variable() -> None:
    result = run_cli("doctor", env=bare_env())

    assert result.returncode == 1
    assert "DEEPSEEK_API_KEY" in result.stderr
    assert "NEO4J_PASSWORD" in result.stderr
    assert "Traceback" not in result.stderr


def configured_env(**overrides: str) -> dict[str, str]:
    """一套完整但指向不可达端口的配置，用于验证失败路径而不依赖真实服务。"""
    return bare_env(
        **{
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "sk-test",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "SILICONFLOW_BASE_URL": "https://api.siliconflow.cn/v1",
            "SILICONFLOW_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-4B",
            "EMBEDDING_DIM": "1024",
            "NEO4J_URI": "bolt://127.0.0.1:1",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
            "DATABASE_URL": "postgresql://idr:secret@127.0.0.1:1/idr",
            **overrides,
        }
    )


def test_doctor_reports_each_unreachable_service() -> None:
    result = run_cli("doctor", env=configured_env())

    assert result.returncode == 1
    assert "Neo4j" in result.stdout
    assert "Postgres" in result.stdout
    assert "Traceback" not in result.stderr

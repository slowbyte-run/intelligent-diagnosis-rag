"""集成契约测试：需要 docker compose 起的真实服务。

默认不跑（pytest 的 addopts 已排除 integration 标记）。
运行前把 .env 导入当前 shell：

    set -a; source .env; set +a
    uv run pytest -m integration
"""

import os

import pytest

from test_cli import configured_env, run_cli

NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (NEO4J_PASSWORD and POSTGRES_PASSWORD),
        reason="需要 NEO4J_PASSWORD 与 POSTGRES_PASSWORD（先 source .env）",
    ),
]


def test_doctor_passes_against_local_services() -> None:
    result = run_cli(
        "doctor",
        env=configured_env(
            NEO4J_URI="bolt://127.0.0.1:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD=NEO4J_PASSWORD,
            DATABASE_URL=(
                f"postgresql://idr:{POSTGRES_PASSWORD}@127.0.0.1:5432/idr"
            ),
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "配置：通过" in result.stdout
    assert "Neo4j：通过" in result.stdout
    assert "Postgres：通过（" in result.stdout
    assert "vector " in result.stdout

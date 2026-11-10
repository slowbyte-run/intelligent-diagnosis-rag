"""集成契约测试：需要 docker compose 起的真实服务。

默认不跑（pytest 的 addopts 已排除 integration 标记）。
运行方式：`uv run pytest -m integration`
"""

import pytest

from test_cli import configured_env, run_cli

pytestmark = pytest.mark.integration


def test_doctor_passes_against_local_services() -> None:
    result = run_cli(
        "doctor",
        env=configured_env(
            NEO4J_URI="bolt://127.0.0.1:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="idrdevpassword",
            DATABASE_URL="postgresql://idr:idrdevpassword@127.0.0.1:5432/idr",
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "配置：通过" in result.stdout
    assert "Neo4j：通过" in result.stdout
    assert "Postgres：通过" in result.stdout

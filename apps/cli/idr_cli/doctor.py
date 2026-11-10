"""doctor 命令：检查配置与本地服务连通性。"""

import sys

from idr_core.config import ConfigError, load_config
from idr_core.health import check_neo4j, check_postgres


def run_doctor() -> int:
    try:
        config = load_config()
    except ConfigError as error:
        print(f"配置错误：{error}", file=sys.stderr)
        return 1

    print("配置：通过")

    results = (
        check_neo4j(config.neo4j_uri, config.neo4j_user, config.neo4j_password),
        check_postgres(config.database_url),
    )

    for result in results:
        status = "通过" if result.ok else "无法连接"
        print(f"{result.service}：{status}（{result.detail}）")

    return 0 if all(result.ok for result in results) else 1

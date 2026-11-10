"""本地服务连通性检查。检查失败返回结果，不让异常冒泡成堆栈。"""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

DEFAULT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class CheckResult:
    service: str
    ok: bool
    detail: str


def redact_url(url: str) -> str:
    """去掉 URL 中的口令，避免检查结果把密钥打到终端。"""
    parts = urlsplit(url)
    if not parts.password:
        return url
    credentials = f"{parts.username}:***" if parts.username else "***"
    host = parts.hostname or ""
    netloc = f"{credentials}@{host}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def check_neo4j(
    uri: str,
    user: str,
    password: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CheckResult:
    from neo4j import GraphDatabase

    try:
        with GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=timeout_seconds,
        ) as driver:
            driver.verify_connectivity()
    except Exception as error:  # noqa: BLE001 - 目标是把任何失败转成可读消息
        return CheckResult("Neo4j", False, f"{redact_url(uri)}：{error}")
    return CheckResult("Neo4j", True, redact_url(uri))


def check_postgres(
    database_url: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CheckResult:
    import psycopg

    try:
        with psycopg.connect(
            database_url,
            connect_timeout=int(timeout_seconds),
        ) as connection:
            connection.execute("SELECT 1")
    except Exception as error:  # noqa: BLE001
        return CheckResult("Postgres", False, f"{redact_url(database_url)}：{error}")
    return CheckResult("Postgres", True, redact_url(database_url))

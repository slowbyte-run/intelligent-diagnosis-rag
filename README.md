# intelligent-diagnosis-rag

旋转设备故障诊断与肺部疾病诊断两个领域，共用一条「文献 → 可溯源知识图谱 → 带引用回答」的流水线。一期做设备领域。

- 规划与阶段判据：[docs/roadmap.md](docs/roadmap.md)
- 架构决策：[docs/adr/](docs/adr/)
- 术语表：[GLOSSARY-MAP.md](GLOSSARY-MAP.md)
- 工程约定：[AGENTS.md](AGENTS.md)

## 快速开始

需要 `uv` 与 docker（含 compose 插件）。

```bash
cp .env.example .env      # 填写 NEO4J_PASSWORD 与 POSTGRES_PASSWORD
uv sync                   # 装依赖
docker compose up -d      # 起 Neo4j 与 Postgres/pgvector
uv run --env-file .env idr doctor   # 检查配置与两个服务的连通性
```

`idr doctor` 会打印配置、Neo4j、Postgres 三项状态；任一项失败则退出码为 1，且不会打印堆栈。

## 入库报告

`idr ingest <目录>` 每次运行都会在 `data/output/reports/` 写一份 JSON 报告，文件名形如
`ingest-20261111T063012Z-3f9a2b1c.json`：时间戳取运行开始时间，随机后缀保证同一秒内的
两次运行也不会互相覆盖；`--reports-dir` 可以改输出目录。空目录也会正常写出报告，各项计数为 0。

报告包含：

- 成功、失败、隔离、跳过四类文献的计数与逐篇明细（含失败原因、隔离原因与跳过原因），以及待审计抽取的计数与明细（类型、名称、原因、出现频次、证据段标识）；
- 每篇文献的证据段数、实体数与关系数；
- 每次模型调用的模型名、输入输出 token 数与耗时；
- 本次运行的模型与参数组合（模型名、地址、temperature、response_format）以及本体版本与内容指纹，用于两次运行之间做可比性核对。

## 测试

```bash
uv run pytest                 # 单元测试，不需要任何服务
set -a; source .env; set +a
uv run pytest -m integration  # 契约测试，需要先 docker compose up -d
```

## 目录

| 路径 | 内容 |
|---|---|
| `src/core` | 共享核心包，与领域无关 |
| `src/rotating` | 设备领域包：本体、抽取提示词、检索编排 |
| `src/pulmonary` | 肺部疾病领域包（二期） |
| `apps/cli` | 命令行入口 |
| `apps/api`、`apps/web` | 服务化（后续里程碑） |
| `docker/init` | 数据库首次初始化脚本 |
| `data/input` | 原始语料（不入库） |

领域包之间禁止互相 import，共享核心包内不得出现领域词汇——后者由 `pytest` 守护。

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

## 问答

```bash
uv run --env-file .env idr ask "轴承外圈疲劳剥落有什么征兆"
uv run --env-file .env idr ask "轴承外圈疲劳剥落有什么征兆" --json   # 供评估使用的结构化结果
```

输出按「问题 → 检索路径 → 命中概念 → 结论 → 每条结论的出处」排列：

```
问题：轴承外圈疲劳剥落有什么征兆
检索路径：图命中
命中概念：轴承外圈疲劳剥落（故障模式，别名「轴承外圈疲劳剥落」子串命中）

结论：
1. 轴承外圈疲劳剥落 —表现为→ 外圈故障频率及其倍频（置信度 0.90）
   依据：arxiv-2206_14153.pdf 第 20 页：Outer race defect = ...
```

- 检索路径三选一并显式标注：`graph`（图命中）、`vector`（向量兜底，后续里程碑接进来）、`refusal`（没有可用依据时明确拒答，`detail` 说明缺在哪一步）。
- 实体链接只用本体别名表，做整句与子串匹配，答案里指出命中了哪条别名。
- 每条结论都回填证据段的文献名、页码与原文；事实边没有 `evidence_ids` 时不进答案。
- 图谱里只存 `doc_id`。文献名按 `--corpus-dir`（默认 `data/input`）中的文件内容哈希对回；从别的目录运行或语料不在默认位置时显式指定。

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

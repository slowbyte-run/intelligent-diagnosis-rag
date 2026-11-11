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

- 检索路径三选一并显式标注：`graph`（图命中）、`vector`（向量兜底：图没命中时按重排分数取证据段，分数低于阈值即不答题）、`refusal`（没有可用依据时明确拒答，`detail` 说明缺在哪一步）。
- 拒答分两种原因：`no_concept`（问题里的说法没有链接到任何本体概念）与 `no_evidence`（概念识别了，但图谱没有带证据的事实、向量兜底也没有候选过阈值）；人类可读输出与 `--json` 的 `refusal_reason` 都给出原因。拒答答案的结论与引用恒为空。
- 图检索最多走两跳：从最具体的命中出发，只走 `has_component`、`has_fault_mode`、`has_cause`、`indicated_by`、`detected_by`、`confused_with` 六类白名单边，三跳及以上不执行。两跳只允许经 `has_component` / `has_fault_mode` 中转，避免把「易混淆故障的征兆」算到当前故障上。
- 路径上的每一步都要有证据；任一步缺证据，该条结论整条丢弃，不缩短成部分路径。多跳结论的人类可读输出展开完整路径（如「滚动轴承 —具有故障模式→ 外圈剥落 —表现为→ 外圈故障频率及其倍频」），`--json` 里对应 `facts[].path`，JSON schema 版本为 3。
- 结果集有上界（当前 20 条）；超出时按关系类型与概念名称排序截断，`--json` 里以 `truncated` 与 `detail` 说明。
- 实体链接只用本体别名表，做整句与子串匹配，答案里指出命中了哪条别名。
- 每条结论都回填证据段的文献名、页码与原文；事实边没有 `evidence_ids` 时不进答案。
- 图谱里只存 `doc_id`。文献名按 `--corpus-dir`（默认 `data/input`）中的文件内容哈希对回；从别的目录运行或语料不在默认位置时显式指定。

## 评估

```bash
uv run --env-file .env idr eval
uv run --env-file .env idr eval --out data/output/reports/m2-baseline.json
```

金标准集在 `src/rotating/idr_rotating/golden_questions.json`：20 题，事实型 12（含 2 道
拒答对照）、对比型 4、跨文献型 4；每题绑定规范概念节点 ID 与证据段 ID，跨文献题还要求
期望证据来自至少两篇文献。题目由实现方起草、用户终审；改题会改变基线，改动请连同报告
一起提交评审。

三项指标（完整定义随报告一起写进 `metric_definitions`）：

- **检索命中率**：可答题中，期望概念全部出现在答案的概念集合、且至少命中一个期望证据段
  的比例；跨文献题还要求每个期望文献都至少出现一条引用。拒答对照题不进分母。
- **引用准确率**：答案实际输出的全部引用中，能在源 PDF 所标页逐字找到的比例。校验时按
  `doc_id`（文件内容哈希）对回 `--corpus-dir` 里的 PDF，重新抽取该页文字再逐字查找，
  不使用裁判模型；没有输出引用时记 `null`。
- **拒答正确率**：全部题中，答案路径是否为拒答与题目期望一致的比例；「该答却拒」与
  「该拒却答」都算错。

报告默认写到 `data/output/reports/eval-<时间戳>.json`（文件名带随机后缀），`--out`
可以指定留档路径；报告含三项指标、按题型拆分、逐题的概念/证据/文献覆盖与引用校验失败
原因。评估过程只读图谱与语料：图谱只走 `GraphReader` 读方法，PDF 只打开读取，唯一的
写出是报告文件。

## 阈值校准

`vector_min_rerank_score` 是向量兜底「有依据 / 拒答」的判据，来自
`src/rotating/config/retrieval.yaml`。它是猜出来的初值，必须用金标准集校准后再写回
（#17）：

```bash
uv run --env-file .env idr calibrate          # 采样 + 扫描，输出推荐值与完整报告
uv run --env-file .env idr calibrate --write  # 同时把阈值与校准标记写回配置
```

采样每题 2 次模型调用（嵌入 + 重排），阈值扫描在同一份分数上离线完成。校准口径、本次
分数分布、当前值为什么是 0.78、以及「单一阈值分不开跨语言证据与拒答对照」的取舍，见
[docs/retrieval-calibration.md](docs/retrieval-calibration.md)。

当前基线（20 题，阈值 0.78）：检索命中率 14/18、引用准确率 298/298、拒答正确率 18/20。

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

# 路线图

先纵向打通设备领域的知识问答链路并拿到评估数字，再横向加厚，最后才启动肺病领域。**肺病领域在阶段 2 通过之前不启动**——在那之前不为它写任何抽象。

## 阶段 0 — 骨架与决策（已完成）

单仓库 uv workspace、三份术语表、ADR-0001（单仓与共享边界）、0002（模型供给）、0003（Neo4j + pgvector）、0004（概念+证据双层）。

## 阶段 1 — 设备领域纵向切片（20 篇文献）

整条链路每个环节只做最薄一版：解析 → 分块 → 抽取 → 概念+证据双层写图 → 向量召回 + 图扩散 → 带引用回答。**只做命令行，不做 UI。**

- 语料：20 篇，中英混合，其中至少 5 篇中文专利（专利的说明书/权利要求结构与论文不同，需要单独的分块规则）
- 完成判据：对 10 个真实问题给出带「篇名 + 段落」引用的答案；答案中每条结论都能在被引用的段落原文中找到

## 阶段 2 — 评估基线（50 条金标准）

- 三项指标：检索命中率（Recall@k / MRR）、引用准确率、拒答正确率
- 金标准的权威来源：公开数据集（CWRU 轴承、PHM 挑战赛、MFPT）的官方描述，用于校准「故障模式 ↔ 特征频率 ↔ 征兆」的映射；文献原文提供出处
- 完成判据：三项指标有可重复的数字；拒答正确率作为硬门槛，不达标不进阶段 3

## 阶段 3 — 本体充实与扩量（100–300 篇）

- ISO 10816 / ISO 13373 锚定顶层本体；清单外的新类型进待审队列，人工审核后入库
- 完成判据：抽取结果 100% 通过本体校验才入库；扩量后指标不下降

## 阶段 4 — 服务化

FastAPI 后端 + 前端 + docker compose，复用 cook-rag 的骨架。私有化降级路径在此阶段必须实测通过（拔掉外网仍能回答）。

- 完成判据：一条 `docker compose up` 起全栈；断网模式下仍可完成一次带引用的问答

## 阶段 5 — 肺病领域启动

- 用 ICD-10 / SNOMED 锚定本体，复刻阶段 1–2 的流程
- 只有在此阶段，才把被两次验证过的逻辑从领域包下沉到 `src/core`
- 完成判据：同类型金标准集通过；`src/core` 中检索不到任何设备领域词汇

## 可复用资产（来自你已存在的仓库）

| 资产 | 路径 | 用途 |
| --- | --- | --- |
| LLM / embedding 适配层 | `rag-anything/cook-rag/src/cook_rag/providers/` | `core/llm` 的起点，含 DeepSeek 与 SiliconFlow 实现 |
| 抽取 prompt + Pydantic schema | `rag/GraphRAG/graphrag_lab/graph.py` | `core/extract` 的起点（含实体/关系类型清单） |
| 评估框架 | `rag-anything/cook-rag/src/cook_rag/eval/`、`eval/golden_set.json`、`eval/thresholds.json` | `core/eval` 的起点，已含 judge 与阈值 |
| Neo4j 装载与图检索 | `all-in-rag/code/C9/rag_modules/graph_data_preparation.py`、`graph_rag_retrieval.py` | `core/graph` 与图遍历检索 |
| Docker / uv 骨架 | `rag-anything/cook-rag/{Dockerfile.backend,docker-compose.yml}`、`docker/backend-entrypoint.sh` | 阶段 4 直接改造 |
| 引用卡片前端 | `rag-anything/cook-rag/frontend/src/components/CitationCard.tsx` | 阶段 4 的引用溯源展示 |
| 路径解析工具 | `all-in-rag/code/project_paths.py` | 入口脚本的路径约定 |

## 既有仓库的约定（沿用）

- 中文写文档、注释、聊天
- 密钥只走环境变量，`.env` 不入库；数据目录不入库
- 每个仓库用 `uv` 管理依赖并提交 `uv.lock`
- 统一在 `code/.venv/bin/python` 这类显式解释器上验证脚本，不依赖隐式激活

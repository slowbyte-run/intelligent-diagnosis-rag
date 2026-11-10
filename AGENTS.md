# intelligent-diagnosis-rag

## 优先级

用户当轮指令 > 本文件 > [`docs/standards/agent-coding-standard.md`](docs/standards/agent-coding-standard.md)。

通用工程规范（总则、何时停下问人、安全、Git、Docker、依赖管理、CI）全部在那份标准里，**本文件不重复**。它逐字复制自 `slowbyte-run/agent-coding-standards`，改动请回上游改。

本机环境事实（代理、GPU、端口占用、资源上限）在 `.agent/local.md`，**不入库**，换机器不适用。

## 项目

两个知识图谱 RAG 领域，共享一条流水线：**旋转设备故障诊断**（一期，`src/rotating`）与**肺部疾病诊断**（二期，`src/pulmonary`）。两者都是「从文献构建知识库 + 带引用回答」的系统，**都不处理信号或影像**。

- 规划与阶段判据：`docs/roadmap.md`
- 架构决策：`docs/adr/`
- 术语：`GLOSSARY-MAP.md`

## 架构约束（硬性）

1. **领域包之间禁止互相 import。** `src/rotating` 与 `src/pulmonary` 无任何依赖关系，只能各自依赖 `src/core`。
2. **`src/core` 禁止出现领域词汇。** 不得出现「轴承」「齿轮」「转子」「肺部」「疾病」「症状」等具体领域词；core 里只允许抽象的流水线概念。
3. **领域特有资产只放领域包内**：本体、抽取提示词、检索编排、金标准集。放进 core 之前先问：它真的与领域无关吗？
4. **模型调用只走 `src/core/llm` 适配层。** 业务代码不直接 import 任何 provider SDK；换供应商只改适配层。
5. **抽取结果必须通过本体校验才入库。** 校验失败是数据错误，不是警告，不得静默丢弃。
6. **答案必须带引用。** 没有证据段支撑的结论不允许出现在回答里；证据不足时走拒答，不给低置信度答案。
7. **不引入编排框架与 ORM。** 禁止 LangChain / LangGraph / LlamaIndex / 任何 KG-RAG 框架作为运行时依赖，也不用 ORM。新增运行时依赖需先说明归属层（`src/core` 还是领域包），并更新 ADR-0005 的依赖白名单。

改动图谱 schema、本体类型清单、评估指标定义之前先停下问人——按标准 §2，这些是本项目的数据模型。

## 本仓库与通用标准不同的地方

- **本仓库要写测试。** `all-in-rag` 里「不写测试／lint／CI，不要自己发明」是教程仓库的家规，**不适用于这里**：`src/core` 被两个领域共用，一旦错了同时污染两边；评估框架本身也是代码。
- **镜像不用 `latest`。** `cook-rag` 现在用 `localhost/postgres-age:latest`，本项目不跟：Neo4j、pgvector 从第一天就钉明确版本，否则 ADR-0003 承诺的私有化交付不可复现。
- **抽取与评估必须可复现**：固定语料快照、固定提示词版本、记录模型名与参数。同一份语料跑两次要能给出可比的结果。
- **端口避开 8000 与 8080**（被 `cook-rag` 占用），默认 API 8010、Web 8081。

## 约定

- 中文写文档、注释、提交信息。
- 密钥只走环境变量，`.env` 不入库；`data/` 下的原始语料不入库。
- 依赖用 `uv` 管理，提交 `uv.lock`。
- 验证脚本时使用显式解释器路径，不依赖隐式激活。

## Agent skills

### Issue tracker

Issues live as GitHub issues on `slowbyte-run/intelligent-diagnosis-rag`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context: `GLOSSARY-MAP.md` at the root points at `src/core`, `src/rotating`, `src/pulmonary`. See `docs/agents/domain.md`.

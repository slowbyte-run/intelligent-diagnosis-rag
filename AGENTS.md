## Agent skills

### Issue tracker

Issues live as GitHub issues on `slowbyte-run/intelligent-diagnosis-rag`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context: `GLOSSARY-MAP.md` at the root points at `src/core`, `src/rotating`, `src/pulmonary`. See `docs/agents/domain.md`.

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

## 约定

- 中文写文档、注释、提交信息。
- 密钥只走环境变量，`.env` 不入库；`data/` 下的原始语料不入库。
- 依赖用 `uv` 管理，提交 `uv.lock`。
- 验证脚本时使用显式解释器路径，不依赖隐式激活。

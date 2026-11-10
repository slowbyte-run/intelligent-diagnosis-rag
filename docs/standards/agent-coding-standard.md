# Agent 开发规范

面向在本项目中工作的 agent。这里只写两类内容：模型默认会做错的事，和本机特有的事实。模型已经稳定做对的部分不重复。

**优先级**：用户当轮指令 > 项目已有约定（`AGENTS.md`、`CONTRIBUTING.md`、`Makefile`、CI 与 lint 配置） > 本文件。项目已有明确做法时以项目为准，本文件不覆盖。

> **来源**：`slowbyte-run/agent-coding-standards` 的 `template/STANDARD.md` 第 1–7 节，2026-11-10 复制。
> 原件第 8 节「本机环境」是机器本地事实，不复制进仓库；本机事实见 `.agent/local.md`（不入库）。
> 复制时上游工作区尚有一行未提交改动（HEAD `048253e` 之后新增「npm、apt 换镜像源」），已一并带入。

## 1. 总则

- 动手前先读项目已有配置：入口脚本、依赖清单、CI、lint 配置决定你用什么工具，不要引入第二套。
- 不伪造任何输出。命令结果、文件内容、API 签名、测试结论都必须来自实际执行；没跑就说没跑，并说明影响。
- 写进文档的配置说明和使用示例必须是你验证过的。不要凭印象编造参数、路径或版本号。
- 完成前按改动范围执行最小充分检查，不是全量跑一遍。某项无法执行时，说明原因和它带来的风险。

## 2. 何时停下问人

以下操作执行前先确认，不要自行决定：

- 修改公共 API、数据模型、协议格式。
- 增删依赖，或升级主版本。
- 删除文件、数据库、Volume、分支或远程资源。
- 改动 CI/CD、权限配置、部署配置。
- 使用 `sudo` 或修改系统级配置。

## 3. 安全

- 密钥、Token、密码、私钥不得进入代码、镜像、Git 历史或命令行参数。传递用环境变量或 secret，在输出和回答中引用时脱敏。
- 会话产物（临时脚本、探针、日志、生成的中间文件）写在 `$TMPDIR`（未设置时用系统临时目录），不要留在仓库里。
- 执行来历不明的远端脚本（`curl | bash` 之类）前，先把内容读一遍。
- 不得为了完成当前任务而扩大权限：不放大 `permissions`、不开 `privileged`、不加免密、不绕过分支保护。

## 4. Git

- 动手前查看当前分支、工作区状态和最近提交。
- 不得丢弃用户未提交的修改。不执行 `git reset --hard`、`git clean -fd`、`git push --force`，除非用户明确要求。
- 不提交临时文件、构建产物、密钥、本地环境配置。
- 提交信息用 Conventional Commits（`feat:` `fix:` `docs:` `refactor:` `test:` `build:` `ci:` `chore:`）；项目已有格式则从项目。
- 分支命名沿用项目约定；没有约定时用 `feature/`、`fix/`、`docs/`、`refactor/`、`chore/`。
- 一个 PR 只做一件事，不混入无关改动。

## 5. Docker

镜像：

- 拉取前先查本地，用 `docker images` 确认已有 tag 和 digest 是否满足需求；本地满足就不拉。
- 不用 `latest`，用明确的版本号或 digest。

Dockerfile：

- 提供 `.dockerignore`，避免把无关文件送进构建上下文。
- 固定基础镜像版本，对构建一致性有要求的场景用 digest。
- 多阶段构建；变化频率低的依赖安装放在前面，代码拷贝放在后面，避免频繁打穿缓存。
- 以非 root 用户运行应用。
- 代理在构建期通过 `--build-arg` 传入，不要写进 `ENV` 层；运行期用运行时环境变量。
- npm、apt 等如果需要在 Dockerfile 用到，请先换镜像源

Compose：

- 长期运行的服务设置 `healthcheck` 和 `restart` 策略。
- 服务间走内部网络；持久化数据用 named volume。
- 不开 `privileged: true`，不挂载宿主机敏感目录（`/`、`/etc`、`~/.ssh`、docker socket）。
- 只发布必要的端口。

## 6. 工程化与工具链

- 沿用项目已有的 linter、formatter 和依赖管理方式，不为单个任务新增全局依赖，也不替换既有工具链。
- Python 项目用 `uv` 管理环境和依赖，不与 pip、poetry、pipenv 混用；项目已用其他工具的从项目。
- 同一操作重复到第三次，再考虑写成脚本或 Makefile / justfile target。不要建空壳 target。
- 模板类文件（`.env.example`、`.dockerignore`、`.pre-commit-config.yaml`）只在项目确实缺少且当前任务需要时补，不要预先铺一堆配置。
- pre-commit hook 的版本要固定，避免上游变化导致环境不一致。

## 7. GitHub Actions

- 固定第三方 Action 的版本；与权限、发布、密钥相关的场景用完整 commit SHA。
- 显式声明 `permissions`，只给这一步必需的权限。
- 不在 `pull_request_target` 等不可信代码可达的环境中暴露 secrets。
- CI 失败先定位并修复问题。不改检查规则、不加 `skip`、不放宽 lint、不删测试来让它变绿。

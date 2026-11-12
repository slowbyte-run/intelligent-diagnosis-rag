# 检索评估：在冻结的图谱上量检索

对应 issue #32（父规格 #30）。上游是 #31 导出的图谱快照
`src/rotating/idr_rotating/graph_snapshot.json` 与 `docs/graph-snapshot.md`。

## 一句话

`uv run idr eval` 默认只读提交进版本库的图谱快照：不连 Neo4j、不调模型、不读
PDF、不需要 `.env`，两三秒跑完；同一份快照连跑两次，三项指标与逐题明细逐项
一致（只有 `generated_at` 会变）。分数的变化只可能来自检索或回答逻辑，不再
混入抽取抽样。

## 为什么要把图冻住

抽取不可复现：同一篇、同一提示词、`temperature=0` 连抽三次，关系产出是
15 / 10 / 3，断言集合两两重合 38.6%（`docs/extraction-variance.md`）。同一份
代码三次重抽，检索命中率是 95.8% / 64.6% / 68.75%。在会漂移的图上跑检索评估，
「改了 X 有没有变好」无法归因。

快照把一次导出时刻的评分闭包冻成文件。评估只读它，因此：

- 图固定：同一个问题永远看到同一张图；
- 快照固定：同一份文件的内容校验和固定；
- 题固定：金标准集记录它绑定的快照校验和，错配直接失败。

抽取评估（#33/#34）走另一条链：固定证据段与标注集，重复 N 次报告分布。两套
评估不要混着读。

## 命令

```bash
# 默认：只读快照，不需要图数据库、模型或网络
uv run idr eval
uv run idr eval --out data/output/reports/eval-snapshot.json

# 指定另一份快照 / 金标准集（两者校验和必须一致）
uv run idr eval --snapshot /path/to/graph_snapshot.json --golden /path/to/golden.json

# 可选：额外做回源 PDF 校验（见下文「引用准确率拆成两个数」）
uv run idr eval --corpus-dir data/input

# 旧路径：实时图谱 + PDF，用来复核历史基线或确认快照确实是线上图的导出
set -a; source .env; set +a
uv run --env-file .env idr eval --live-graph --corpus-dir data/input \
  --out data/output/reports/eval-live.json
```

图变了要重新固定时：

```bash
set -a; source .env; set +a
uv run --env-file .env idr snapshot          # 输出文件、内容校验和与条目数
uv run pytest src/rotating/tests/test_golden_snapshot_binding.py -q
```

`idr snapshot` 只导出、不改金标准集。重新导出后必须显式更新
`golden_questions.json` 的 `snapshot.checksum`，让「这批评分对应哪张图」成为
一次可审阅的评估资产改动，而不是评估命令的副作用。

## 快照读端口

`idr_core.graph.snapshot.SnapshotGraphReader` 把快照适配成既有的 `GraphReader`
读端口，`ask` 的调用方式一行不改：

```python
reader = SnapshotGraphReader.load(DEFAULT_SNAPSHOT_PATH)
answer = ask(question, deps=AskDeps(graph_store=reader))
```

- `facts_from(concept_id, relation_types)`：按起点与关系类型过滤，返回**未聚合的
  并行边**；聚合仍由 `aggregate_facts` 统一做。
- `concepts_by_ids` / `evidence_by_ids`：只返回快照里存在的记录，缺的不编造。
- `evidence_by_page`：只返回闭包内该页的证据段，按页面字符区间升序。闭包外的
  同页邻居不在文件里，引用展示上下文可能比线上短；不会拼出页面里不存在的句子。
- 边里的锚点当前全为空（M2 冻结图谱没有锚点），读侧按整段引用处理，与线上一致。

**快照评估不接向量兜底。** pgvector 表与嵌入不在快照里，`AskDeps` 只注入
`graph_store` 时，`ask` 保持「图路径失败即拒答」的既有行为。因此这里量的是
图路径检索与引用组装，不是向量召回（#28 单独量）。这个选择是显式的，报告里
不会把「没测向量」算成任何一项分数。

## 校验和绑定

金标准集 `golden_questions.json` 的 `schema_version` 为 2，顶层多了一项：

```json
"snapshot": {"checksum": "sha256:3432…"}
```

- 校验和是快照**内容**的 SHA-256（去掉 `checksum` 字段后的规范化 JSON），
  重新缩进不变，改一个字的证据原文会变（见 `docs/graph-snapshot.md`）。
- 加载金标准集只校验格式；评估入口 `evaluate_snapshot` 在打分前核对实际快照的
  内容校验和。不一致时退出码 1，stderr 写明两个校验和、金标准集路径、快照路径，
  以及「确认换图时把 `snapshot.checksum` 更新为实际值」——不继续跑，也不悄悄
  按新图错分。
- 比较的是内容而不是路径：同一份快照复制到哪里都算同一张图；重新导出后哪怕
  文件名不变也不再是同一张图。

## 三项指标与两个引用口径

报告里的 `metric_definitions` 会随报告留档，下面是同一份定义的摘要。

| 指标 | 定义 | 分母 |
|---|---|---|
| `retrieval_hit_rate` 检索命中率 | 至少命中一个期望证据段；跨文献题还要求每个期望文献至少一条引用 | 可答题 48 |
| `citation_text_consistency` 引用文本自洽率 | 引文是快照中所引证据段原文的逐字片段，且文献标识、页码与快照记录一致 | 实际输出引用条数 |
| `refusal_correctness` 拒答正确率 | 是否拒答与 `expect_refusal` 一致 | 全部 50 题 |

`retrieval_hit_rate_excluding_unreachable`（可达口径）是归因用的附加口径，见下节。

### 决定：引用准确率拆成「文本自洽」与「PDF 可回源」

**选择了 (b) 拆成两个数，而不是 (a) 让快照带上回源所需信息。**

理由：

1. **快照故意只装评分闭包。** 当前 158 段证据、184.5 KiB，可以进版本库审阅 diff。
   只把同页邻居带全就会涨到 1365 段、约 640 KiB；带上完整页面文字更大，快照就不
   再是「能审阅的评估资产」。
2. **PDF 回源验证的是快照资产，不是检索可重复性。** 快照里的证据正文是导出时
   从 PDF 抽的；它是否忠实于 PDF，应该在重新导出快照时核对，而不是每次检索评估
   都重开 20 篇 PDF。快照评估要回答的是「在这份冻结正文上，检索与引用组装有没
   有问题」。
3. **语料不在版本库。** `data/input` 不入库；默认路径不能依赖本机恰好有 PDF。

因此报告里：

- `citation_text_consistency`（默认总是计算）：引用拼装与冻结图谱的自洽性。
  它能抓住「凭空造句」「引文挂错证据段/页/文献」「引用了闭包外的东西」；
  它抓不住「快照证据正文本身与 PDF 不一致」——那是下面这一项。
- `citation_pdf_verifiable`（只有显式传 `--corpus-dir` 才算）：把答案引用逐条
  回源 PDF、在所标页逐字核对，micro 口径不变。未传时 `checked=false`、
  `citation_pdf_accuracy=null`，`unavailable_reason` 写明「只读快照，没有 PDF」；
  **不会把未测当成 100%**。
- CLI 输出把默认那项标成「引用文本自洽率（快照原文逐字，非回源 PDF）」；报告
  的指标名里不再有容易误读的 `citation_accuracy`。

对冻结快照，`citation_text_consistency` 是构造上接近 100% 的指标（引用直接取自
快照原文），它的价值是守护引用组装路径的回归；要判断快照本身是否失真，看
`--corpus-dir` 的那一项。

**当前快照的锚点为空**，引文是证据段原文或其中的查询期选句，因此「逐字片段」
可以直接核对。将来若重新导出带短锚点的快照，引用展示会把同页相邻证据段并入
上下文，一条引文可能跨多段；届时这个口径要改成核对锚点片段（或把同页文本也
纳入校验），否则会把正常的上下文扩展误判为引用失败。这份快照不需要那个复杂度。

## 结构上无法引用的期望证据

导出快照不导出概念→证据内部边（`EVIDENCED_BY`），检索端口也不读它。金标准里
有 **6 条期望证据**只挂在这条链路上，冻结图谱里没有任何关系边引用它们，图路径
在结构上就不可能命中：

| 期望证据 | 涉及题目 |
|---|---|
| `ev:4a8dc5af5de2cfe4:19:3961-4466` | F-21、P-11 |
| `ev:4a8dc5af5de2cfe4:20:2446-2660` | P-01 |
| `ev:4a8dc5af5de2cfe4:21:606-762` | X-06 |
| `ev:742178038a41d5ea:13:1241-1606` | F-11 |
| `ev:906bd81774ba8b1a:33:5848-6279` | F-22、X-02、X-11 |
| `ev:906bd81774ba8b1a:46:8173-8335` | F-14 |

6 条证据散落在 9 道题里（有的题把它列为多条期望证据中的一条）。报告不把它们
当普通未命中：

- `attribution.unreachable_evidence` 列出每条证据与涉及题目；
- 每道题的 `cases[].unreachable_expected_evidence` 标出该题哪几条期望证据不可达；
- 期望证据**全部**不可达的题（当前只有 F-21，期望证据仅 1 条）从
  `retrieval_hit_rate_excluding_unreachable` 的分母中排除，并单列
  `unreachable_question_ids`；
- 部分不可达的题（8 道）仍按全量口径参与命中率，因为它们的其他期望证据可达；
  未命中时 `cases[].miss_attribution` 记为
  `partially_unreachable_expected_evidence`。
- 全量口径 `retrieval_hit_rate` 保持不变，便于与历史基线对照。

## 当前基线（提交的这份快照）

图谱：61 个概念 / 165 条关系边 / 158 段证据，内容校验和
`sha256:343227369845f2927e78a16e9a9460d37b4eace9e21869a4a7144bed3bd02e58`。
金标准：50 题（可答 48，拒答对照 2）。

| 指标 | 数值 |
|---|---|
| 检索命中率（全量口径） | 37/48 = **77.1%** |
| 事实型 | 20/24 = 83.3% |
| 对比型 | 9/12 = 75% |
| 跨文献型 | 8/12 = 66.7% |
| 引用文本自洽率 | 843/843 = **100%** |
| 拒答正确率 | 47/50 = **94%** |
| 可达口径检索命中率（排除 F-21） | 37/47 = **78.7%** |
| 引用回源 PDF | 未测（`null`，未传 `--corpus-dir`） |

逐题明细：`cases[].miss_attribution` 把 11 道未命中题分成普通漏检与内部边归因；
3 道可答题被拒（F-21、F-25、P-11），拒答正确率因此是 47/50。路径分布：图命中
45、拒答 5、向量兜底 0。

与旧 50 题基线（`docs/m2-baseline.md`）的关系：

- 检索命中率 77.1% 与旧基线数值相同，按题型拆分也相同（20/24、9/12、8/12）；
- 旧基线是在会漂移的图上、含向量兜底依赖跑出的**抽样**，引用准确率 845/845 是
  回源 PDF 口径；本表是固定快照、无向量兜底的**可重复**口径，引用数 843 来自
  图路径引用的实际输出。数值接近不代表可以逐项对比，引用那一栏尤其不是同一个
  度量；
- 旧基线与旧报告原样保留，用 `idr eval --live-graph` 可复核；`data/` 下的旧报告
  不入库，但 `docs/m2-baseline.md` 记录了当时的运行条件与失败分析。

## 两次运行一致性的复核

```bash
uv run idr eval --out run1.json
uv run idr eval --out run2.json
python - <<'PY'
import json
a = json.load(open("run1.json")); b = json.load(open("run2.json"))
print(a["result_digest"] == b["result_digest"])   # True
a.pop("generated_at"); b.pop("generated_at")
print(a == b)                                     # True
PY
```

`result_digest` 是对 `metrics`、`attribution`、引用 PDF 口径与逐题明细算的
SHA-256，不含 `generated_at`；两次运行摘要相同即结果一致。当前提交上的摘要为
`sha256:f9615e6858da0ae5d712f06304facd495105ffe659665407612c64e3b2459e64`。
报告里只有 `generated_at` 会变。

## 这套评估能回答与不能回答什么

能：

- 改了检索/遍历/引用组装后，固定图谱上的三项指标有没有变化；
- 变更前后逐题差异、失败类型（缺概念、期望证据未命中、缺文献、引用校验失败）；
- 快照与金标准集是否对应（校验和）；
- 期望证据在冻结图谱上是否结构可达。

不能：

- 抽取质量与稳定性（看抽取评估，固定证据段与标注集）；
- 向量召回质量与兜底阈值（快照不含嵌入与 pgvector）；
- 闭包外的问题（需要重新导出快照）；
- 快照证据正文是否忠实于 PDF（除非显式传 `--corpus-dir`，且那是另一个数）。

## 相关

- 快照格式与导出：`docs/graph-snapshot.md`
- 旧 50 题基线：`docs/m2-baseline.md`
- 抽取方差：`docs/extraction-variance.md`
- 父规格：issue #30；本票：#32

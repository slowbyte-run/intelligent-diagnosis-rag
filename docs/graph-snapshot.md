# 图谱快照：把评分用的图冻成一份文件

对应 issue #31（父规格 #30）。文件本体是 `src/rotating/idr_rotating/graph_snapshot.json`，
与金标准集、本体同级，作为领域包的评估资产进版本库。

## 为什么要有快照

抽取不可复现（同输入两次重抽，断言集合只重合 38.6%，见 `extraction-variance.md`），
图谱因此会随重抽漂移。在漂移的图上跑检索评估，「这次改动变好还是变坏」无法归因：
同一份代码三次重抽，检索命中率能是 95.8% / 64.6% / 68.75%。

快照把一次导出时刻的图闭包冻结成文件。检索评估只读快照，不连 Neo4j、不重新抽取，
同一个问题永远看到同一张图，分数的变化只可能来自检索或回答逻辑。

## 一条命令

```bash
set -a; source .env; set +a
uv run idr snapshot                       # 写到领域包内默认位置
uv run idr snapshot --out /tmp/s.json     # 或指定位置
```

命令输出文件路径、内容校验和与条目数，并显式报告「金标准绑定、但不在任何关系边上」
的期望证据（当前 6 条）。导出只走 `GraphReader` 读端口，只读：不写图谱、不碰语料、
不联网（除 Neo4j 本身）。

当前线上图谱导出（图谱冻结于 M2）：50 题、61 个概念、165 条关系边、158 段证据段，
约 185 KiB。整库是 78 个概念、179 条关系边、6806 段证据——快照只取闭包，证据段约为
整库的 2.3%。

## 闭包规则

与 `idr_rotating.traversal.traverse` 的读取范围一致，否则快照上的评估会与线上行为分叉：

1. **起点**：金标准每题 `expected_concepts` 的并集，加上每题 `question` 经本体别名表
   链接到的概念。评分调用的是 `ask(question)`，问句链接到的概念会成为检索起点；
   只按题面绑定的概念导出，会让「问句命中、但期望概念里没写」的题在快照上改变行为。
   问句链接到、但图谱里不存在的概念跳过——线上 `ask` 同样查不到、同样拒答。
2. **一跳**：从每个起点取白名单关系（`EDGE_WHITELIST`，六类）的出边。
3. **两跳**：一跳里沿结构关系（`has_component` / `has_fault_mode`）到达的节点作为
   中转，再取它们的白名单出边。三跳及以上不取（`MAX_HOPS = 2`）。
4. **边**：同一对概念、同一关系、同一文献的一条并行边原样保留，不聚合；没有
   `evidence_ids` 的边不导出（聚合层本来就会丢弃）。
5. **概念**：起点与已导出边的两端。边的端点缺失就是闭包不完整，导出失败。
6. **证据段**：已导出边引用的证据段，加上金标准直接绑定的证据段。后者即使不在任何
   关系边上也带正文进快照，加载期才能核对「题目引用的东西确实存在」。

闭包之外的数据一条不进：闭包外的节点、非白名单关系、三跳边、未被引用的证据段原文。
金标准绑定的概念或证据段在线上图谱里不存在时，导出直接报错，不生成半份快照。

## 文件格式

`schema_version = 1`、`format = "idr-graph-snapshot"`。顶层键序固定，集合全部定序：

```json
{
  "format": "idr-graph-snapshot",
  "schema_version": 1,
  "checksum": "sha256:…",
  "closure": {
    "seed_concept_ids": ["concept:…"],
    "relation_types": ["has_component", "…"],
    "structural_relations": ["has_component", "has_fault_mode"],
    "max_hops": 2
  },
  "concepts": [
    {"id": "concept:…", "name": "…", "type": "…"}
  ],
  "edges": [
    {
      "head_id": "concept:…",
      "relation": "indicated_by",
      "tail_id": "concept:…",
      "doc_id": "0123456789abcdef",
      "evidence_ids": ["ev:…"],
      "confidence": 0.75,
      "anchors": [
        {"evidence_id": "ev:…", "start": 0, "end": 12, "ambiguous": false}
      ]
    }
  ],
  "evidence": [
    {
      "id": "ev:…",
      "doc_id": "0123456789abcdef",
      "page": 12,
      "charspan": [357, 1511],
      "text": "证据段正文…"
    }
  ]
}
```

约定：

- `closure` 说明这份快照覆盖的图范围；它是文件的一部分，随校验和一起被
  金标准集记录，用来回答「这批评分用的是哪版图谱」。
- `edges` 保留并行边（同一概念对、每篇文献一条）；聚合规则仍在读侧
  （`aggregate_facts`），快照不做。边内部 `evidence_ids` 与 `anchors` 按证据段标识排序。
- `anchors` 的 `[start, end)` 是证据段 `text` 内的字符区间，左闭右开；
  `ambiguous` 表示同一片段在该段里出现多次。锚点可以缺失或只覆盖部分证据段
  （旧边没有锚点），读侧对此必须退回整段引用。
- `evidence.charspan` 是证据段在所属页面文字里的字符区间，用于同页上下文的
  连续切片判断。
- 不导出 `bbox`、文献名、PDF 与概念→证据内部边（`EVIDENCED_BY`）：检索端口不读它们。

## 校验和与确定性

校验和是**内容**校验和：对去掉 `checksum` 字段后的规范化 JSON 算 SHA-256。

- 规范化 = 键排序、无多余空白、`ensure_ascii=False`、UTF-8 编码。因此重新缩进文件
  不会改变校验和，改一个字的证据原文会。
- 同一份图连续导出两次，字节与校验和完全一致：文件里没有时间戳、主机名、绝对路径；
  `concepts` / `edges` / `evidence` / `closure` 的各集合与边内部的证据清单、锚点全部
  定序。`tests/test_snapshot_integration.py` 在真实图谱上验证这一点。
- 读侧加载即校验：`idr_core.graph.snapshot.load_snapshot(path)` 先核校验和，再解析；
  任何字段缺失、拼错或版本不认识都报 `SnapshotError`，不退回默认值。

## 读侧（#32 已实现）

`idr_core.graph.snapshot.SnapshotGraphReader` 把快照适配成既有的 `GraphReader`
读端口，`ask` 的调用方式一行不改。行为与在线端口一致：

- `facts_from(concept_id, relation_types)`：按 `head_id` 精确匹配、按 `relation` 过滤，
  返回未聚合的 `FactEdge`（并行边原样返回，交给 `aggregate_facts`）。
- `concepts_by_ids` / `evidence_by_ids`：按标识过滤；查不到的标识不出现在结果里，
  与线上端口一致，不为缺失项编造记录。
- `evidence_by_page(doc_id, page)`：返回快照内该页的证据段，按 `charspan.start` 升序。
  **快照不含闭包外的同页邻居**；短锚点的展示上下文因此可能比线上短。退回的展示文本
  仍是页面文字的连续切片，不会拼出页面里不存在的句子。
- 加载时校验 `format`/`schema_version`/`checksum`；金标准集记录的校验和与文件不符时
  评估直接失败并说明原因。
- 不用快照驱动向量兜底：pgvector 表与嵌入不在快照里。检索评估只注入
  `graph_store`，`ask` 因此保持「图路径失败即拒答」。

引用准确率在快照路径下拆成「文本自洽」与「PDF 可回源」两个数，理由与口径见
`docs/retrieval-eval.md`；评估命令、校验和绑定与当前基线也在那一页。

## 已知边界

- **不是图数据库备份。** 闭包之外的节点与边不在文件里，不能用它回答闭包外的问题；
  那类问题应当重新导出，而不是让读侧猜。
- **同页邻居不导出。** 闭包内 158 段证据分布在 81 页；若把同页邻居全带上会涨到
  1365 段、正文多 213k 字符（约 640 KiB），快照就不再是「能审阅的 diff」。
- **概念→证据内部边不导出。** 检索端口不读它，但金标准里有 6 条期望证据只挂在
  这条链路上，图路径无法引用它们；导出命令会打印这份清单，不让它静默埋着。
- **当前图谱里锚点数组全为空。** 锚点字段总是写入（有锚点时按 `[start, end)` 导出），
  但 M2 冻结的这张图没有锚点，读侧对空锚点必须按整段引用处理——这与线上
  `ask` 的行为一致，不是导出丢了数据。

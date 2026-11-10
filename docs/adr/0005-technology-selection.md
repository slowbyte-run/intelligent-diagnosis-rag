# 技术选型：分层混合，不引入编排框架

每个链路只在「成熟库能当零件用」时才引入第三方：PDF 解析（pdfplumber、PaddleOCR）、图驱动（`neo4j`）、向量（pgvector + psycopg）、API（FastAPI）用库；知识层——本体校验、抽取提示词、章节切分、检索编排、生成与引用、评估——全部自写。**不引入 LangChain / LangGraph / LlamaIndex，也不把任何 KG-RAG 框架当运行时。**

选择理由有两条。其一，本项目的差异化全部在知识层，而受评的九个框架没有一个接受「抽取必须过本体校验才入库」这条硬约束，它们都是抽完直接写。其二，本项目的编排是确定性流水线（解析→抽取→写图→检索→生成），不是 agent 式的动态决策，图编排在这里是负担。单人维护还意味着：任何需要持续跟上游合并分支的方案都应排除。

## Considered Options

- **MS GraphRAG**：无 Neo4j 存储实现（只有 LanceDB / Azure AI Search / CosmosDB），引用只到文档级。排除。
- **RAGFlow**：全仓 `neo4j` 命中 0，图存在自己的 ES 里；默认栈带 SSPL 的 Elasticsearch。排除。
- **KAG**：必须先部署 OpenSPG 引擎，半年无 release。排除。
- **LightRAG**：唯一同时支持 Neo4j + pgvector、且把 `reference_id` 段落引用写进提示词契约的候选。**不作为依赖，但其引用契约与提示词结构被借鉴。**
- **neo4j-graphrag**：Apache-2.0、活跃，`SimpleKGPipeline` 的 Document/Chunk 词法图值得借鉴；但它同样没有本体层。**不作为依赖。**
- **LlamaIndex**：零件箱而非流水线。排除。
- **RAGAS**：依赖 LangChain，且 `faithfulness`（答案能否由上下文推出）与本项目的**引用准确率**（结论能否在它标注的那一段里找到）不是同一个定义。排除。

## Consequences

- **运行时依赖白名单**：`pdfplumber`、`paddleocr`（可选 extra）、`pydantic`、`openai`（仅作 OpenAI 兼容客户端）、`psycopg`、`neo4j`、`fastapi`、`uvicorn`、`pyyaml`、`tenacity`。新增依赖需先说明归属层（`src/core` 还是领域包）并更新本清单。
- **许可证排除**：PyMuPDF（AGPL-3.0，会被链接进本进程，网络服务也会被传染）、Marker（权重 OpenRAIL-M）。Neo4j Community 的 GPL-3.0 另行接受，见 ADR-0003。
- **中文关键词检索暂缓**：Postgres 默认不支持中文分词，为避免私有化交付时要求客户编译扩展，阶段 1–2 只用「向量召回 + 图遍历 + rerank」；等有了评估基线，再拿数字决定是否引入分词扩展。
- **放弃框架的代价**：多格式文档支持、内置增量索引、社区维护的提示词，都要自己补。这是有意用工作量换取可控性。

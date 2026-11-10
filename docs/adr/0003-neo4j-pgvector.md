# 图用 Neo4j，向量用 Postgres + pgvector

图谱存储选 Neo4j 5 Community，向量与元数据选 Postgres + pgvector，两者用 docker compose 各起一个服务。证据节点的正文与位置偏移存 Neo4j，同一证据段的向量存在 pgvector，以证据节点 ID 关联。

本项目相对同类系统的核心增量是**多跳图推理**——从征兆经故障模式追到部件、再追到文献证据——而 Apache AGE 的短板恰好落在多跳查询、图算法和可视化上；客户演示时需要直接展示子图，Neo4j Browser 是现成的，AGE Viewer 生态则基本停滞。私有化交付签长期维护合同后，「Postgres 大版本被 AGE 钉死」是真实负债。向量部分不需要专用向量库，pgvector 足够，因此不引入 Qdrant 之类的第三个服务。

## 与本仓库其他项目的关系

`rag-anything/cook-rag` 的 ADR-0001 做了相反的选择（单 Postgres + AGE 承载关系、向量与 agent 状态）。那条决策针对的是**浅图**：菜品与配料之间只有一跳关系，图只是检索的补充。本项目的图是**双层且需要多跳遍历**的推理结构，两者前提不同，因此结论不同。不要以为其中一处写错了。

## 许可证：接受 GPL-3.0，并承担分发义务

Neo4j Community 是 **GPL-3.0**。已确认接受，但两点必须说清楚：

- **不会传染本项目的代码。** 服务通过 Bolt 与 Neo4j 通信，是两个独立进程，不构成衍生作品。
- **但私有化交付属于分发。** 把 Neo4j 打进镜像送到客户机房，就要履行 GPL-3.0 义务：随交付提供对应源码或书面要约、保留许可与版权声明、不得阻止客户替换该组件。

注意这条只适用于 **GPL-3.0**，不适用于 AGPL。同样的理由**不**适用于 PyMuPDF 之类会被链接进本进程的 AGPL 组件（见 ADR-0005）。

## Consequences

- 私有化交付从「一个数据库」变成「两个服务」。docker compose 一并交付，客户需要接受多一个容器。
- Neo4j 的 JVM 堆在 15 GB 内存的开发机上要显式限制（建议 `NEO4J_server_memory_heap_max__size=1G`），否则会挤掉其余服务。
- 存储 schema 一旦定稿，迁移到另一套图存储等于重写全部 Cypher 与装载逻辑；这是本项目最贵的一次性决策。

## 修订记录

- **2026-11-11 去掉 Postgres 主版本号。** 初稿写作「Postgres 16 + pgvector」，但主版本与本决策的实质（选 pgvector 而不是专用向量库）无关，属于过度限定；实现时用的是 `pgvector/pgvector:0.8.7-pg17`，镜像标签钉到 pgvector 版本，不钉 Postgres 主版本。ADR 不该规定与决策理由无关的细节。


# 设计取舍与源码导航

## 一条调查如何执行

1. 绑定游戏和会话，解析日期、玩家条件、来源与任务要求。
2. 检查资料和覆盖，定位目标补丁；需要刷新或补采时受授权和数量约束。
3. 条件统计使用确定性过滤和计数；问题证据使用混合检索。
4. 读取执行反馈，识别缺口，在步骤、时间及请求预算内补查。
5. 对数字路径和值、来源哈希及逐字引文核验，生成摘要与完整报告。
6. 无法完成时保留已验证结果和限制；不以无数据冒充零反馈。

## 核心模块

| 模块 | 文件 | 作用 |
|---|---|---|
| 状态与编排 | [runtime.py](../gamelens/core/runtime.py) | decide/execute/finalize/repair/partial/fail 状态循环 |
| 工具定义 | [registry.py](../gamelens/tools/registry.py) | Schema、调用说明、审批和执行策略 |
| 条件契约 | [query_contract.py](../gamelens/core/query_contract.py) | 日期、分组、人群口径与工具参数一致性 |
| 统计 | [investigation.py](../gamelens/tools/investigation.py) | 筛选与分组统计，不用检索 Top-K 当分母 |
| 检索 | [index.py](../gamelens/rag/index.py)、[tools.py](../gamelens/rag/tools.py) | 原文切块、双路召回、融合、可选重排与筛选 |
| 证据核验 | [grounding.py](../gamelens/core/grounding.py) | 结构化事实与原文引用绑定 |
| 执行与预算 | [executor.py](../gamelens/tools/executor.py)、[budget.py](../gamelens/core/budget.py) | 隔离进程、超时清理、有限重试和原子请求预留 |
| 评测 | [benchmark_score.py](../gamelens/evaluation/benchmark_score.py) | 硬校验与语义评分分开记录 |

## 检索设计

硬条件过滤 → BM25 与 BGE-M3 Dense 独立召回 → RRF 融合 → 来源去重候选池 → 可选 BGE CrossEncoder → 来源配额/Top-K → 聚焦流程 LLM 证据角色筛选 → 原文核验。

- BM25 k1=1.2、b=0.75；RRF 以各路名次 `1/(60+rank)` 为基础融合，小幅主题提示不是硬删除。
- 默认 700 字符切块、100 字符重叠，原始子串与 offset 保留；清洗只改检索文本。
- BGE-M3 固定 revision，CPU 编码，1024 维；SQLite 保存 Chunk 与 JSON 向量，当前为扫描式相似度计算，不是 ANN 服务。
- BGE Reranker 固定 revision，候选最多 20，CPU batch 2；默认 off，auto 降级必须记录，required 失败不能冒充已测。
- 公告讨论留言是独立精确采集/查询来源，目前不在 `GameIndex.corpus()` 向量语料内。

## 为什么这样选

单 Agent 适合共享同一目标和过滤条件的调查，不为技术覆盖增加多 Agent 通信。SQLite 适合本地单用户与受限规模，后续扩展要先测瓶颈。统计分母必须确定，LLM 不能自行更改。Reranker 不能找回召回阶段已丢失的证据；实测门槛未通过，不能为了展示默认打开。

会话记忆、模型上下文投影和 Checkpoint 分开：历史模型结论不视为事实，当前明确条件优先。预算、来源和引用检查改善可追溯性，但不证明语义或因果结论必然正确。

# Changelog

## 4.0.0 (2026-07-16)

**重大升级**：移植 LivingMemory 核心创新 — MemoryAtom 记忆系统，让 Bot 拥有"活"的长期记忆。

从 [astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory) 学习并移植了以下优秀设计（不包含图谱模块，因工作量过大且当前 KB RAG 已覆盖文档检索场景）：

### 新增功能

#### 1. MemoryAtom 数据模型 `astrdb/memory/models.py`

把记忆从"整段总结"升级为"细粒度原子"。每个 atom 是独立的事实单元，拥有：
- **5 种类型**：EPISODIC(事件) / FACTUAL(事实) / RELATIONAL(关系) / PREFERENCE(偏好) / PLANNED(计划)
- **独立 TTL**：按类型+重要性+强化次数动态计算（公式：`base_ttl × (0.5 + importance) × (1 + min(0.5, reinforcement × 0.1))`）
- **3 种衰减曲线**：LINEAR / EXPONENTIAL / STEP（PLANNED 类型到期骤降至 0.05）
- **四级生命周期**：ACTIVE → EXPIRED → FORGOTTEN → 物理删除
- **强化机制**：被相似内容命中时 confidence EMA 更新 + TTL 续期

#### 2. AtomClassifier 纯规则分类 `astrdb/memory/atom_classifier.py`

**零 LLM 调用**，基于正则模式匹配：
- PLANNED：时间指示词 + 动作动词（"明天开会"）
- PREFERENCE：偏好词（"喜欢"、"讨厌"）
- RELATIONAL：关系词（"同事"、"朋友"）
- FACTUAL：状态词（"是"、"有"、"属于"）
- EPISODIC：动作但无时间

支持中文相对时间解析（"明天"/"下周三"/"5月30日" → unix timestamp）。

#### 3. AtomLifecycleManager `astrdb/memory/lifecycle.py`

四级状态机 + 强化机制：
- `expire_stale_atoms()`：ACTIVE → EXPIRED（TTL 到期）
- `forget_expired_atoms()`：EXPIRED + 7天 → FORGOTTEN（从 FTS 移除）
- `cleanup_forgotten()`：FORGOTTEN + 30天 → 物理删除
- `reinforce_similar()`：Jaccard 相似度 ≥ 0.6 时强化（confidence EMA + TTL 续期）

#### 4. DecayScheduler `astrdb/memory/decay_scheduler.py`

每日定时衰减 + 自动清理：
- 每日 00:05 执行（可配置）
- 重要性衰减：`importance × (1-decay_rate)^days`
- 访问强化降权：最近 30 天访问过的 atom 衰减率减半
- 自动清理：删除 >30天 且 importance<0.3 的 atoms
- 生命周期维护：状态机推进

#### 5. MemoryProcessor 双通道摘要 `astrdb/memory/processor.py`

对话总结 + atom 抽取：
- 把对话消息格式化为 LLM 可读文本（群聊带昵称+时间前缀）
- 调 MaiBot LLM 生成 JSON（summary / key_facts / topics / importance）
- JSON 解析容错（直接解析 / markdown 代码块 / 正则提取 / 尾逗号修复）
- 质量校验（summary 长度 / key_facts 非空 / importance 范围）
- **双通道摘要**：canonical_summary（检索用，summary + key_facts 拼接）+ persona_summary（注入用，纯 summary）

#### 6. AtomRetriever 多维加权检索 `astrdb/memory/retriever.py`

移植 LivingMemory HybridRetriever 核心思想：
- **多维加权**：`score = bm25×0.5 + importance×0.25 + recency×0.25`（再乘 decay_factor）
- **MMR 去重**：Jaccard 词袋相似度，避免返回重复内容
- **LRU 缓存**：相同查询 45s 内复用结果（写入/删除/衰减时自动失效）
- **score_breakdown**：返回各维度分数明细，便于调试

#### 7. 多模式注入适配器 `astrdb/memory/injection.py`

3 种注入方式（LivingMemory 有 6 种，简化为最常用的）：
- `extra_user_content`：追加新 user message（推荐，不污染历史）
- `user_message_before`：拼到当前 user message 前面
- `user_message_after`：拼到当前 user message 后面

注入文本带明确 BEGIN/END 标记 + 类型/重要性/时间信息 + "PAST records, trust current conversation" 提示。

#### 8. 8 个 Memory API + 2 个 LLM Tool

**API**：
- `astrdb.memorize` — 主动记忆
- `astrdb.mem.search` — 检索记忆
- `astrdb.mem.list` — 列出 atoms
- `astrdb.mem.stats` — 记忆统计
- `astrdb.mem.reinforce` — 强化 atom
- `astrdb.mem.forget` — 删除 atom
- `astrdb.mem.decay_now` — 手动触发衰减
- `astrdb.mem.process` — 处理对话生成 atoms

**LLM Tool**：
- `memory_search` — 让 LLM 主动检索记忆
- `memory_memorize` — 让 LLM 主动记忆

#### 9. 自动注入 Hook

注册 `@HookHandler("maisaka.planner.before_request", order=LATE)`，在 LLM 调用前自动检索记忆并注入。与 KB 注入器（order=NORMAL）并存，各自注入不同内容。

### 与 A_memorix 的关系

本插件的 Memory 模块**不直接取代** MaiBot 自带的 A_memorix，而是作为**补充**：
- A_memorix：聊天摘要 + 人物画像（MaiBot 内置，深度集成）
- 本插件 Memory：细粒度 MemoryAtom + 衰减调度 + 多维加权检索（插件层，可禁用）

两者可以并存，各自注入 user message。如果用户想完全用本插件取代 A_memorix，可以在 MaiBot 配置中禁用 A_memorix。

### 测试

- **93 个 pytest 测试全部通过**（74 原有 + 19 Memory 新增）
  - test_memory.py — Memory 模块完整测试（19 个）
    - 模型测试（TTL/衰减/类型配置）
    - 分类器测试（5 种类型识别）
    - AtomStore CRUD + FTS + 强化
    - 生命周期测试（过期→遗忘→清理）
    - 衰减测试
    - 检索器测试（多维加权 + MMR + 缓存）
    - 注入测试

### 设计来源

| LivingMemory 模块 | 本插件对应 | 移植方式 |
|---|---|---|
| `core/models/memory_atom.py` | `astrdb/memory/models.py` | 完整移植，简化为 SQLModel |
| `core/processors/atom_classifier.py` | `astrdb/memory/atom_classifier.py` | 完整移植，零改造 |
| `storage/atom_store.py` | `astrdb/memory/atom_store.py` | 简化移植，去掉 AstrBot logger 依赖 |
| `core/managers/atom_lifecycle_manager.py` | `astrdb/memory/lifecycle.py` | 简化移植 |
| `core/schedulers/decay_scheduler.py` | `astrdb/memory/decay_scheduler.py` | 简化移植，去掉备份逻辑 |
| `core/processors/memory_processor.py` | `astrdb/memory/processor.py` | 简化移植，LLM 调用改为延迟注入 |
| `core/retrieval/hybrid_retriever.py` | `astrdb/memory/retriever.py` | 移植核心思想（多维加权+MMR+缓存） |
| `core/utils/injection_adapter.py` | `astrdb/memory/injection.py` | 简化为 3 种注入方式 |

**未移植**（取舍）：
- ❌ GraphMemory（图谱）— 工作量大，当前 KB RAG 已覆盖文档检索
- ❌ DualRouteRetriever — 依赖图谱
- ❌ Web UI 图谱可视化 — 工作量大
- ❌ FAISS — 当前 numpy 在 1 万 chunk 以下性能足够

---

## 3.0.0 (2026-07-16)

**重大升级**：补齐召回与注入机制、消息前缀拦截、Web 管理界面三大功能，全面完善插件可用性。

### 新增功能

#### 1. 消息前缀拦截器 `astrdb/interceptor.py`

通过 MaiBot Hook 机制实现"不记录、不回复"的消息拦截。

- 注册 `@HookHandler("chat.receive.before_process", mode=BLOCKING, order=EARLY)`
- 命中前缀时返回 `{"action": "abort"}`，主链路直接 return
- 拦截后消息不进 chat_manager、不进 message_repository、A_memorix 也读不到
- 默认前缀：`/` `[` `#`
- 可配置前缀列表，支持长短前缀优先匹配（`!!` 优先于 `!`）
- 可配置是否记录被拦截消息的预览（前 60 字）

配置：
```toml
[interceptor]
enabled = true
prefixes = ["/", "[", "#"]
log_blocked = true
```

#### 2. 自动召回 + 注入器 `astrdb/injector.py`

在 LLM 调用前自动检索知识库并注入到 prompt，与 A_memorix 兼容。

- 注册 `@HookHandler("maisaka.planner.before_request", mode=BLOCKING)`
- 在 planner 构造完 messages 之后、调 LLM 之前触发
- 自动提取最后一条 user 消息作为 query
- 检索知识库，命中高置信度结果时作为新 user message append 到 messages
- **去重逻辑**：检查最近 N 条消息是否有 `knowledge_search` tool 调用，有则跳过（避免与 LLM 主动调重复）
- 与 A_memorix 的 heuristic_injector 互不冲突（两者各自追加 user message）

配置：
```toml
[injector]
enabled = true
min_score = 0.01           # RRF 融合分数阈值
min_vector_score = 0.3     # 向量相似度阈值
top_k = 3                  # 注入几条
max_chars = 2000           # 注入文本最大字符数
dedup_lookback = 6         # 去重检查的消息数
skip_if_tool_called = true # LLM 已调过 tool 时跳过
```

#### 3. Web 管理界面 `astrdb/webui/server.py`

独立的 FastAPI + uvicorn Web server，提供完整的知识库管理界面。

**端点**：
- `GET /` — SPA HTML 页面（嵌入式前端，无需分发多文件）
- `GET /api/stats` — 知识库统计
- `GET /api/files` — 文件列表（支持 status/category 过滤）
- `GET /api/files/{file_id}/chunks` — 查看某文件的切片详情
- `POST /api/search` — 检索测试（支持混合/向量/BM25 三种模式）
- `POST /api/ingest` — 触发增量导入
- `POST /api/rebuild` — 强制全量重建
- `GET /api/config` — 读取当前配置
- `PUT /api/config` — 更新配置（写 config.toml，触发 FileWatcher 热重载）

**前端功能**：
- 📊 统计面板：文件数、chunks 数、tokens、大小、embedding 模型
- 📁 文件面板：文件列表 + 状态过滤 + 切片查看
- 🔍 检索测试：实时查询，显示分数、来源、章节路径
- ⚙️ 配置面板：JSON 编辑器，保存后自动热重载

**安全**：
- 默认监听 127.0.0.1（只本机访问）
- 可配置 token 认证（Bearer token）
- 端口可配置（默认 8765，避开 MaiBot WebUI 的 8001）

配置：
```toml
[webui]
enabled = true
host = "127.0.0.1"
port = 8765
token = ""  # 留空则无认证
```

### A_memorix 兼容性

本插件与 MaiBot 自带的 A_memorix 记忆系统**完全兼容，互不冲突**：

| 维度 | A_memorix | 本插件 KB |
|---|---|---|
| 数据来源 | 聊天摘要、人物画像 | 用户导入的文档 |
| 存储 | MaiBot 主库 + FAISS | 独立 SQLite + numpy |
| 检索 | chat_scope 隔离 | 全局可见 |
| 注入点 | `_build_planner_injected_user_messages` | `maisaka.planner.before_request` Hook |
| 注入方式 | heuristic user message | 自动召回的 user message |
| 冲突 | 无（两者各自追加 user message） | 无 |

启动时若检测到 A_memorix 启用，本插件的自动注入器会智能去重，避免与 LLM 主动调 `knowledge_search` tool 重复。

### 测试

- **74 个 pytest 测试全部通过**（47 原有 + 27 新增）
  - test_interceptor.py — 前缀拦截器（8 个测试）
  - test_injector.py — 自动注入器（8 个测试）
  - test_webui.py — Web UI 端到端（7 个测试，含 token 认证）

### 依赖

新增：
- `fastapi >= 0.100.0` — Web 框架
- `uvicorn >= 0.23.0` — ASGI server
- `tomlkit >= 0.12.0` — 配置文件读写（保留注释）

---

## 2.0.0 (2026-07-16)

**重大升级**：新增独立的知识库 RAG 模块 — 通用文本（小说/游戏/世界观设定）向量化作为 LLM 外置知识库。

针对 A_memorix 在结构化知识库场景的三大痛点（滑动窗口硬切、LLM 自由抽取关系混乱、chat_scope 过滤导致检索失败），本模块完全重写：

- **按 markdown 标题语义切分** — 保留 `#`/`##`/`###` 标题层级路径，不做字符数硬切
- **不抽取关系** — 直接存原文 chunks + embedding，避免 LLM 自由抽取的混乱
- **无 chat_scope 过滤** — 知识库就是全局可见的，导入即可被任何会话检索到
- **混合检索** — 向量（numpy cosine）+ BM25（SQLite FTS5 trigram）+ RRF 融合

### 新增功能

- **数据模型**（2 张新表）
  - `kb_files` — 文件元数据（路径、hash、状态、chunk 数、tokens）
  - `kb_chunks` — 切块原文 + embedding（BLOB numpy float32）+ 标题路径

- **FTS5 全文检索**
  - 使用 `trigram` 分词器（SQLite 3.34+），对中文友好
  - 短查询（< 3 字符）回退 LIKE 兜底
  - BM25 分数排序

- **Markdown 语义切分器** `kb/chunker.py`
  - 按 `#` 标题切分章节，保留层级路径
  - 章节内按段落（双换行）累积到目标大小
  - 单段落超长按句号硬切
  - 短尾部 chunk 自动合并到上一个

- **Embedding 抽象层** `kb/embedder.py`
  - `MaiBotEmbedder` — 走 MaiBot `self.ctx.llm.embed`（推荐，零配置）
  - `OpenAICompatibleEmbedder` — 兼容 OpenAI / DeepSeek / Moonshot / 智谱
  - `DummyEmbedder` — 哈希伪随机向量，仅测试用

- **内存向量索引** `kb/vector_store.py`
  - 基于 numpy 矩阵乘法的 cosine 检索
  - 启动时从 SQLite BLOB 加载到内存
  - 增量更新（add/remove）
  - 1 万 chunk × 1024 维 ≈ 40MB 内存

- **混合检索** `kb/search.py`
  - 向量检索 + BM25 检索并行
  - RRF（Reciprocal Rank Fusion）融合两路结果
  - 可分别控制 use_vector / use_bm25
  - 支持 category / file_ids 过滤

- **批量导入器** `kb/importer.py`
  - 扫描目录所有 .md / .txt 文件
  - 增量更新：基于 file_hash 判断变更
  - 自动切分 + embedding + 入库 + 内存索引同步
  - `force_rebuild=True` 强制全量重建

- **8 个 KB 公开 API**
  - `astrdb.kb.ingest_directory` — 批量扫描目录导入
  - `astrdb.kb.ingest_file` — 单文件导入
  - `astrdb.kb.search` — 混合检索（推荐）
  - `astrdb.kb.search_vector` — 仅向量检索
  - `astrdb.kb.search_bm25` — 仅 BM25 检索
  - `astrdb.kb.list_files` — 列出已入库文件
  - `astrdb.kb.delete_file` — 删除某文件及其 chunks
  - `astrdb.kb.stats` — 知识库统计
  - `astrdb.kb.reload_index` — 重载内存索引

- **LLM Tool** `knowledge_search`
  - 让 MaiBot 在对话中主动调用知识库检索
  - 当用户询问世界观/剧情/角色/设定时自动触发
  - 返回格式化文本供 LLM 引用

- **配置**
  - `[knowledge_base]` section 控制：启用、目录、切分参数、embedding 提供方
  - 启动时自动增量导入新文件（可关闭）

### 端到端验证

用真实原神知识库（55 个 markdown 文件，982 KB）验证：
- 导入：55 个文件全部成功，切分出 1211 个 chunks，约 29 万 tokens
- 检索：12 个测试查询全部能找到相关结果
  - "法涅斯是谁" → 命中提瓦特总史第三幕
  - "尼伯龙根" → 命中祷歌 + 总史"提瓦特最初的主人"
  - "世界树" → BM25 分数 8.42 命中基础设定文件
  - "降临者" → BM25 分数 8.73 命中降临者一览表
  - "桑多涅" → 命中至冬第六幕"木偶加入愚人众，成为桑多涅"

### 测试

- **47 个 pytest 测试全部通过**（27 原有 + 20 KB 新增）
  - test_kb_chunker.py — 切分逻辑（10 个测试）
  - test_kb.py — 端到端集成（10 个测试）

---

## 1.0.0 (2026-07-16)

首次发布 — 完整移植 AstrBot 数据库设计到 MaiBot 插件运行时。

### 新增

- **18 张表**：移植自 AstrBot `astrbot/core/db/po.py`
  - `conversations`、`preferences`、`platform_message_history`、`platform_sessions`
  - `personas`、`persona_folders`、`cron_jobs`
  - `platform_stats`、`provider_stats`
  - `umo_aliases`、`attachments`、`api_keys`、`dashboard_trusted_devices`
  - `chatui_projects`、`session_project_relations`
  - `command_configs`、`command_conflicts`、`webchat_threads`

- **异步 DAO** `AstrBotDatabase` 类
  - 基于 SQLAlchemy[asyncio] + aiosqlite
  - 完整 PRAGMA 调优套餐（WAL / busy_timeout / mmap_size 等）
  - `get_db()` 上下文管理器 + `_run_in_tx` 事务包装
  - 幂等 `_ensure_xxx_column` 列补齐

- **SharedPreferences 三层 KV API**
  - `global_get/put/remove` — 全局配置
  - `session_get/put/remove` — 按 UMO 隔离的会话配置
  - `plugin_get/put/remove/list` — 插件私有数据
  - `is_migration_done/mark_migration_done` — 迁移标记

- **自研幂等迁移框架**
  - `@register_migration` 装饰器
  - `run_migrations` 自动跳过已完成的迁移
  - 迁移状态存到 `preferences` 表自身

- **15 个公开 API**（通过 `@API` 装饰器暴露）
  - `astrdb.kv.{get,put,delete,list}`
  - `astrdb.conv.{create,get,list,update_content,delete}`
  - `astrdb.persona.{list,get}`
  - `astrdb.msg.{add,list}`
  - `astrdb.stats.{count,incr_platform}`

- **管理命令** `/adb`
  - `stats` — 显示各表行数
  - `tables` — 列出所有表名
  - `backup` — 手动备份
  - `export <table>` — 导出某张表前 100 行为 JSON

- **AstrBot 数据导入器**
  - `python -m importers.astrbot_importer --src ... --dst ...`
  - 支持从 AstrBot `data_v4.db` 一键迁移 14 张表

- **示例调用插件** `maibot-astrbot-db-demo`
  - 演示如何通过 `self.ctx.api.call(...)` 调用本插件

- **测试覆盖**
  - 27 个 pytest 测试（含 3 个端到端集成测试）
  - 全部通过

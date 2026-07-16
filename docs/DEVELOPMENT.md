# 开发者文档

本文档面向需要了解内部实现、调用 API 或二次开发的开发者。普通用户请看 [README](../README.md)。

## 架构概览

```
plugin.py                  插件入口，组合所有 Mixin
├── maikb/                 核心包
│   ├── __init__.py        全局单例 (init_db / get_db / sp)
│   ├── models.py          20 张 SQLModel 表定义
│   ├── database.py        异步 DAO (MaiKBDatabase)
│   ├── preferences.py     三层 KV API (SharedPreferences)
│   ├── interceptor.py     消息前缀拦截器 (HookHandler)
│   ├── injector.py        知识库自动注入器 (HookHandler)
│   ├── migrations/        幂等迁移机制
│   ├── kb/                知识库 RAG 模块
│   │   ├── chunker.py     Markdown 语义切分
│   │   ├── embedder.py    Embedding 抽象层 (maibot / openai / dummy)
│   │   ├── vector_store.py numpy 向量索引
│   │   ├── search.py      混合检索 + RRF 融合
│   │   ├── importer.py    批量导入 (增量 / 全量重建)
│   │   └── api.py         9 个 KB API + LLM Tool
│   └── webui/
│       └── server.py      FastAPI Web 管理界面
├── importers/
│   └── astrbot_importer.py  从 AstrBot data_v4.db 导入
└── tests/                 74 个测试 + maibot_sdk 桩
```

## Hook 处理器

### 消息前缀拦截器

| 属性 | 值 |
|---|---|
| Hook | `chat.receive.before_process` |
| 模式 | `BLOCKING` |
| 优先级 | `EARLY` |
| 作用 | 命中前缀的消息直接 abort，不记录、不回复、不进 A_memorix |

### 知识库自动注入器

| 属性 | 值 |
|---|---|
| Hook | `maisaka.replyer.before_model_request` |
| 模式 | `BLOCKING` |
| 优先级 | `NORMAL` |
| 作用 | 在 LLM 调用前自动检索知识库，将结果插入到最后一条 user 消息之前 |

注入位置选择"最后一条 user 消息之前"而非"消息列表末尾"，是因为 LLM 更倾向于关注对话结尾的内容——把 RAG 上下文放在末尾会干扰模型的直接回复。插在最后一条 user 消息之前，既能提供参考，又不污染对话尾部。

去重逻辑：检查最近 N 条消息（`dedup_lookback`），若 LLM 已调用 `knowledge_search` Tool 则跳过自动注入。

## 对外 API

其他插件通过 `self.ctx.api.call(...)` 调用。

### 数据库 / 对话 API (15 个)

| API | 描述 |
|---|---|
| `maikb.kv.get` | 读取 KV（支持 default） |
| `maikb.kv.put` | 写入 KV（upsert） |
| `maikb.kv.delete` | 删除 KV |
| `maikb.kv.list` | 列出某 scope 下所有 KV |
| `maikb.conv.create` | 创建对话 |
| `maikb.conv.get` | 按 ID 获取对话 |
| `maikb.conv.list` | 按 UMO 列出对话 |
| `maikb.conv.update_content` | 更新对话内容 |
| `maikb.conv.delete` | 删除对话 |
| `maikb.persona.list` | 列出人格 |
| `maikb.persona.get` | 获取人格详情 |
| `maikb.msg.add` | 追加消息历史 |
| `maikb.msg.list` | 列出消息历史 |
| `maikb.stats.count` | 统计表行数 |
| `maikb.stats.incr_platform` | 自增平台消息统计 |

### 知识库 API (9 个)

| API | 描述 |
|---|---|
| `maikb.kb.ingest_directory` | 批量扫描目录导入 |
| `maikb.kb.ingest_file` | 单文件导入 |
| `maikb.kb.list_files` | 列出已入库文件 |
| `maikb.kb.search` | 混合检索（向量 + BM25 + RRF） |
| `maikb.kb.search_vector` | 仅向量检索 |
| `maikb.kb.search_bm25` | 仅 BM25 检索 |
| `maikb.kb.delete_file` | 删除文件及其 chunks |
| `maikb.kb.stats` | 知识库统计 |
| `maikb.kb.reload_index` | 重载内存索引 |

### LLM Tool

`knowledge_search(query: str)` — LLM 可在对话中主动调用检索知识库。

## 数据库表清单

| 表名 | 用途 |
|---|---|
| `conversations` | LLM 对话历史（OpenAI 消息数组） |
| `preferences` | 万能 KV 表 (scope, scope_id, key, value:JSON) |
| `platform_message_history` | 平台消息历史 |
| `platform_sessions` | 平台会话 |
| `personas` / `persona_folders` | LLM 人格定义与文件夹 |
| `cron_jobs` | 定时任务 |
| `platform_stats` | 平台消息统计（按时间桶） |
| `provider_stats` | LLM Provider 调用统计（token 用量） |
| `umo_aliases` | UMO 字符串到友好名映射 |
| `attachments` | 文件附件元数据 |
| `api_keys` / `dashboard_trusted_devices` | API Key 与可信设备 |
| `chatui_projects` / `session_project_relations` | ChatUI 项目 |
| `command_configs` / `command_conflicts` | 命令注册与冲突 |
| `webchat_threads` | WebChat 线程 |
| `kb_files` | 知识库文件元数据 |
| `kb_chunks` | 知识库切片（含 embedding BLOB） |
| `kb_chunks_fts` | FTS5 虚拟表（trigram 分词） |

## 设计模式

### UMO 字符串

跨平台身份统一为 `platform:type:session_id` 字符串，不维护独立用户表。

```python
from maikb import build_umo, parse_umo
umo = build_umo("aiocqhttp", "GroupMessage", "123456789")
# → "aiocqhttp:GroupMessage:123456789"
```

### 万能 KV 表

一张 `preferences` 表覆盖全局配置、会话配置、插件数据、迁移标记：

```
scope: 'global' | 'umo' | 'plugin' | 'migration'
scope_id: 'global' | UMO | plugin_id
key: str
value: dict (JSON, 业务值放在 value["val"])
UNIQUE (scope, scope_id, key)
```

### 双 ID 设计

`ConversationV2` 同时有 `inner_conversation_id`（自增 int，内部索引）和 `conversation_id`（UUID 字符串，对外稳定）。

### SQLite PRAGMA 调优

```sql
PRAGMA journal_mode=WAL;       -- 多读单写不阻塞
PRAGMA busy_timeout=30000;     -- 30s 锁等待
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=20000;       -- ~80MB
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=134217728;    -- 128MB
```

### 幂等迁移

- 迁移函数用装饰器注册，幂等执行
- 迁移状态存到 `preferences` 表自身（无需单独 schema_migrations 表）
- 每次启动跑 `PRAGMA table_info` → 缺列就 `ALTER TABLE ADD COLUMN`，双保险

## 检索算法

### 向量检索

- 存储：SQLite BLOB 字段 (`numpy.float32.tobytes()`)
- 索引：启动时全部加载到内存（1 万 chunk × 1024 维 ≈ 40MB）
- 检索：numpy 矩阵乘法（cosine similarity），暴力但极快
- 归一化：查询向量和库向量都归一化，dot product 即 cosine

### BM25 检索

- 引擎：SQLite FTS5（虚拟表 `kb_chunks_fts`）
- 分词器：`trigram`（对中文友好，支持 ≥3 字符子串匹配）
- 短查询兜底（< 3 字符）：回退 `LIKE` 查真实表 `kb_chunks`（非 FTS 虚拟表，因为 FTS 虚拟表列不支持 LIKE 子串过滤）
- FTS 失败兜底：同样回退 `LIKE` 查真实表
- 排序：BM25 分数取负数转成越大越好

### RRF 融合

```
score(d) = sum( 1 / (k + rank_i(d)) )  for each retriever i
```

- `k = 60`（标准值）
- 只用排名不用原始分数，避免两路分数尺度不一致
- 同时被两路召回的文档得分更高

## 切分策略

按 Markdown 标题层级语义切分，不做字符数硬切：

| 维度 | A_memorix | 本插件 |
|---|---|---|
| 切分单位 | 1600 字符滑动窗口 | Markdown 标题章节 |
| 标题感知 | 仅识别 `#` 作场景分隔 | 完整保留 `#`/`##`/`###` 层级路径 |
| 段落边界 | 不感知，硬切 | 按双换行分段，累积到目标大小输出 |
| 重叠 | 400 字符重复存储 | 不重叠 |
| 关系抽取 | LLM 自由生成三元组 | 不抽取，直接存原文 |

每个 chunk 保留 `title_path`（如 `["蒙德", "第二幕", "法涅斯的诞生"]`），检索结果中返回，LLM 可引用来源章节。

## 关于"世界书"

世界书不做独立子模块。知识库的 markdown 语义切分已保留 `title_path`，每个 chunk 天然等价于一条世界书条目（有标题、有正文、有章节归属）。另起一套"条目 CRUD + 独立检索"会和知识库完全重叠。

世界书能力被吸收进知识库：

- 条目 = chunk（`heading` 是条目名，`content` 是条目内容）
- 章节归属 = `title_path`
- 分类 = `category`（检索时传 `category="某书"` 隔离多本世界书）
- 条目管理 = Web UI 的上传 / 删除 / 分类过滤

## 测试

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio
pytest -v
```

`tests/_sdk_stub/` 提供 `maibot_sdk` 最小化桩，`conftest.py` 优先用真实 SDK、缺失时自动回退。74 个测试覆盖：模型 / KV / 对话 / 迁移 / 统计 / KB 切分 / KB 检索 / 拦截器 / 注入器 / Web UI 端到端 / 集成测试。

## 从 AstrBot 导入

```bash
python -m importers.astrbot_importer \
    --src /path/to/AstrBot/data/data_v4.db \
    --dst /path/to/MaiBot/data/plugins/maibot-team.maikb/maikb.db
```

支持表：`platform_stats`、`provider_stats`、`conversations`、`persona_folders`、`personas`、`cron_jobs`、`preferences`、`platform_message_history`、`webchat_threads`、`platform_sessions`、`umo_aliases`、`attachments`、`command_configs`、`command_conflicts`。

字段对齐策略：源库可能缺本插件新增列，按交集导入；自增主键跳过避免冲突。

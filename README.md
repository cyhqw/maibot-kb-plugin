# MaiBot 知识库

MaiBot 的知识库 RAG 插件。把 Markdown / TXT 文档向量化，在对话中自动检索并注入相关内容，让 Bot 能回答世界观、剧情、设定等问题。

## 功能

- **Markdown 语义切分** — 按标题层级切分，保留章节路径，不做字符数硬切
- **混合检索** — 向量（numpy cosine）+ BM25（SQLite FTS5 trigram）+ RRF 融合
- **自动注入** — 对话中自动检索知识库，结果插入到最后一条 user 消息之前
- **LLM Tool** — 注册 `knowledge_search`，Bot 可主动检索
- **消息拦截** — 命中前缀（`/` `[` `#`）的消息不记录、不回复
- **Web 管理界面** — 统计面板 / 文件管理（上传/删除/分类过滤）/ 检索测试 / 配置编辑 / 暗色模式
- **SQLite 存储抽象** — 20 张表 + 异步 DAO + 三层 KV，可供其他插件调用

## 安装

```bash
# 1. 复制到 MaiBot plugins 目录
cp -r maikb /path/to/MaiBot/plugins/

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 MaiBot，插件自动初始化
```

## 快速开始

### 1. 放入知识库文件

把 `.md` / `.txt` 文件放到插件 data 目录下的 `knowledge_base/`：

```
data/plugins/maibot-team.maikb/
├── maikb.db                # 数据库（自动创建）
├── config.toml             # 配置
└── knowledge_base/         ← 放这里
    ├── 世界观设定.md
    ├── 角色档案.md
    └── ...
```

支持嵌套子目录，所有 `.md` / `.markdown` / `.txt` 文件会被自动扫描。

### 2. 配置 Embedding

编辑 `config.toml`，在 `[knowledge_base]` 段选一种 embedding 提供方：

```toml
[knowledge_base]
enabled = true
auto_ingest_on_start = true

# 推荐：用 MaiBot 自己的 embedding 服务
embedding_provider = "maibot"
embedding_model = "text-embedding-3-small"
embedding_dimension = 1536

# 或用 OpenAI 兼容接口
# embedding_provider = "openai"
# embedding_api_key = "sk-..."
# embedding_base_url = "https://api.openai.com/v1"

default_category = "genshin"  # 给文件打分类标签
```

### 3. 启动

启动后插件自动：创建数据库 → 加载已有向量 → 扫描目录增量导入新文件。之后 MaiBot 在对话中会自动检索知识库并注入相关内容。

## Web 管理界面

默认 `http://127.0.0.1:8765`，可在 `config.toml` 的 `[webui]` 段配置端口和 token。

功能：统计概览 / 文件管理（上传/删除/分类过滤/切片详情）/ 检索测试 / 配置热重载 / 明暗主题切换。

## 配置项

| 段 | 作用 |
|---|---|
| `[database]` | 数据库文件名、自动备份 |
| `[knowledge_base]` | 切分参数、embedding 配置、分类标签 |
| `[interceptor]` | 前缀拦截开关与列表 |
| `[injector]` | 自动注入阈值、top_k、去重 |
| `[webui]` | 监听地址、端口、token |

## 常见问题

**检索结果不相关** — 检查 `embedding_provider` 是否为 `maibot` 或 `openai`。`dummy` 模式只有哈希伪随机向量，没有语义能力。

**中文短查询（2 字以下）检索不到** — trigram 分词器要求 FTS5 查询 ≥ 3 字符。短查询会自动回退 `LIKE` 子串匹配，但仍建议多写几个字提高命中率。

**文件更新后没重新导入** — 增量导入基于 `file_hash`。确认文件内容确实改了（不只是 mtime），或调 `maikb.kb.ingest_directory` API 强制扫描。

## 开发者文档

API 列表、数据库表结构、检索算法细节、Hook 处理器、设计模式、测试方法等见 [开发者文档](docs/DEVELOPMENT.md)。

## 许可证

GPL-v3.0-or-later

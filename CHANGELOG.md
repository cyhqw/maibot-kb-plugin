# Changelog

## v1.0.0 — 首个正式发布

MaiBot 知识库插件，提供 Markdown 文档向量化、混合检索与自动注入。

### 功能

- **Markdown 语义切分** — 按标题层级切分，保留章节路径（`title_path`），不做字符数硬切
- **混合检索** — 向量（numpy cosine）+ BM25（SQLite FTS5 trigram）+ RRF 融合
- **自动注入** — 通过 `maisaka.replyer.before_model_request` Hook 在 LLM 调用前自动检索并注入
- **LLM Tool** — 注册 `knowledge_search`，Bot 可主动检索知识库
- **消息拦截** — 通过 `chat.receive.before_process` Hook 拦截前缀消息（`/` `[` `#`）
- **Web 管理界面** — 统计面板 / 文件管理（上传/删除/分类过滤/切片详情）/ 检索测试 / 配置热重载 / 暗色模式
- **SQLite 存储抽象** — 20 张表 + 异步 DAO + 三层 KV + 幂等迁移，24 个公开 API 供其他插件调用

### 技术细节

- 纯 numpy + SQLite，零外部向量数据库依赖
- FTS5 trigram 分词器对中文友好，短查询回退 LIKE 查真实表
- 增量导入基于 `file_hash`，跳过未变更文件
- 支持 MaiBot / OpenAI 兼容 / Dummy 三种 embedding 后端
- `category` 字段隔离多本世界书，无需独立子模块
- 测试桩 `tests/_sdk_stub/` 让无 SDK 环境也能跑测试（74 passed）

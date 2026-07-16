"""AstrBot 数据库移植插件入口

通过 MaiBot 插件 SDK 注册：
- on_load: 初始化数据库（建表 + PRAGMA + 迁移）
- on_unload: 关闭引擎
- @API astrdb.kv: 暴露 KV 三件套给其他插件
- @API astrdb.conversation: 暴露对话 CRUD
- @API astrdb.persona: 暴露人格 CRUD
- @API astrdb.message_history: 暴露消息历史
- @API astrdb.stats: 暴露统计 API
- @Command /adb: 管理命令（统计/备份/导出）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from maibot_sdk import API, Command, Field, MaiBotPlugin, PluginConfigBase

import astrdb
from astrdb import (
    AstrBotDatabase,
    build_umo,
    close_db,
    get_db,
    get_sp,
    init_db,
)
from astrdb.interceptor import InterceptorMixin
from astrdb.injector import InjectorMixin
from astrdb.kb.api import KbApiMixin, close_kb, init_kb, load_kb_index
from astrdb.memory.api import (
    MemoryApiMixin,
    close_memory,
    init_memory,
    init_memory_async,
    set_llm_fn as mem_set_llm_fn,
)
from astrdb.memory.decay_scheduler import DecayScheduler
from astrdb.webui import WebServer


# ----------------------------------------------------------------------
# 配置模型
# ----------------------------------------------------------------------

class DatabaseSectionConfig(PluginConfigBase):
    """数据库基础配置。"""

    __ui_label__: ClassVar[str] = "数据库"
    __ui_icon__: ClassVar[str] = "database"
    __ui_order__: ClassVar[int] = 0

    config_version: str = Field(default="1.0.0", description="配置版本号")
    enabled: bool = Field(default=True, description="是否启用数据库插件")
    db_filename: str = Field(
        default="astrbot.db",
        description="数据库文件名（保存在插件 data_dir 下）",
    )
    auto_backup_on_start: bool = Field(
        default=False,
        description="启动时是否自动备份一次数据库（保留最近 7 份）",
    )


class AdminSectionConfig(PluginConfigBase):
    """管理命令配置。"""

    __ui_label__: ClassVar[str] = "管理命令"
    __ui_icon__: ClassVar[str] = "settings"
    __ui_order__: ClassVar[int] = 1

    admin_users: List[str] = Field(
        default_factory=list,
        description="允许使用 /adb 命令的用户 ID 列表（platform:user_id 格式）",
    )


class KnowledgeBaseSectionConfig(PluginConfigBase):
    """知识库（RAG）配置。"""

    __ui_label__: ClassVar[str] = "知识库"
    __ui_icon__: ClassVar[str] = "book-open"
    __ui_order__: ClassVar[int] = 2

    enabled: bool = Field(default=True, description="是否启用知识库模块")
    knowledge_dir: str = Field(
        default="knowledge_base",
        description="知识库源文件目录（相对于插件 data_dir，存放 .md/.txt 文件）",
    )
    auto_ingest_on_start: bool = Field(
        default=True,
        description="启动时是否自动扫描 knowledge_dir 并增量导入新文件",
    )
    target_chars: int = Field(default=500, description="目标 chunk 字符数")
    max_chars: int = Field(default=1500, description="单 chunk 最大字符数")
    min_chars: int = Field(default=80, description="单 chunk 最小字符数（小于此值合并到上一个）")

    # Embedding 配置
    embedding_provider: str = Field(
        default="dummy",
        description="embedding 提供方：maibot（推荐，走 MaiBot llm.embed）/ openai（兼容接口）/ dummy（仅测试）",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="embedding 模型名（maibot/openai 模式用）",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="embedding 向量维度（必须与模型实际维度一致）",
    )
    embedding_api_key: str = Field(
        default="",
        description="openai 模式必填；maibot 模式留空",
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="openai 模式可选；可指向 DeepSeek/Moonshot/智谱等兼容接口",
    )
    embedding_batch_size: int = Field(default=16, description="embedding 批量大小")
    default_category: str = Field(
        default="",
        description="导入时给所有文件打上的默认 category 标签，留空则不打",
    )


class InterceptorSectionConfig(PluginConfigBase):
    """消息前缀拦截器配置。"""

    __ui_label__: ClassVar[str] = "消息拦截"
    __ui_icon__: ClassVar[str] = "shield"
    __ui_order__: ClassVar[int] = 3

    enabled: bool = Field(default=True, description="是否启用前缀拦截")
    prefixes: List[str] = Field(
        default_factory=lambda: ["/", "[", "#"],
        description="触发拦截的前缀字符列表；命中后消息不记录、不回复",
    )
    log_blocked: bool = Field(
        default=True,
        description="是否在日志中记录被拦截的消息（前 60 字预览）",
    )


class InjectorSectionConfig(PluginConfigBase):
    """自动召回 + 注入器配置。"""

    __ui_label__: ClassVar[str] = "自动注入"
    __ui_icon__: ClassVar[str] = "zap"
    __ui_order__: ClassVar[int] = 4

    enabled: bool = Field(
        default=True,
        description="是否启用自动召回；启用后在 LLM 调用前自动检索知识库并注入相关内容",
    )
    min_score: float = Field(
        default=0.01,
        description="RRF 融合分数阈值，低于此值不注入",
    )
    min_vector_score: float = Field(
        default=0.3,
        description="向量相似度阈值；同时低于此值且 BM25=0 的不注入",
    )
    top_k: int = Field(
        default=3,
        description="注入几条检索结果",
    )
    max_chars: int = Field(
        default=2000,
        description="注入文本最大字符数（超出会截断）",
    )
    dedup_lookback: int = Field(
        default=6,
        description="检查最近 N 条消息避免与 LLM 主动调 tool 重复",
    )
    skip_if_tool_called: bool = Field(
        default=True,
        description="LLM 已调过 knowledge_search tool 时跳过自动注入",
    )


class WebUISectionConfig(PluginConfigBase):
    """Web 管理界面配置。"""

    __ui_label__: ClassVar[str] = "Web UI"
    __ui_icon__: ClassVar[str] = "globe"
    __ui_order__: ClassVar[int] = 5

    enabled: bool = Field(default=True, description="是否启动 Web 管理界面")
    host: str = Field(
        default="127.0.0.1",
        description="监听地址；127.0.0.1 只本机访问，0.0.0.0 允许外部",
    )
    port: int = Field(
        default=8765,
        description="监听端口；避开 MaiBot WebUI 默认的 8001",
    )
    token: str = Field(
        default="",
        description="访问令牌；留空则无认证（仅本机调试用）",
    )


class MemorySectionConfig(PluginConfigBase):
    """记忆系统（LivingMemory 移植）配置。"""

    __ui_label__: ClassVar[str] = "记忆系统"
    __ui_icon__: ClassVar[str] = "brain"
    __ui_order__: ClassVar[int] = 6

    enabled: bool = Field(
        default=True,
        description="是否启用 MemoryAtom 记忆系统（移植自 LivingMemory）",
    )

    # 自动注入
    auto_inject: bool = Field(
        default=True,
        description="是否在 LLM 调用前自动检索记忆并注入到 prompt",
    )
    injection_mode: str = Field(
        default="extra_user_content",
        description="注入方式：extra_user_content / user_message_before / user_message_after",
    )
    injection_top_k: int = Field(default=3, description="注入几条记忆")
    injection_min_score: float = Field(default=0.1, description="注入最低分数")
    injection_max_chars: int = Field(default=2000, description="注入文本最大字符数")

    # 衰减调度
    decay_enabled: bool = Field(default=True, description="是否启用每日衰减调度")
    decay_rate: float = Field(default=0.01, description="每日重要性衰减率")
    decay_check_hour: int = Field(default=0, description="衰减执行小时（0-23）")
    decay_check_minute: int = Field(default=5, description="衰减执行分钟（0-59）")
    cleanup_days_threshold: int = Field(default=30, description="清理超过 N 天的低重要性记忆")
    cleanup_importance_threshold: float = Field(default=0.3, description="清理重要性低于此值的记忆")

    # 对话总结
    summary_trigger_rounds: int = Field(
        default=10,
        description="每 N 轮对话触发一次总结（生成 atoms）",
    )

    # LLM Tool
    enable_memory_search_tool: bool = Field(
        default=True,
        description="是否注册 memory_search LLM Tool",
    )
    enable_memory_memorize_tool: bool = Field(
        default=True,
        description="是否注册 memory_memorize LLM Tool",
    )


class AstrBotDbConfig(PluginConfigBase):
    """AstrBot 数据库移植插件配置。"""

    database: DatabaseSectionConfig = Field(default_factory=DatabaseSectionConfig)
    admin: AdminSectionConfig = Field(default_factory=AdminSectionConfig)
    knowledge_base: KnowledgeBaseSectionConfig = Field(default_factory=KnowledgeBaseSectionConfig)
    interceptor: InterceptorSectionConfig = Field(default_factory=InterceptorSectionConfig)
    injector: InjectorSectionConfig = Field(default_factory=InjectorSectionConfig)
    webui: WebUISectionConfig = Field(default_factory=WebUISectionConfig)
    memory: MemorySectionConfig = Field(default_factory=MemorySectionConfig)


# ----------------------------------------------------------------------
# 插件主类
# ----------------------------------------------------------------------

class AstrBotDbPlugin(MaiBotPlugin, KbApiMixin, InterceptorMixin, InjectorMixin, MemoryApiMixin):
    """AstrBot 数据库移植插件。"""

    config_model = AstrBotDbConfig

    # 保存最近一次初始化的 db 路径，用于诊断
    _db_path: Path | None = None
    _kb_dir: Path | None = None
    _web_server: WebServer | None = None
    _mem_decay_scheduler: DecayScheduler | None = None

    async def on_load(self) -> None:
        """插件加载：初始化数据库 + KB + 拦截器 + 注入器 + Memory + Web UI。"""

        cfg = self.config.database
        if not cfg.enabled:
            self.ctx.logger.warning("AstrBot DB 插件已被禁用（database.enabled=false）")
            return

        # 数据库路径：使用 MaiBot 分配给插件的 data_dir
        db_path = self.ctx.paths.data_dir / cfg.db_filename
        self._db_path = db_path

        self.ctx.logger.info(f"初始化 AstrBot 数据库: {db_path}")
        await init_db(str(db_path))

        # 可选：启动时自动备份
        if cfg.auto_backup_on_start:
            await self._auto_backup()

        # 测试连通性
        db = get_db()
        tables = await db.list_tables()
        self.ctx.logger.info(f"AstrBot 数据库就绪，共 {len(tables)} 张表")

        # 知识库初始化
        kb_cfg = self.config.knowledge_base
        if kb_cfg.enabled:
            await self._init_kb(kb_cfg)
        else:
            self.ctx.logger.info("知识库模块已禁用（knowledge_base.enabled=false）")

        # 拦截器（通过 @HookHandler 自动注册，这里只做日志）
        interceptor_cfg = self.config.interceptor
        if interceptor_cfg.enabled:
            self.ctx.logger.info(
                f"前缀拦截器已启用: prefixes={interceptor_cfg.prefixes}"
            )

        # 注入器（KB 注入）
        injector_cfg = self.config.injector
        if injector_cfg.enabled:
            self.ctx.logger.info(
                f"KB 自动召回+注入器已启用: top_k={injector_cfg.top_k} "
                f"min_score={injector_cfg.min_score}"
            )

        # Memory 模块（LivingMemory 移植）
        mem_cfg = self.config.memory
        if mem_cfg.enabled:
            await self._init_memory(mem_cfg)
        else:
            self.ctx.logger.info("Memory 模块已禁用（memory.enabled=false）")

        # Web UI
        webui_cfg = self.config.webui
        if webui_cfg.enabled:
            await self._start_web_ui(webui_cfg)

    async def _init_memory(self, mem_cfg: MemorySectionConfig) -> None:
        """初始化 Memory 模块。"""

        # 构造 LLM 调用函数（延迟注入）
        # MaiBot 模式下通过 ctx.llm.generate 调用
        llm_fn = None
        try:
            llm_fn = self._make_memory_llm_fn()
        except Exception as exc:
            self.ctx.logger.warning(
                f"无法构造 Memory LLM 函数，对话总结功能将不可用: {exc}"
            )

        init_memory(
            db=get_db(),
            llm_generate_fn=llm_fn,
            injector_enabled=mem_cfg.auto_inject,
            injector_mode=mem_cfg.injection_mode,
            injector_top_k=mem_cfg.injection_top_k,
            injector_min_score=mem_cfg.injection_min_score,
            injector_max_chars=mem_cfg.injection_max_chars,
            summary_trigger_rounds=mem_cfg.summary_trigger_rounds,
        )
        await init_memory_async(get_db())

        # 注册 Memory Hook（自动注入记忆到 prompt）
        # 通过 InjectorMixin 的 hook_auto_inject 已注册，这里不再重复
        # Memory 注入通过单独的 HookHandler 实现（待添加）

        # 启动衰减调度器
        if mem_cfg.decay_enabled:
            from astrdb.memory.api import _mem_atom_store, _mem_lifecycle
            if _mem_atom_store and _mem_lifecycle:
                self._mem_decay_scheduler = DecayScheduler(
                    _mem_atom_store,
                    _mem_lifecycle,
                    decay_rate=mem_cfg.decay_rate,
                    check_hour=mem_cfg.decay_check_hour,
                    check_minute=mem_cfg.decay_check_minute,
                    cleanup_days_threshold=mem_cfg.cleanup_days_threshold,
                    cleanup_importance_threshold=mem_cfg.cleanup_importance_threshold,
                )
                self._mem_decay_scheduler.start()
                self.ctx.logger.info(
                    f"Memory 衰减调度器已启动: 每日 {mem_cfg.decay_check_hour:02d}:"
                    f"{mem_cfg.decay_check_minute:02d} 执行"
                )

        self.ctx.logger.info("Memory 模块（LivingMemory 移植）已就绪")

    def _make_memory_llm_fn(self):
        """构造 MemoryProcessor 用的 LLM 调用函数。

        通过 MaiBot 的 ctx.llm.generate 调用，返回 async (prompt, system_prompt) -> str
        """

        async def _llm_generate(*, prompt: str, system_prompt: str = "") -> str:
            try:
                result = await self.ctx.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                if isinstance(result, str):
                    return result
                if isinstance(result, dict):
                    return str(result.get("content") or result.get("text") or "")
                return str(result)
            except Exception as exc:
                self.ctx.logger.warning(f"Memory LLM 调用失败: {exc}")
                return ""

        return _llm_generate

    async def _start_web_ui(self, webui_cfg: WebUISectionConfig) -> None:
        """启动 Web 管理 server。"""

        self._web_server = WebServer(
            plugin=self,
            host=webui_cfg.host,
            port=webui_cfg.port,
            token=webui_cfg.token,
        )
        try:
            await self._web_server.start()
            self.ctx.logger.info(
                f"Web UI 已启动: http://{webui_cfg.host}:{webui_cfg.port}"
            )
        except Exception as exc:
            self.ctx.logger.error(f"Web UI 启动失败: {exc}", exc_info=True)
            self._web_server = None

    async def _init_kb(self, kb_cfg: KnowledgeBaseSectionConfig) -> None:
        """初始化知识库模块。"""

        kb_dir = self.ctx.paths.data_dir / kb_cfg.knowledge_dir
        kb_dir.mkdir(parents=True, exist_ok=True)
        self._kb_dir = kb_dir

        # 构造 embedding 配置
        embedding_config = {
            "provider": kb_cfg.embedding_provider,
            "model": kb_cfg.embedding_model,
            "dimension": kb_cfg.embedding_dimension,
            "api_key": kb_cfg.embedding_api_key,
            "base_url": kb_cfg.embedding_base_url,
            "batch_size": kb_cfg.embedding_batch_size,
            "default_category": kb_cfg.default_category or None,
        }

        # MaiBot 模式下传入 embed 函数
        maibot_embed_fn = None
        if kb_cfg.embedding_provider == "maibot":
            try:
                maibot_embed_fn = self.ctx.llm.embed
                self.ctx.logger.info("使用 MaiBot LLMCapability.embed 作为 embedding 服务")
            except AttributeError:
                self.ctx.logger.warning(
                    "无法获取 self.ctx.llm.embed，回退到 dummy embedder（仅测试用）"
                )

        init_kb(
            knowledge_dir=kb_dir,
            embedding_config=embedding_config,
            maibot_embed_fn=maibot_embed_fn,
        )

        # 加载已有向量到内存
        loaded = await load_kb_index()
        self.ctx.logger.info(f"已从数据库加载 {loaded} 个向量到内存索引")

        # 自动增量导入
        if kb_cfg.auto_ingest_on_start:
            self.ctx.logger.info(f"开始扫描知识库目录: {kb_dir}")
            # 复用 importer（已通过 init_kb 创建）
            from astrdb.kb.api import _kb_importer
            if _kb_importer is not None:
                result = await _kb_importer.ingest_directory()
                self.ctx.logger.info(
                    f"知识库扫描完成: scanned={result.scanned} new={result.new} "
                    f"updated={result.updated} unchanged={result.unchanged} "
                    f"failed={result.failed} chunks={result.chunks}"
                )
                if result.failures:
                    for fp, err in result.failures[:5]:
                        self.ctx.logger.error(f"  导入失败 {fp}: {err}")

    async def on_unload(self) -> None:
        """插件卸载：关闭 Web UI + Memory + KB + 数据库。"""

        if self._mem_decay_scheduler is not None:
            await self._mem_decay_scheduler.stop()
            self._mem_decay_scheduler = None
        close_memory()
        if self._web_server is not None:
            await self._web_server.stop()
            self._web_server = None
        close_kb()
        await close_db()
        self.ctx.logger.info("AstrBot 数据库、KB、Memory 与 Web UI 已关闭")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        """配置热更新：仅记录日志，不重启数据库。"""

        self.ctx.logger.info(f"配置更新（scope={scope}, version={version}）")
        # 数据库文件路径变更需要手动重启插件
        if scope == "self":
            new_filename = (
                config_data.get("database", {}).get("db_filename", "astrbot.db")
            )
            if self._db_path and self._db_path.name != new_filename:
                self.ctx.logger.warning(
                    f"数据库文件名变更为 {new_filename}，需重新加载插件才会生效"
                )

    # ==================================================================
    # 内部工具
    # ==================================================================

    async def _auto_backup(self) -> None:
        """启动时备份，保留最近 7 份。"""

        if not self._db_path or not self._db_path.exists():
            return

        import shutil
        import time

        backup_dir = self.ctx.paths.data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"astrbot_{ts}.db"
        shutil.copy2(self._db_path, backup_path)
        self.ctx.logger.info(f"已备份数据库到 {backup_path}")

        # 保留最近 7 份
        backups = sorted(backup_dir.glob("astrbot_*.db"))
        for old in backups[:-7]:
            old.unlink()
            self.ctx.logger.info(f"清理旧备份: {old.name}")

    def _is_admin(self, platform: str, user_id: str) -> bool:
        """检查用户是否在 admin 白名单中。"""

        admin_list = self.config.admin.admin_users or []
        scoped = f"{platform}:{user_id}"
        return scoped in admin_list

    # ==================================================================
    # 对外 API（其他插件通过 self.ctx.api.call('astrdb.kv', ...) 调用）
    # ==================================================================

    # ----- KV API -----

    @API("astrdb.kv.get", description="读取 KV 值", version="1", public=True)
    async def api_kv_get(
        self,
        scope: str,
        scope_id: str,
        key: str,
        default: Any = None,
        **_: Any,
    ) -> Any:
        """读取 KV 值。

        Args:
            scope: "global" / "umo" / "plugin"
            scope_id: 全局为 "global"，UMO scope 为 UMO 字符串，plugin scope 为 plugin_id
            key: 键名
            default: 默认值

        Returns:
            Any: 值（不存在时返回 default）
        """

        sp = get_sp()
        return await sp.get_async(scope, scope_id, key, default)

    @API("astrdb.kv.put", description="写入 KV 值", version="1", public=True)
    async def api_kv_put(
        self,
        scope: str,
        scope_id: str,
        key: str,
        value: Any,
        **_: Any,
    ) -> dict[str, Any]:
        """写入 KV 值（upsert）。"""

        sp = get_sp()
        await sp.put_async(scope, scope_id, key, value)
        return {"success": True, "scope": scope, "scope_id": scope_id, "key": key}

    @API("astrdb.kv.delete", description="删除 KV 值", version="1", public=True)
    async def api_kv_delete(
        self, scope: str, scope_id: str, key: str, **_: Any
    ) -> dict[str, Any]:
        """删除 KV 值。"""

        sp = get_sp()
        deleted = await sp.remove_async(scope, scope_id, key)
        return {"success": deleted, "existed": deleted}

    @API("astrdb.kv.list", description="列出某 scope 下所有 KV", version="1", public=True)
    async def api_kv_list(
        self,
        scope: str,
        scope_id: str,
        key_prefix: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        """列出某 scope+scope_id 下所有 KV（支持前缀过滤）。"""

        sp = get_sp()
        items = await sp.list_async(scope, scope_id, key_prefix)
        return {"items": items, "count": len(items)}

    # ----- Conversation API -----

    @API("astrdb.conv.create", description="创建对话", version="1", public=True)
    async def api_conv_create(
        self,
        platform: str,
        message_type: str,
        session_id: str,
        platform_id: str | None = None,
        title: str | None = None,
        persona_id: str | None = None,
        content: list[Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """创建对话。

        Args:
            platform: 平台名（如 aiocqhttp）
            message_type: 消息类型（如 GroupMessage / FriendMessage）
            session_id: 会话 ID
            platform_id: 平台 ID（默认等于 platform）
        """

        umo = build_umo(platform, message_type, session_id)
        db = get_db()
        conv = await db.create_conversation(
            user_id=umo,
            platform_id=platform_id or platform,
            content=content,
            title=title,
            persona_id=persona_id,
        )
        return {
            "conversation_id": conv.conversation_id,
            "user_id": conv.user_id,
            "platform_id": conv.platform_id,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
        }

    @API("astrdb.conv.get", description="按 ID 获取对话", version="1", public=True)
    async def api_conv_get(self, conversation_id: str, **_: Any) -> dict[str, Any] | None:
        db = get_db()
        conv = await db.get_conversation_by_id(conversation_id)
        if conv is None:
            return None
        return _conv_to_dict(conv)

    @API("astrdb.conv.list", description="按 UMO 列出对话", version="1", public=True)
    async def api_conv_list(
        self,
        platform: str,
        message_type: str,
        session_id: str,
        limit: int = 50,
        **_: Any,
    ) -> dict[str, Any]:
        umo = build_umo(platform, message_type, session_id)
        db = get_db()
        convs = await db.get_conversations_by_user(umo, limit=limit)
        return {
            "items": [_conv_to_dict(c) for c in convs],
            "count": len(convs),
        }

    @API("astrdb.conv.update_content", description="更新对话内容", version="1", public=True)
    async def api_conv_update_content(
        self,
        conversation_id: str,
        content: list[Any],
        token_usage_delta: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        db = get_db()
        ok = await db.update_conversation_content(
            conversation_id, content, token_usage_delta
        )
        return {"success": ok}

    @API("astrdb.conv.delete", description="删除对话", version="1", public=True)
    async def api_conv_delete(self, conversation_id: str, **_: Any) -> dict[str, Any]:
        db = get_db()
        ok = await db.delete_conversation(conversation_id)
        return {"success": ok}

    # ----- Persona API -----

    @API("astrdb.persona.list", description="列出人格", version="1", public=True)
    async def api_persona_list(
        self, folder_id: str | None = None, **_: Any
    ) -> dict[str, Any]:
        db = get_db()
        personas = await db.list_personas(folder_id=folder_id)
        return {
            "items": [
                {
                    "persona_id": p.persona_id,
                    "name": p.name,
                    "folder_id": p.folder_id,
                    "is_default": p.is_default,
                    "sort_order": p.sort_order,
                }
                for p in personas
            ],
            "count": len(personas),
        }

    @API("astrdb.persona.get", description="获取人格详情", version="1", public=True)
    async def api_persona_get(self, persona_id: str, **_: Any) -> dict[str, Any] | None:
        db = get_db()
        p = await db.get_persona(persona_id)
        if p is None:
            return None
        return {
            "persona_id": p.persona_id,
            "name": p.name,
            "system_prompt": p.system_prompt,
            "begin_dialogs": p.begin_dialogs,
            "tools": p.tools,
            "skills": p.skills,
            "folder_id": p.folder_id,
            "is_default": p.is_default,
        }

    # ----- Message History API -----

    @API("astrdb.msg.add", description="追加消息历史", version="1", public=True)
    async def api_msg_add(
        self,
        platform: str,
        message_type: str,
        session_id: str,
        content: dict[str, Any],
        sender_id: str | None = None,
        sender_name: str | None = None,
        llm_checkpoint_id: str | None = None,
        platform_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        umo = build_umo(platform, message_type, session_id)
        db = get_db()
        rec = await db.add_message_history(
            platform_id=platform_id or platform,
            user_id=umo,
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            llm_checkpoint_id=llm_checkpoint_id,
        )
        return {"id": rec.id, "user_id": rec.user_id}

    @API("astrdb.msg.list", description="列出消息历史", version="1", public=True)
    async def api_msg_list(
        self,
        platform: str,
        message_type: str,
        session_id: str,
        limit: int = 50,
        before_id: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        umo = build_umo(platform, message_type, session_id)
        db = get_db()
        records = await db.get_message_history(umo, limit=limit, before_id=before_id)
        return {
            "items": [
                {
                    "id": r.id,
                    "sender_id": r.sender_id,
                    "sender_name": r.sender_name,
                    "content": r.content,
                    "llm_checkpoint_id": r.llm_checkpoint_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ],
            "count": len(records),
        }

    # ----- Stats API -----

    @API("astrdb.stats.count", description="统计表行数", version="1", public=True)
    async def api_stats_count(self, table_name: str, **_: Any) -> dict[str, Any]:
        db = get_db()
        try:
            count = await db.count_rows(table_name)
            return {"table": table_name, "count": count}
        except Exception as exc:
            return {"table": table_name, "error": str(exc)}

    @API("astrdb.stats.incr_platform", description="自增平台消息统计", version="1", public=True)
    async def api_stats_incr_platform(
        self,
        timestamp: int,
        platform_id: str,
        platform_type: str,
        count: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        db = get_db()
        await db.incr_platform_stat(timestamp, platform_id, platform_type, count)
        return {"success": True}

    # ==================================================================
    # 管理命令 /adb
    # ==================================================================

    @Command(
        "astrdb_admin",
        description="AstrBot 数据库管理命令",
        pattern=r"(?P<cmd>^/adb(?:\s+.+)?\s*$)",
    )
    async def handle_adb(
        self,
        stream_id: str = "",
        platform: str = "",
        user_id: str = "",
        matched_groups: dict | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str, bool]:
        """处理 /adb 命令。

        用法：
            /adb                  显示帮助
            /adb stats            显示各表行数
            /adb tables           列出所有表
            /adb backup           手动备份
            /adb export <table>   导出表为 JSON
        """

        # 权限检查
        if not self._is_admin(platform, user_id):
            await self.ctx.send.text("你没有权限使用 /adb 命令", stream_id)
            return False, "无权限", True

        if not stream_id:
            return False, "无法获取 stream_id", True

        cmd = (matched_groups or {}).get("cmd", "").strip()
        parts = cmd.split() if cmd else ["/adb"]
        n = len(parts)

        if n == 1:
            await self.ctx.send.text(_ADB_HELP, stream_id)
            return True, "已发送帮助", True

        sub = parts[1]
        if sub == "stats":
            await self._cmd_stats(stream_id)
        elif sub == "tables":
            await self._cmd_tables(stream_id)
        elif sub == "backup":
            await self._cmd_backup(stream_id)
        elif sub == "export" and n >= 3:
            await self._cmd_export(parts[2], stream_id)
        else:
            await self.ctx.send.text(_ADB_HELP, stream_id)
        return True, "命令完成", True

    async def _cmd_stats(self, stream_id: str) -> None:
        db = get_db()
        tables = await db.list_tables()
        lines = ["【AstrBot 数据库统计】", ""]
        for t in tables:
            try:
                c = await db.count_rows(t)
                lines.append(f"  {t}: {c}")
            except Exception as exc:
                lines.append(f"  {t}: <error: {exc}>")
        await self.ctx.send.text("\n".join(lines), stream_id)

    async def _cmd_tables(self, stream_id: str) -> None:
        db = get_db()
        tables = await db.list_tables()
        await self.ctx.send.text(
            f"共 {len(tables)} 张表:\n" + "\n".join(f"  - {t}" for t in tables),
            stream_id,
        )

    async def _cmd_backup(self, stream_id: str) -> None:
        if not self._db_path or not self._db_path.exists():
            await self.ctx.send.text("数据库文件不存在", stream_id)
            return
        import shutil
        import time
        backup_dir = self.ctx.paths.data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"astrbot_manual_{ts}.db"
        shutil.copy2(self._db_path, backup_path)
        await self.ctx.send.text(f"已备份到 {backup_path.name}", stream_id)

    async def _cmd_export(self, table_name: str, stream_id: str) -> None:
        """简单导出某张表为 JSON（限 100 行）。"""

        import json
        from sqlalchemy import text
        db = get_db()
        async with db.get_db() as session:
            result = await session.execute(
                text(f"SELECT * FROM {table_name} LIMIT 100")
            )
            rows = result.fetchall()

        if not rows:
            await self.ctx.send.text(f"表 {table_name} 为空或不存在", stream_id)
            return

        cols = list(rows[0]._mapping.keys())
        data = [dict(zip(cols, row)) for row in rows]
        # 序列化 datetime
        for row in data:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()

        export_path = self.ctx.paths.data_dir / f"export_{table_name}.json"
        export_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        await self.ctx.send.text(
            f"已导出 {len(data)} 行到 {export_path.name}", stream_id
        )


# ----------------------------------------------------------------------
# 工具函数与常量
# ----------------------------------------------------------------------

_ADB_HELP = (
    "AstrBot 数据库管理命令\n"
    "/adb           显示此帮助\n"
    "/adb stats     显示各表行数\n"
    "/adb tables    列出所有表名\n"
    "/adb backup    手动备份数据库\n"
    "/adb export <table>  导出某张表前 100 行为 JSON"
)


def _conv_to_dict(conv) -> dict[str, Any]:
    return {
        "conversation_id": conv.conversation_id,
        "user_id": conv.user_id,
        "platform_id": conv.platform_id,
        "title": conv.title,
        "persona_id": conv.persona_id,
        "token_usage": conv.token_usage,
        "content": conv.content,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


# ----------------------------------------------------------------------
# 插件工厂函数（MaiBot SDK 入口）
# ----------------------------------------------------------------------

def create_plugin() -> AstrBotDbPlugin:
    """MaiBot Runner 调用的工厂函数。"""

    return AstrBotDbPlugin()

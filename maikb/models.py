"""maikb.models

移植自 AstrBot v4 的 SQLModel 表定义。

完整复刻 AstrBot `astrbot/core/db/po.py` 的 17 张表 + 1 个 TimestampMixin，
保留 AstrBot 的设计精髓：

- UMO 字符串（platform:type:session_id）作为跨平台身份
- 万能 KV 表 preferences（scope, scope_id, key, value:JSON）
- 双 ID 设计（自增 int + 业务 UUID）
- TimestampMixin 自动维护时间戳

注意：本文件所有 SQLModel 类都注册在同一个 SQLModel.metadata 中，
由 maikb.database.MaiKBDatabase.initialize() 在启动时统一 create_all。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, LargeBinary, Text, UniqueConstraint
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    """统一的 UTC 时间生成器，避免测试时无法 freeze 时间。"""

    return datetime.now(timezone.utc)


def _uuid_str() -> str:
    """生成 UUID 字符串（用于业务 ID 列）。"""

    return str(uuid.uuid4())


class TimestampMixin(SQLModel):
    """自动维护 created_at / updated_at 时间戳。

    继承此 Mixin 的表会自动获得两个字段：
    - created_at: 创建时间，默认 UTC now
    - updated_at: 更新时间，每次 UPDATE 时由 SQLAlchemy 自动刷新
    """

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column_kwargs={"onupdate": _utcnow},
    )


# ============================================================
# 1. 平台统计表 platform_stats
# ============================================================

class PlatformStat(TimestampMixin, SQLModel, table=True):
    """平台消息统计（按时间分桶）。"""

    __tablename__ = "platform_stats"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int = Field(index=True, description="时间桶起点（unix 秒）")
    platform_id: str = Field(index=True, description="平台 ID")
    platform_type: str = Field(description="平台类型（aiocqhttp / webchat / ...）")
    count: int = Field(default=0, description="该时段消息数")

    __table_args__ = (
        UniqueConstraint("timestamp", "platform_id", "platform_type", name="uix_platform_stats_bucket"),
    )


# ============================================================
# 2. LLM Provider 调用统计 provider_stats
# ============================================================

class ProviderStat(TimestampMixin, SQLModel, table=True):
    """LLM Provider 调用记录，包含 token 用量与耗时。"""

    __tablename__ = "provider_stats"

    id: Optional[int] = Field(default=None, primary_key=True)
    umo: str = Field(index=True, description="统一消息来源字符串")
    conversation_id: Optional[str] = Field(default=None, index=True)
    provider_id: str = Field(index=True, description="Provider 配置 ID")
    provider_model: str = Field(index=True, description="模型名")
    status: str = Field(index=True, description="success / failed / cancelled")
    agent_type: str = Field(default="unknown", index=True)
    request_type: Optional[str] = Field(default=None)
    time_cost: float = Field(default=0.0, description="耗时秒数")
    timestamp: int = Field(index=True)
    token_input: int = Field(default=0)
    token_input_other: int = Field(default=0)
    token_input_cached: int = Field(default=0)
    token_output: int = Field(default=0)
    extra: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)


# ============================================================
# 3. 会话表 conversations
# ============================================================

class ConversationV2(TimestampMixin, SQLModel, table=True):
    """对话表（移植自 AstrBot ConversationV2）。

    采用双 ID 设计：
    - inner_conversation_id: 自增 int，便于内部索引
    - conversation_id: UUID 字符串，对外稳定，可暴露给客户端
    """

    __tablename__ = "conversations"

    inner_conversation_id: Optional[int] = Field(
        default=None, primary_key=True, sa_column_kwargs={"autoincrement": True}
    )
    conversation_id: str = Field(
        max_length=36, unique=True, default_factory=_uuid_str, index=True
    )
    platform_id: str = Field(index=True)
    user_id: str = Field(index=True, description="UMO 字符串")
    content: Optional[list[Any]] = Field(default=None, sa_type=JSON, description="OpenAI 消息数组")
    title: Optional[str] = Field(default=None)
    persona_id: Optional[str] = Field(default=None, index=True)
    token_usage: int = Field(default=0)


# ============================================================
# 4. 人格文件夹 persona_folders
# ============================================================

class PersonaFolder(TimestampMixin, SQLModel, table=True):
    """人格文件夹（树形结构，parent_id 自引用）。"""

    __tablename__ = "persona_folders"

    id: Optional[int] = Field(default=None, primary_key=True)
    folder_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    name: str = Field(max_length=128)
    parent_id: Optional[str] = Field(default=None, index=True, description="父文件夹 folder_id；根为 None")
    sort_order: int = Field(default=0)


# ============================================================
# 5. 人格表 personas
# ============================================================

class Persona(TimestampMixin, SQLModel, table=True):
    """LLM 人格定义。"""

    __tablename__ = "personas"

    id: Optional[int] = Field(default=None, primary_key=True)
    persona_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    name: str = Field(max_length=128)
    system_prompt: str = Field(default="", sa_type=Text)
    begin_dialogs: list[Any] = Field(default_factory=list, sa_type=JSON)
    tools: list[str] = Field(default_factory=list, sa_type=JSON)
    skills: list[Any] = Field(default_factory=list, sa_type=JSON)
    folder_id: Optional[str] = Field(default=None, index=True)
    sort_order: int = Field(default=0)
    is_default: bool = Field(default=False)
    custom_error_message: Optional[str] = Field(default=None)


# ============================================================
# 6. 定时任务 cron_jobs
# ============================================================

class CronJob(TimestampMixin, SQLModel, table=True):
    """定时任务（移植自 AstrBot CronJob）。"""

    __tablename__ = "cron_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    name: str = Field(max_length=128)
    cron_expression: str = Field(max_length=128)
    timezone: str = Field(default="Asia/Shanghai")
    payload: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    enabled: bool = Field(default=True)
    persistent: bool = Field(default=True)
    run_once: bool = Field(default=False, description="是否只执行一次")
    status: str = Field(default="pending", index=True, description="pending / running / done / failed")
    last_run_at: Optional[int] = Field(default=None)
    next_run_time: Optional[int] = Field(default=None, index=True)
    last_error: Optional[str] = Field(default=None)


# ============================================================
# 7. ★ 万能 KV 表 preferences（AstrBot 的核心设计）
# ============================================================

class Preference(TimestampMixin, SQLModel, table=True):
    """通用偏好/配置 KV 表。

    复合 UNIQUE 约束 (scope, scope_id, key) 保证原子 upsert。

    - scope='global', scope_id='global'  → 全局配置
    - scope='umo',     scope_id=<UMO>    → 按会话配置
    - scope='plugin',  scope_id=plugin_id→ 插件私有数据
    - scope='migration', scope_id='global' → 迁移完成标记

    value 是 JSON dict，业务值放在 value["val"] 中。
    """

    __tablename__ = "preferences"

    id: Optional[int] = Field(default=None, primary_key=True)
    scope: str = Field(max_length=32, index=True)
    scope_id: str = Field(max_length=128, index=True)
    key: str = Field(max_length=128, index=True)
    value: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)

    __table_args__ = (
        UniqueConstraint("scope", "scope_id", "key", name="uix_preference_scope_scope_id_key"),
    )


# ============================================================
# 8. 平台消息历史 platform_message_history
# ============================================================

class PlatformMessageHistory(TimestampMixin, SQLModel, table=True):
    """平台消息历史（与 MaiBot 自己的 mai_messages 表不冲突，这里存 LLM 视角的消息）。"""

    __tablename__ = "platform_message_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    platform_id: str = Field(index=True)
    user_id: str = Field(index=True, description="UMO 字符串")
    sender_id: Optional[str] = Field(default=None, index=True)
    sender_name: Optional[str] = Field(default=None)
    content: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    llm_checkpoint_id: Optional[str] = Field(default=None, index=True, description="关联的 conversation_id")


# ============================================================
# 9. WebChat 线程 webchat_threads
# ============================================================

class WebChatThread(TimestampMixin, SQLModel, table=True):
    """WebChat 线程表。"""

    __tablename__ = "webchat_threads"

    id: Optional[int] = Field(default=None, primary_key=True)
    thread_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    creator: str = Field(index=True)
    parent_session_id: Optional[str] = Field(default=None, index=True)
    parent_message_id: Optional[str] = Field(default=None, index=True)
    base_checkpoint_id: Optional[str] = Field(default=None, index=True)


# ============================================================
# 10. 平台会话 platform_sessions
# ============================================================

class PlatformSession(TimestampMixin, SQLModel, table=True):
    """平台会话表（一个 UMO 对应一个 session）。"""

    __tablename__ = "platform_sessions"

    inner_id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    platform_id: str = Field(default="webchat", index=True)
    creator: str = Field(index=True, description="UMO 字符串")
    display_name: Optional[str] = Field(default=None)
    is_group: bool = Field(default=False)


# ============================================================
# 11. UMO 别名 umo_aliases
# ============================================================

class UmoAlias(TimestampMixin, SQLModel, table=True):
    """UMO 字符串到友好名称的映射。"""

    __tablename__ = "umo_aliases"

    id: Optional[int] = Field(default=None, primary_key=True)
    umo: str = Field(unique=True, index=True, description="UMO 字符串")
    creator_sender_id: Optional[str] = Field(default=None, index=True)
    auto_name: Optional[str] = Field(default=None)
    user_alias: Optional[str] = Field(default=None, description="用户自定义别名")


# ============================================================
# 12. 附件 attachments
# ============================================================

class Attachment(TimestampMixin, SQLModel, table=True):
    """文件附件索引（实际文件存磁盘，这里存元数据）。"""

    __tablename__ = "attachments"

    inner_attachment_id: Optional[int] = Field(default=None, primary_key=True)
    attachment_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    path: str = Field(max_length=1024)
    type: str = Field(max_length=64, description="image / audio / file / video")
    mime_type: Optional[str] = Field(default=None)
    size: int = Field(default=0)
    sha256: Optional[str] = Field(default=None, index=True)


# ============================================================
# 13. API Key api_keys
# ============================================================

class ApiKey(TimestampMixin, SQLModel, table=True):
    """API Key 表（哈希存储，不存明文）。"""

    __tablename__ = "api_keys"

    inner_id: Optional[int] = Field(default=None, primary_key=True)
    key_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str)
    key_hash: str = Field(unique=True, index=True, description="SHA256(key)")
    key_prefix: str = Field(max_length=16, description="明文前 8 位，便于识别")
    scopes: list[str] = Field(default_factory=list, sa_type=JSON)
    created_by: Optional[str] = Field(default=None)
    last_used_at: Optional[int] = Field(default=None)
    expires_at: Optional[int] = Field(default=None, index=True)
    revoked_at: Optional[int] = Field(default=None)


# ============================================================
# 14. Dashboard 可信设备 dashboard_trusted_devices
# ============================================================

class DashboardTrustedDevice(TimestampMixin, SQLModel, table=True):
    """Dashboard 二次验证可信设备。"""

    __tablename__ = "dashboard_trusted_devices"

    id: Optional[int] = Field(default=None, primary_key=True)
    token_hash: str = Field(index=True, unique=True)
    totp_secret_hash: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    expires_at: int = Field(index=True)


# ============================================================
# 15. ChatUI 项目 chatui_projects
# ============================================================

class ChatUIProject(TimestampMixin, SQLModel, table=True):
    """ChatUI 项目表。"""

    __tablename__ = "chatui_projects"

    inner_id: Optional[int] = Field(default=None, primary_key=True)
    project_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    creator: str = Field(index=True)
    emoji: str = Field(default="📁", max_length=8)
    title: str = Field(max_length=128)
    description: Optional[str] = Field(default=None)
    workspace_type: str = Field(default="default", max_length=32)
    workspace_path: Optional[str] = Field(default=None, max_length=1024)


# ============================================================
# 16. 会话-项目关联 session_project_relations
# ============================================================

class SessionProjectRelation(TimestampMixin, SQLModel, table=True):
    """会话与 ChatUI 项目的关联表。"""

    __tablename__ = "session_project_relations"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(unique=True, index=True)
    project_id: str = Field(index=True)


# ============================================================
# 17. 命令配置 command_configs（业务字符串主键）
# ============================================================

class CommandConfig(TimestampMixin, SQLModel, table=True):
    """命令配置表（直接用 handler_full_name 作主键）。"""

    __tablename__ = "command_configs"

    handler_full_name: str = Field(primary_key=True, max_length=512)
    plugin_name: Optional[str] = Field(default=None, index=True)
    module_path: Optional[str] = Field(default=None)
    original_command: str = Field(max_length=128)
    resolved_command: str = Field(max_length=128)
    enabled: bool = Field(default=True)
    conflict_key: Optional[str] = Field(default=None, index=True)
    resolution_strategy: str = Field(default="rename", max_length=32)
    note: Optional[str] = Field(default=None, sa_type=Text)
    extra_data: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    auto_managed: bool = Field(default=False)


# ============================================================
# 18. 命令冲突 command_conflicts
# ============================================================

class CommandConflict(TimestampMixin, SQLModel, table=True):
    """命令冲突记录。"""

    __tablename__ = "command_conflicts"

    id: Optional[int] = Field(default=None, primary_key=True)
    conflict_key: str = Field(max_length=128, index=True)
    handler_full_name: str = Field(max_length=512, index=True)
    status: str = Field(default="unresolved", index=True)
    resolution: Optional[str] = Field(default=None)
    resolved_command: Optional[str] = Field(default=None)

    __table_args__ = (
        UniqueConstraint("conflict_key", "handler_full_name", name="uix_command_conflict_key_handler"),
    )


# ============================================================
# 工具函数
# ============================================================

def build_umo(platform: str, message_type: str, session_id: str) -> str:
    """构造 UMO（Unified Message Origin）字符串。

    AstrBot 的核心设计：用单一字符串表达跨平台身份。
    格式：platform_name:message_type:session_id

    示例：
        >>> build_umo("aiocqhttp", "GroupMessage", "123456789")
        'aiocqhttp:GroupMessage:123456789'
        >>> build_umo("webchat", "FriendMessage", "webchat!astrbot!user123")
        'webchat:FriendMessage:webchat!astrbot!user123'
    """

    return f"{platform}:{message_type}:{session_id}"


def parse_umo(umo: str) -> tuple[str, str, str]:
    """解析 UMO 字符串为 (platform, message_type, session_id) 三元组。"""

    parts = umo.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"非法 UMO 字符串（应为 platform:type:session_id 格式）: {umo!r}")
    return parts[0], parts[1], parts[2]


# ============================================================
# 知识库表（KB） — 新增，用于 RAG
# ============================================================

class KnowledgeFile(TimestampMixin, SQLModel, table=True):
    """知识库文件元数据。

    不同于 attachments 表（通用附件索引），KnowledgeFile 专用于
    文本知识库：记录文件来源、切分状态、chunk 数等。
    """

    __tablename__ = "kb_files"

    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    file_path: str = Field(max_length=1024, description="相对于 knowledge_base_dir 的相对路径")
    file_name: str = Field(max_length=256, index=True)
    file_hash: str = Field(max_length=64, index=True, description="SHA256，用于增量更新判断")
    file_size: int = Field(default=0, description="原始字节数")
    encoding: str = Field(default="utf-8")
    title: Optional[str] = Field(default=None, description="从第一个 # 提取的文档标题")
    category: Optional[str] = Field(default=None, index=True, description="用户分类，如 genshin/lore")
    tags: list[str] = Field(default_factory=list, sa_type=JSON)
    chunk_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    last_ingested_at: Optional[datetime] = Field(default=None)
    status: str = Field(default="pending", index=True, description="pending/processing/ready/failed")
    error: Optional[str] = Field(default=None, sa_type=Text)


class KnowledgeChunk(SQLModel, table=True):
    """知识库切块（chunk）。

    每条记录对应文档中的一段文本，附带：
    - 标题路径（保留章节层级）
    - 原文内容
    - embedding（BLOB，numpy float32 的 .tobytes()）
    - embedding 元信息（模型名、时间）

    embedding 直接存在 chunk 表里，避免 JOIN；启动时全部加载到内存。
    """

    __tablename__ = "kb_chunks"

    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)
    file_id: str = Field(index=True, description="关联 KnowledgeFile.file_id")
    chunk_index: int = Field(default=0, description="在文件中的顺序，0-based")
    title_path: list[str] = Field(default_factory=list, sa_type=JSON, description='["蒙德", "第二幕", "月宫与葬火"]')
    heading: Optional[str] = Field(default=None, sa_type=Text, description="最近的标题（title_path 最后一项）")
    content: str = Field(sa_type=Text)
    content_hash: str = Field(max_length=64, index=True, description="内容 SHA256，用于去重")
    token_count: int = Field(default=0, description="粗略 token 估算（中文按字，英文按词）")
    char_count: int = Field(default=0)

    # 向量字段（与 chunk 同表，避免 JOIN）
    embedding: Optional[bytes] = Field(default=None, sa_type=LargeBinary, description="numpy float32 .tobytes()")
    embedding_model: Optional[str] = Field(default=None, max_length=128)
    embedded_at: Optional[datetime] = Field(default=None)



__all__ = [
    "TimestampMixin",
    "PlatformStat",
    "ProviderStat",
    "ConversationV2",
    "PersonaFolder",
    "Persona",
    "CronJob",
    "Preference",
    "PlatformMessageHistory",
    "WebChatThread",
    "PlatformSession",
    "UmoAlias",
    "Attachment",
    "ApiKey",
    "DashboardTrustedDevice",
    "ChatUIProject",
    "SessionProjectRelation",
    "CommandConfig",
    "CommandConflict",
    "KnowledgeFile",
    "KnowledgeChunk",
    "build_umo",
    "parse_umo",
]

"""maikb — MaiBot 知识库插件核心包

提供：
- MaiKBDatabase: 异步 DAO 类
- SharedPreferences: 三层 KV API
- 全部 SQLModel 表定义
- 迁移框架
- 全局单例 get_db / sp / init_db / close_db

用法（在 MaiBot 插件中）：

    from maikb import get_db, sp, build_umo

    db = await get_db()
    conv = await db.create_conversation(
        user_id=build_umo("aiocqhttp", "GroupMessage", "123456"),
        platform_id="aiocqhttp",
    )

    await sp.plugin_put("my-plugin", "counter", 42)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .database import MaiKBDatabase
from .models import (
    ApiKey,
    Attachment,
    ChatUIProject,
    CommandConfig,
    CommandConflict,
    ConversationV2,
    CronJob,
    DashboardTrustedDevice,
    KnowledgeChunk,
    KnowledgeFile,
    Preference,
    Persona,
    PersonaFolder,
    PlatformMessageHistory,
    PlatformSession,
    PlatformStat,
    ProviderStat,
    SessionProjectRelation,
    UmoAlias,
    WebChatThread,
    TimestampMixin,
    build_umo,
    parse_umo,
)
from .preferences import SharedPreferences
from .migrations import (
    list_registered_migrations,
    register_migration,
    run_migrations,
)

# KB 模块不在此处导入，避免循环依赖
# 使用时通过 `from maikb.kb import ...` 或 `import maikb.kb as kb` 显式导入


# ----------------------------------------------------------------------
# 全局单例（plugin.py 在 on_load 中调用 init_db，在 on_unload 中调用 close_db）
# ----------------------------------------------------------------------

_db: Optional[MaiKBDatabase] = None
_sp: Optional[SharedPreferences] = None


async def init_db(db_path: str | Path) -> MaiKBDatabase:
    """初始化全局数据库单例。

    在插件 on_load 钩子中调用。会执行：
    1. 创建引擎与 sessionmaker
    2. 建表（SQLModel.metadata.create_all）
    3. PRAGMA 调优
    4. 幂等列补齐
    5. 跑所有已注册的迁移
    """

    global _db, _sp
    _db = MaiKBDatabase(db_path)
    await _db.initialize()

    # 跑迁移
    results = await run_migrations(_db)
    if not all(results.values()):
        failed = [name for name, ok in results.items() if not ok]
        # 不抛异常，只记录日志；插件可以继续运行
        import logging
        logging.getLogger("maikb").error(
            f"部分迁移失败: {failed}，请检查日志"
        )

    _sp = SharedPreferences(_db)
    return _db


async def close_db() -> None:
    """关闭全局数据库单例。在插件 on_unload 钩子中调用。"""

    global _db, _sp
    if _db is not None:
        await _db.close()
    _db = None
    _sp = None


def get_db() -> MaiKBDatabase:
    """获取全局数据库单例。

    Returns:
        MaiKBDatabase: 已初始化的数据库实例。

    Raises:
        RuntimeError: 数据库尚未初始化。
    """

    if _db is None:
        raise RuntimeError(
            "MaiKBDatabase 尚未初始化，请先调用 init_db() "
            "（通常在插件 on_load 中由 plugin.py 完成）"
        )
    return _db


def get_sp() -> SharedPreferences:
    """获取全局 SharedPreferences 单例。"""

    if _sp is None:
        raise RuntimeError("SharedPreferences 尚未初始化")
    return _sp


# 暴露 sp 作为模块级别名（与 AstrBot 原版用法一致）
class _SPProxy:
    """SharedPreferences 延迟代理，允许 `from maikb import sp` 在 init 前导入。"""

    def __getattr__(self, name: str):
        return getattr(get_sp(), name)


sp = _SPProxy()


__all__ = [
    # 数据库
    "MaiKBDatabase",
    "init_db",
    "close_db",
    "get_db",
    "get_sp",
    "sp",
    # SharedPreferences
    "SharedPreferences",
    # 模型
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
    # 工具
    "build_umo",
    "parse_umo",
    # 迁移
    "register_migration",
    "run_migrations",
    "list_registered_migrations",
]

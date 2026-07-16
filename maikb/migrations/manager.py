"""maikb.migrations.manager

自研幂等迁移框架 — 移植自 AstrBot 的迁移机制。

设计：
- 不使用 Alembic（太重）
- 用 preferences 表自身记录"哪个迁移跑过了"
- 每次启动都跑一遍所有迁移函数，已完成的自动跳过
- 迁移函数签名：async def migrate_xxx(db: MaiKBDatabase) -> None
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from ..database import MaiKBDatabase


logger = logging.getLogger("maikb.migrations")


MigrationFn = Callable[[MaiKBDatabase], Awaitable[None]]


# 全局迁移注册表
_REGISTRY: list[tuple[str, MigrationFn]] = []


def register_migration(name: str):
    """装饰器：注册一个迁移函数。

    用法：

        @register_migration("add_xxx_column")
        async def migrate_add_xxx_column(db: MaiKBDatabase) -> None:
            ...
    """

    def decorator(fn: MigrationFn) -> MigrationFn:
        _REGISTRY.append((name, fn))
        return fn

    return decorator


async def run_migrations(db: MaiKBDatabase) -> dict[str, bool]:
    """跑所有已注册的迁移（幂等，已完成的自动跳过）。

    返回 {migration_name: succeeded}。
    """

    from ..preferences import SharedPreferences
    sp = SharedPreferences(db)

    results: dict[str, bool] = {}
    for name, fn in _REGISTRY:
        if await sp.is_migration_done(name):
            results[name] = True
            continue

        logger.info(f"开始执行迁移: {name}")
        try:
            await fn(db)
            await sp.mark_migration_done(name)
            results[name] = True
            logger.info(f"迁移完成: {name}")
        except Exception as exc:
            logger.error(f"迁移失败: {name}: {exc}", exc_info=True)
            results[name] = False
            # 不抛出，让其他迁移继续尝试；但启动日志会有 ERROR

    return results


def list_registered_migrations() -> list[str]:
    """列出所有已注册的迁移名。"""

    return [name for name, _ in _REGISTRY]


# ======================================================================
# 内置迁移：从 AstrBot v3 数据库导入
# ======================================================================

# 这个迁移是空操作 — 真正的导入由 importers/astrbot_importer.py 单独脚本完成。
# 但我们仍然注册一个标记，确保未来 schema 变更时有清晰的版本路径。

@register_migration("initial_schema_v1")
async def _migrate_initial_schema_v1(db: MaiKBDatabase) -> None:
    """v1 初始 schema — 由 SQLModel.metadata.create_all 自动完成。"""

    # 实际上 create_all 在 initialize() 中已经跑过了，这里只是打个标记
    pass


__all__ = [
    "register_migration",
    "run_migrations",
    "list_registered_migrations",
]

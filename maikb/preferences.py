"""maikb.preferences

SharedPreferences 三层 KV API — 移植自 AstrBot `astrbot/core/utils/shared_preferences.py`。

三层 scope：
    scope='global', scope_id='global'    → 全局配置
    scope='umo',     scope_id=<UMO>      → 按会话配置
    scope='plugin',  scope_id=plugin_id  → 插件私有数据
    scope='migration', scope_id='global' → 迁移完成标记

同步 API 是为了和 AstrBot 原版兼容；异步 API (xxx_async) 才是真正用到的。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .database import MaiKBDatabase


logger = logging.getLogger("maikb.preferences")


_SCOPE_GLOBAL = "global"
_SCOPE_UMO = "umo"
_SCOPE_PLUGIN = "plugin"
_SCOPE_MIGRATION = "migration"


class SharedPreferences:
    """SharedPreferences 三层 KV API。

    用法（在 plugin.py 中通过全局单例访问）：

        from maikb import sp
        await sp.put_async("plugin", "my-plugin", "counter", 42)
        value = await sp.get_async("plugin", "my-plugin", "counter", default=0)
    """

    def __init__(self, db: MaiKBDatabase) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # 异步 API（推荐）
    # ------------------------------------------------------------------

    async def get_async(
        self, scope: str, scope_id: str, key: str, default: Any = None
    ) -> Any:
        """读取 KV 值，不存在时返回 default。"""

        value = await self._db.get_preference(scope, scope_id, key)
        return default if value is None else value

    async def put_async(
        self, scope: str, scope_id: str, key: str, value: Any
    ) -> None:
        """写入 KV 值（upsert）。"""

        await self._db.upsert_preference(scope, scope_id, key, value)

    async def remove_async(self, scope: str, scope_id: str, key: str) -> bool:
        """删除 KV 值，返回是否删除成功。"""

        return await self._db.delete_preference(scope, scope_id, key)

    async def list_async(
        self, scope: str, scope_id: str, key_prefix: str = ""
    ) -> dict[str, Any]:
        """列出某 scope+scope_id 下所有 KV（支持前缀过滤）。"""

        return await self._db.list_preferences(scope, scope_id, key_prefix)

    # ------------------------------------------------------------------
    # 全局 scope 便捷方法
    # ------------------------------------------------------------------

    async def global_get(self, key: str, default: Any = None) -> Any:
        return await self.get_async(_SCOPE_GLOBAL, _SCOPE_GLOBAL, key, default)

    async def global_put(self, key: str, value: Any) -> None:
        await self.put_async(_SCOPE_GLOBAL, _SCOPE_GLOBAL, key, value)

    async def global_remove(self, key: str) -> bool:
        return await self.remove_async(_SCOPE_GLOBAL, _SCOPE_GLOBAL, key)

    # ------------------------------------------------------------------
    # 会话 scope 便捷方法
    # ------------------------------------------------------------------

    async def session_get(self, umo: str, key: str, default: Any = None) -> Any:
        return await self.get_async(_SCOPE_UMO, umo, key, default)

    async def session_put(self, umo: str, key: str, value: Any) -> None:
        await self.put_async(_SCOPE_UMO, umo, key, value)

    async def session_remove(self, umo: str, key: str) -> bool:
        return await self.remove_async(_SCOPE_UMO, umo, key)

    # ------------------------------------------------------------------
    # 插件 scope 便捷方法（最重要！其他插件用这个）
    # ------------------------------------------------------------------

    async def plugin_get(
        self, plugin_id: str, key: str, default: Any = None
    ) -> Any:
        return await self.get_async(_SCOPE_PLUGIN, plugin_id, key, default)

    async def plugin_put(self, plugin_id: str, key: str, value: Any) -> None:
        await self.put_async(_SCOPE_PLUGIN, plugin_id, key, value)

    async def plugin_remove(self, plugin_id: str, key: str) -> bool:
        return await self.remove_async(_SCOPE_PLUGIN, plugin_id, key)

    async def plugin_list(
        self, plugin_id: str, key_prefix: str = ""
    ) -> dict[str, Any]:
        return await self.list_async(_SCOPE_PLUGIN, plugin_id, key_prefix)

    # ------------------------------------------------------------------
    # 迁移 scope 便捷方法
    # ------------------------------------------------------------------

    async def is_migration_done(self, migration_name: str) -> bool:
        return bool(
            await self.get_async(
                _SCOPE_MIGRATION, _SCOPE_GLOBAL, f"migration_done_{migration_name}", False
            )
        )

    async def mark_migration_done(self, migration_name: str) -> None:
        await self.put_async(
            _SCOPE_MIGRATION, _SCOPE_GLOBAL, f"migration_done_{migration_name}", True
        )


__all__ = ["SharedPreferences"]

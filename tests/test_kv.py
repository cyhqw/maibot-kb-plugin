"""tests.test_kv — KV 三层 scope 测试"""

import pytest

from maikb import SharedPreferences
from maikb.models import build_umo


@pytest.mark.asyncio
async def test_global_kv(db_instance):
    """全局 KV 基本 CRUD。"""

    sp = SharedPreferences(db_instance)

    # 写入
    await sp.global_put("app_name", "AstrBotDB")
    # 读取
    assert await sp.global_get("app_name") == "AstrBotDB"
    # 默认值
    assert await sp.global_get("nonexistent", default="default") == "default"
    # 删除
    assert await sp.global_remove("app_name") is True
    assert await sp.global_get("app_name") is None
    # 重复删除
    assert await sp.global_remove("app_name") is False


@pytest.mark.asyncio
async def test_session_kv(db_instance):
    """按 UMO 隔离的会话 KV。"""

    sp = SharedPreferences(db_instance)
    umo1 = build_umo("aiocqhttp", "GroupMessage", "111")
    umo2 = build_umo("aiocqhttp", "GroupMessage", "222")

    await sp.session_put(umo1, "topic", "技术讨论")
    await sp.session_put(umo2, "topic", "闲聊")

    assert await sp.session_get(umo1, "topic") == "技术讨论"
    assert await sp.session_get(umo2, "topic") == "闲聊"

    # 列出
    items = await sp.list_async("umo", umo1)
    assert "topic" in items
    assert items["topic"] == "技术讨论"


@pytest.mark.asyncio
async def test_plugin_kv(db_instance):
    """插件隔离 KV。"""

    sp = SharedPreferences(db_instance)

    await sp.plugin_put("plugin-a", "counter", 1)
    await sp.plugin_put("plugin-b", "counter", 100)

    assert await sp.plugin_get("plugin-a", "counter") == 1
    assert await sp.plugin_get("plugin-b", "counter") == 100

    # 前缀过滤
    await sp.plugin_put("plugin-a", "config.theme", "dark")
    await sp.plugin_put("plugin-a", "config.lang", "zh-CN")
    await sp.plugin_put("plugin-a", "user.last_login", "2024-01-01")

    config_items = await sp.plugin_list("plugin-a", "config.")
    assert len(config_items) == 2
    assert config_items["config.theme"] == "dark"
    assert config_items["config.lang"] == "zh-CN"


@pytest.mark.asyncio
async def test_upsert(db_instance):
    """upsert 同一个 key 应该是更新而不是插入。"""

    sp = SharedPreferences(db_instance)

    await sp.global_put("counter", 1)
    await sp.global_put("counter", 2)
    await sp.global_put("counter", 3)

    assert await sp.global_get("counter") == 3

    # 验证 preferences 表中 counter 这个 key 只有一行
    from sqlalchemy import text
    async with db_instance.get_db() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM preferences WHERE scope='global' "
                "AND scope_id='global' AND key='counter'"
            )
        )
        count = int(result.scalar() or 0)
    assert count == 1


@pytest.mark.asyncio
async def test_complex_values(db_instance):
    """KV 存复杂 JSON 值。"""

    sp = SharedPreferences(db_instance)

    complex_value = {
        "name": "test",
        "tags": ["a", "b", "c"],
        "nested": {"x": 1, "y": [True, False, None]},
        "count": 42,
    }
    await sp.global_put("complex", complex_value)
    loaded = await sp.global_get("complex")
    assert loaded == complex_value
    assert loaded["tags"] == ["a", "b", "c"]
    assert loaded["nested"]["y"] == [True, False, None]


@pytest.mark.asyncio
async def test_migration_marker(db_instance):
    """迁移完成标记应能正常写入读取。"""

    sp = SharedPreferences(db_instance)

    assert await sp.is_migration_done("test_xxx") is False
    await sp.mark_migration_done("test_xxx")
    assert await sp.is_migration_done("test_xxx") is True

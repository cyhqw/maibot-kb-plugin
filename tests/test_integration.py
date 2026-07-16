"""tests.test_integration — 端到端集成测试

模拟两个插件通过 API 互调的场景：
- Plugin A (maikb) 提供 maikb.kv.put / maikb.kv.get 等 API
- Plugin B (consumer) 通过 self.ctx.api.call('maikb.kv.put', ...) 调用

由于测试环境没有 MaiBot Runtime，这里直接构造一个 plugin 实例，
绕过 RPC，把 API 调用直接路由到 plugin 实例的方法上。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 让测试能找到 maikb 和 plugin 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
# maibot_sdk 由 conftest 统一处理（真实 SDK 优先，否则回退到测试桩）


class FakePluginContext:
    """模拟 MaiBot PluginContext，把 api.call 直接路由到 plugin 实例方法。"""

    def __init__(self, plugin):
        self._plugin = plugin
        self.logger = plugin._get_logger() if hasattr(plugin, "_get_logger") else __import__("logging").getLogger("test")

    class _SendCapability:
        def __init__(self, ctx):
            self._ctx = ctx
            self.sent_messages = []

        async def text(self, msg, stream_id="", **_):
            self.sent_messages.append((stream_id, msg))

    class _ApiCapability:
        def __init__(self, ctx):
            self._ctx = ctx

        async def call(self, api_name, *, version="", **kwargs):
            """直接路由到 plugin 实例的方法。"""

            plugin = self._ctx._plugin
            # 找到对应 api_name 的方法
            for attr_name in dir(plugin):
                attr = getattr(plugin, attr_name, None)
                if attr is None:
                    continue
                # 检查是否被 @API 装饰
                from maibot_sdk.components import _COMPONENT_INFO_ATTR, APIComponentInfo
                info = getattr(attr, _COMPONENT_INFO_ATTR, None)
                if isinstance(info, APIComponentInfo) and info.name == api_name:
                    return await attr(**kwargs)
            raise AttributeError(f"API {api_name!r} 未注册")

    class _Paths:
        def __init__(self, data_dir, runtime_dir):
            self.data_dir = Path(data_dir)
            self.runtime_dir = Path(runtime_dir)

    def __getattr__(self, name):
        # 延迟构造 capabilities
        if name == "send":
            self.send = self._SendCapability(self)
            return self.send
        if name == "api":
            self.api = self._ApiCapability(self)
            return self.api
        raise AttributeError(name)


@pytest.mark.asyncio
async def test_end_to_end_kv_via_api(tmp_path):
    """端到端：通过模拟 API 调用 KV put/get。"""

    import plugin as plugin_module
    from maikb import init_db, close_db

    p = plugin_module.MaiKBPlugin()

    # 注入配置
    p._plugin_config_data = {
        "database": {"enabled": True, "db_filename": "test_e2e.db", "config_version": "1.0.0", "auto_backup_on_start": False},
        "admin": {"admin_users": [], "config_version": "1.0.0"},
        # 集成测试聚焦 DB/KV/对话；关闭 KB 与 Web UI 避免起 dummy embedder 与固定端口 server
        "knowledge_base": {"enabled": False, "config_version": "1.0.0"},
        "webui": {"enabled": False, "config_version": "1.0.0"},
    }
    p._plugin_config_instance = plugin_module.MaiKBConfig(**p._plugin_config_data)

    # 注入上下文
    data_dir = tmp_path / "data"
    runtime_dir = tmp_path / "runtime"
    data_dir.mkdir()
    runtime_dir.mkdir()

    # 用最小化的 PluginContext 子类
    from maibot_sdk.context import PluginContext, PluginPaths
    ctx = PluginContext(
        plugin_id="maibot-team.maikb",
        rpc_call=None,
        paths=PluginPaths(data_dir=data_dir, runtime_dir=runtime_dir),
    )
    p._set_context(ctx)

    # on_load
    await p.on_load()

    # 现在通过 API 路由调用 KV
    # 构造一个假的 caller，通过 ctx.api.call 调用本插件的 api_kv_put / api_kv_get
    # 由于测试环境没有 RPC，我们直接调用 plugin 实例的方法（绕过 ctx.api）

    # put
    result = await p.api_kv_put(
        scope="plugin",
        scope_id="test-plugin",
        key="greeting",
        value="hello world",
    )
    assert result["success"] is True

    # get
    val = await p.api_kv_get(
        scope="plugin",
        scope_id="test-plugin",
        key="greeting",
        default="",
    )
    assert val == "hello world"

    # list
    listing = await p.api_kv_list(
        scope="plugin", scope_id="test-plugin", key_prefix=""
    )
    assert listing["count"] == 1
    assert listing["items"]["greeting"] == "hello world"

    # delete
    result = await p.api_kv_delete(
        scope="plugin", scope_id="test-plugin", key="greeting"
    )
    assert result["success"] is True

    # 确认已删除
    val = await p.api_kv_get(
        scope="plugin",
        scope_id="test-plugin",
        key="greeting",
        default="<gone>",
    )
    assert val == "<gone>"

    await p.on_unload()


@pytest.mark.asyncio
async def test_end_to_end_conversation_via_api(tmp_path):
    """端到端：通过 API 创建对话、列出对话、删除对话。"""

    import plugin as plugin_module
    from maikb import init_db, close_db

    p = plugin_module.MaiKBPlugin()
    p._plugin_config_data = {
        "database": {"enabled": True, "db_filename": "test_e2e_conv.db", "config_version": "1.0.0", "auto_backup_on_start": False},
        "admin": {"admin_users": [], "config_version": "1.0.0"},
        "knowledge_base": {"enabled": False, "config_version": "1.0.0"},
        "webui": {"enabled": False, "config_version": "1.0.0"},
    }
    p._plugin_config_instance = plugin_module.MaiKBConfig(**p._plugin_config_data)

    from maibot_sdk.context import PluginContext, PluginPaths
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ctx = PluginContext(
        plugin_id="maibot-team.maikb",
        rpc_call=None,
        paths=PluginPaths(data_dir=data_dir, runtime_dir=tmp_path / "runtime"),
    )
    p._set_context(ctx)

    await p.on_load()

    # create
    result = await p.api_conv_create(
        platform="aiocqhttp",
        message_type="GroupMessage",
        session_id="999999",
        title="集成测试对话",
    )
    assert "conversation_id" in result
    cid = result["conversation_id"]

    # get
    fetched = await p.api_conv_get(conversation_id=cid)
    assert fetched is not None
    assert fetched["title"] == "集成测试对话"
    assert fetched["user_id"] == "aiocqhttp:GroupMessage:999999"

    # list
    listing = await p.api_conv_list(
        platform="aiocqhttp",
        message_type="GroupMessage",
        session_id="999999",
    )
    assert listing["count"] == 1
    assert listing["items"][0]["conversation_id"] == cid

    # update content
    new_content = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
    upd = await p.api_conv_update_content(
        conversation_id=cid, content=new_content, token_usage_delta=20
    )
    assert upd["success"] is True

    # 验证更新
    fetched = await p.api_conv_get(conversation_id=cid)
    assert fetched["content"] == new_content
    assert fetched["token_usage"] == 20

    # delete
    result = await p.api_conv_delete(conversation_id=cid)
    assert result["success"] is True

    # 确认已删除
    fetched = await p.api_conv_get(conversation_id=cid)
    assert fetched is None

    await p.on_unload()


@pytest.mark.asyncio
async def test_end_to_end_message_history(tmp_path):
    """端到端：消息历史写入与查询。"""

    import plugin as plugin_module

    p = plugin_module.MaiKBPlugin()
    p._plugin_config_data = {
        "database": {"enabled": True, "db_filename": "test_e2e_msg.db", "config_version": "1.0.0", "auto_backup_on_start": False},
        "admin": {"admin_users": [], "config_version": "1.0.0"},
        "knowledge_base": {"enabled": False, "config_version": "1.0.0"},
        "webui": {"enabled": False, "config_version": "1.0.0"},
    }
    p._plugin_config_instance = plugin_module.MaiKBConfig(**p._plugin_config_data)

    from maibot_sdk.context import PluginContext, PluginPaths
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ctx = PluginContext(
        plugin_id="maibot-team.maikb",
        rpc_call=None,
        paths=PluginPaths(data_dir=data_dir, runtime_dir=tmp_path / "runtime"),
    )
    p._set_context(ctx)

    await p.on_load()

    # 写入 5 条消息
    for i in range(5):
        await p.api_msg_add(
            platform="aiocqhttp",
            message_type="GroupMessage",
            session_id="888888",
            content={"role": "user", "content": f"消息 {i}"},
            sender_id=f"user-{i}",
            sender_name=f"用户{i}",
        )

    # 查询
    result = await p.api_msg_list(
        platform="aiocqhttp",
        message_type="GroupMessage",
        session_id="888888",
        limit=10,
    )
    assert result["count"] == 5
    # 应该按 id 倒序（最新优先）
    contents = [item["content"]["content"] for item in result["items"]]
    assert "消息 4" in contents[0]

    await p.on_unload()

"""AstrBot DB Demo — 示例调用插件

演示如何通过 MaiBot 插件间 API 调用 astrbot-db-port 插件：
- /demo kv <key> [value]   读写 KV
- /demo conv create <title>  创建对话
- /demo conv list            列出当前用户的所有对话
- /demo stats                查看数据库统计
"""

from __future__ import annotations

from typing import Any, ClassVar

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase


class DemoConfig(PluginConfigBase):
    __ui_label__: ClassVar[str] = "Demo"

    plugin: PluginSectionConfig = Field(default_factory=lambda: PluginSectionConfig())


class PluginSectionConfig(PluginConfigBase):
    __ui_label__: ClassVar[str] = "插件"
    config_version: str = Field(default="1.0.0")
    enabled: bool = Field(default=True)


class AstrBotDbDemoPlugin(MaiBotPlugin):
    """Demo 插件：调用 astrbot-db-port 的 API。"""

    config_model = DemoConfig

    async def on_load(self) -> None:
        self.ctx.logger.info("AstrBot DB Demo 插件已加载")

    async def on_unload(self) -> None:
        pass

    # ------------------------------------------------------------------
    # /demo kv <key> [value]
    # ------------------------------------------------------------------

    @Command(
        "demo_kv",
        description="读写 KV 示例",
        pattern=r"^/demo\s+kv\s+(?P<key>\S+)(?:\s+(?P<value>.+))?\s*$",
    )
    async def cmd_kv(
        self,
        stream_id: str = "",
        platform: str = "",
        user_id: str = "",
        matched_groups: dict | None = None,
        **_: Any,
    ) -> tuple[bool, str, bool]:
        key = (matched_groups or {}).get("key", "")
        value = (matched_groups or {}).get("value")

        if value is None:
            # 读
            result = await self.ctx.api.call(
                "astrdb.kv.get",
                version="1",
                scope="plugin",
                scope_id="maibot-team.astrbot-db-demo",
                key=key,
                default="<未设置>",
            )
            await self.ctx.send.text(f"KV[{key}] = {result}", stream_id)
        else:
            # 写
            # 简单类型转换：尝试解析为 JSON，失败则当字符串
            import json
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                parsed = value

            await self.ctx.api.call(
                "astrdb.kv.put",
                version="1",
                scope="plugin",
                scope_id="maibot-team.astrbot-db-demo",
                key=key,
                value=parsed,
            )
            await self.ctx.send.text(
                f"已写入 KV[{key}] = {parsed!r}", stream_id
            )

        return True, "done", True

    # ------------------------------------------------------------------
    # /demo conv create <title>
    # ------------------------------------------------------------------

    @Command(
        "demo_conv_create",
        description="创建对话示例",
        pattern=r"^/demo\s+conv\s+create\s+(?P<title>.+)\s*$",
    )
    async def cmd_conv_create(
        self,
        stream_id: str = "",
        platform: str = "",
        user_id: str = "",
        matched_groups: dict | None = None,
        **_: Any,
    ) -> tuple[bool, str, bool]:
        title = (matched_groups or {}).get("title", "").strip()
        if not title:
            await self.ctx.send.text("请提供对话标题", stream_id)
            return False, "no title", True

        result = await self.ctx.api.call(
            "astrdb.conv.create",
            version="1",
            platform=platform or "unknown",
            message_type="FriendMessage",
            session_id=user_id or "unknown",
            title=title,
        )

        if isinstance(result, dict) and "conversation_id" in result:
            await self.ctx.send.text(
                f"对话已创建 ✅\n"
                f"  ID: {result['conversation_id']}\n"
                f"  Title: {title}",
                stream_id,
            )
        else:
            await self.ctx.send.text(
                f"创建失败: {result}", stream_id
            )

        return True, "done", True

    # ------------------------------------------------------------------
    # /demo conv list
    # ------------------------------------------------------------------

    @Command(
        "demo_conv_list",
        description="列出当前用户的对话",
        pattern=r"^/demo\s+conv\s+list\s*$",
    )
    async def cmd_conv_list(
        self,
        stream_id: str = "",
        platform: str = "",
        user_id: str = "",
        **_: Any,
    ) -> tuple[bool, str, bool]:
        result = await self.ctx.api.call(
            "astrdb.conv.list",
            version="1",
            platform=platform or "unknown",
            message_type="FriendMessage",
            session_id=user_id or "unknown",
            limit=10,
        )

        if not isinstance(result, dict):
            await self.ctx.send.text(f"查询失败: {result}", stream_id)
            return False, "failed", True

        items = result.get("items", [])
        if not items:
            await self.ctx.send.text("你还没有任何对话", stream_id)
            return True, "empty", True

        lines = [f"你的对话（共 {result.get('count', 0)} 个）:", ""]
        for i, c in enumerate(items, 1):
            lines.append(
                f"{i}. {c.get('title') or '<无标题>'}"
                f"\n   ID: {c.get('conversation_id')}"
                f"\n   tokens: {c.get('token_usage', 0)}"
            )
        await self.ctx.send.text("\n".join(lines), stream_id)
        return True, "done", True

    # ------------------------------------------------------------------
    # /demo stats
    # ------------------------------------------------------------------

    @Command(
        "demo_stats",
        description="查看数据库统计",
        pattern=r"^/demo\s+stats\s*$",
    )
    async def cmd_stats(
        self,
        stream_id: str = "",
        **_: Any,
    ) -> tuple[bool, str, bool]:
        """查询几个关键表的行数。"""

        tables = [
            "conversations",
            "preferences",
            "platform_message_history",
            "personas",
            "platform_stats",
        ]
        lines = ["【AstrBot DB 统计】", ""]
        for t in tables:
            result = await self.ctx.api.call(
                "astrdb.stats.count",
                version="1",
                table_name=t,
            )
            if isinstance(result, dict) and "count" in result:
                lines.append(f"  {t}: {result['count']}")
            else:
                lines.append(f"  {t}: <error>")

        await self.ctx.send.text("\n".join(lines), stream_id)
        return True, "done", True


def create_plugin() -> AstrBotDbDemoPlugin:
    return AstrBotDbDemoPlugin()

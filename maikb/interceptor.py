"""maikb.interceptor

消息前缀拦截器 — 通过 MaiBot Hook 机制实现"不记录、不回复"。

机制：
- 注册 `chat.receive.before_process` Hook（BLOCKING + EARLY）
- 在消息进入主链路前检查前缀
- 命中前缀时返回 {"action": "abort"}，主链路直接 return
- 此时消息还没进 chat_manager / message_repository / A_memorix
  所以记忆系统也读不到这条消息

支持的前缀（默认）：
- `/`  命令前缀（MaiBot 自带命令系统也用这个，但本拦截器更早）
- `[`  常见的引述/机器人指令前缀
- `#`  标签/注释前缀

可配置：
- 启用/禁用
- 自定义前缀字符列表
- 是否记录被拦截的消息（debug 用）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from maibot_sdk import HookHandler
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder


logger = logging.getLogger("maikb.interceptor")


# 默认前缀字符
DEFAULT_PREFIXES = ["/", "[", "#"]


def extract_message_text(message: Any) -> str:
    """从 MaiBot 消息对象中提取纯文本。

    MaiBot 的 message 可能是 dict 或对象，尝试多种字段。
    """

    if message is None:
        return ""

    # dict 形式
    if isinstance(message, dict):
        for key in ("processed_plain_text", "plain_text", "raw_message", "text", "content"):
            v = message.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    # 对象形式
    for attr in ("processed_plain_text", "plain_text", "raw_message", "text", "content"):
        v = getattr(message, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def should_block(text: str, prefixes: list[str]) -> tuple[bool, str]:
    """判断文本是否命中前缀。

    优先匹配最长前缀（如 "!!" 优先于 "!"），避免短前缀"吃掉"长前缀。

    Returns:
        (blocked, matched_prefix)
    """

    if not text:
        return False, ""
    text = text.lstrip()  # 允许前导空格
    if not text:
        return False, ""
    # 按长度降序，优先匹配长前缀
    sorted_prefixes = sorted(
        [p for p in prefixes if p],
        key=len,
        reverse=True,
    )
    for prefix in sorted_prefixes:
        if text.startswith(prefix):
            return True, prefix
    return False, ""


class InterceptorMixin:
    """前缀拦截器 Mixin，由 MaiKBPlugin 继承。

    配置项（在 [interceptor] section）：
    - enabled: bool = True
    - prefixes: list[str] = ["/", "[", "#"]
    - log_blocked: bool = True
    """

    @HookHandler(
        "chat.receive.before_process",
        name="maikb_prefix_guard",
        description="拦截带特定前缀（/ [ # 等）的消息，不记录、不回复",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
    )
    async def hook_prefix_guard(
        self,
        message: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """前缀拦截 Hook。"""

        try:
            cfg = self.config.interceptor  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            # 配置未注入，不拦截
            return {"action": "continue"}

        if not cfg.enabled:
            return {"action": "continue"}

        text = extract_message_text(message)
        if not text:
            return {"action": "continue"}

        blocked, matched = should_block(text, cfg.prefixes or DEFAULT_PREFIXES)
        if not blocked:
            return {"action": "continue"}

        # 命中前缀，拦截
        if cfg.log_blocked:
            try:
                logger_info = self.ctx.logger  # type: ignore[attr-defined]
            except Exception:
                logger_info = logger
            preview = text[:60].replace("\n", " ")
            logger_info.info(
                f"消息被前缀拦截器拦截 (prefix={matched!r}): {preview!r}"
            )

        return {
            "action": "abort",
            "custom_result": {
                "blocked_by": "maikb.prefix_guard",
                "matched_prefix": matched,
            },
        }


__all__ = [
    "InterceptorMixin",
    "DEFAULT_PREFIXES",
    "extract_message_text",
    "should_block",
]

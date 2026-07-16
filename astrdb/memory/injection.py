"""astrdb.memory.injection

记忆注入适配器 — 多模式注入 + 格式化。

移植自 LivingMemory `core/utils/injection_adapter.py`。

支持 3 种注入方式（LivingMemory 有 6 种，这里简化为最常用的）：
- extra_user_content: 追加到 user message（推荐，不污染历史）
- user_message_before: 拼到当前 user message 前面
- user_message_after: 拼到当前 user message 后面

格式化策略（移植自 LivingMemory format_memories_for_injection）：
- 带明确的 BEGIN/END 标记
- 包含 importance、time、type 信息
- 强调"PAST records, trust current conversation"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .retriever import AtomSearchHit


logger = logging.getLogger("astrdb.memory.injection")


# 注入文本模板（移植自 LivingMemory，简化）
_INJECTION_TEMPLATE = """<Memory-Reference>
--- BEGIN HISTORICAL MEMORY REFERENCE ---
以下是从过往对话中提取的历史记忆，供回答时参考：

{chunks}

--- END HISTORICAL MEMORY REFERENCE ---

注意：
- 以上内容均为历史记录，可能与当前对话有出入
- 优先信任当前对话中的新信息
- 引用时请注明记忆来源（如"根据之前的记录..."）
</Memory-Reference>"""


def format_atoms_for_injection(hits: list[AtomSearchHit]) -> str:
    """格式化 atoms 为 LLM 可读的注入文本。"""

    if not hits:
        return ""

    lines = []
    for i, h in enumerate(hits, 1):
        # 时间格式化
        time_str = ""
        if h.created_at_ts > 0:
            try:
                dt = datetime.fromtimestamp(h.created_at_ts, tz=timezone.utc)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        # 类型中文映射
        type_map = {
            "episodic": "事件",
            "factual": "事实",
            "relational": "关系",
            "preference": "偏好",
            "planned": "计划",
            "unknown": "其他",
        }
        type_str = type_map.get(h.atom_type, h.atom_type)

        lines.append(
            f"记忆 #{i} [{type_str}] (重要性: {h.importance:.2f}, 置信度: {h.confidence:.2f}, 时间: {time_str})"
        )
        lines.append(f"  {h.content}")
        if h.entities:
            lines.append(f"  相关实体: {', '.join(h.entities)}")
        lines.append("")

    chunks_text = "\n".join(lines)
    return _INJECTION_TEMPLATE.format(chunks=chunks_text)


def inject_into_messages(
    messages: list[dict[str, Any]],
    injection_text: str,
    mode: str = "extra_user_content",
) -> list[dict[str, Any]]:
    """把注入文本插入到 messages 列表。

    Args:
        messages: 原始 messages 列表
        injection_text: 注入文本
        mode: 注入方式
            - extra_user_content: 追加新的 user message（推荐）
            - user_message_before: 拼到当前 user message 前面
            - user_message_after: 拼到当前 user message 后面

    Returns:
        修改后的 messages 列表（不修改原列表）
    """

    if not injection_text or not messages:
        return list(messages)

    modified = list(messages)

    if mode == "extra_user_content":
        # 追加新的 user message
        modified.append({"role": "user", "content": injection_text})

    elif mode == "user_message_before":
        # 找到最后一条 user message，前面拼接
        for i in range(len(modified) - 1, -1, -1):
            if modified[i].get("role") == "user":
                original = modified[i].get("content", "")
                if isinstance(original, str):
                    modified[i] = {
                        **modified[i],
                        "content": injection_text + "\n\n" + original,
                    }
                break
        else:
            # 没有 user message，直接追加
            modified.append({"role": "user", "content": injection_text})

    elif mode == "user_message_after":
        # 找到最后一条 user message，后面拼接
        for i in range(len(modified) - 1, -1, -1):
            if modified[i].get("role") == "user":
                original = modified[i].get("content", "")
                if isinstance(original, str):
                    modified[i] = {
                        **modified[i],
                        "content": original + "\n\n" + injection_text,
                    }
                break
        else:
            modified.append({"role": "user", "content": injection_text})

    else:
        logger.warning(f"未知注入方式: {mode}，回退到 extra_user_content")
        modified.append({"role": "user", "content": injection_text})

    return modified


class MemoryInjector:
    """记忆注入器 — 在 LLM 调用前自动检索 atoms 并注入。

    与 astrdb.injector.InjectorMixin（KB 注入器）的区别：
    - KB 注入器：检索知识库文档（KB RAG）
    - Memory 注入器：检索对话记忆 atoms（LivingMemory 移植）

    两者可以并存，各自注入不同的内容。
    """

    def __init__(
        self,
        retriever,
        *,
        enabled: bool = True,
        injection_mode: str = "extra_user_content",
        top_k: int = 3,
        min_score: float = 0.1,
        max_chars: int = 2000,
        dedup_lookback: int = 6,
        skip_if_tool_called: bool = True,
    ) -> None:
        self._retriever = retriever
        self._enabled = enabled
        self._mode = injection_mode
        self._top_k = top_k
        self._min_score = min_score
        self._max_chars = max_chars
        self._dedup_lookback = dedup_lookback
        self._skip_if_tool_called = skip_if_tool_called

    async def inject(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """检索记忆并注入到 messages。

        Returns:
            (injected, modified_messages)
        """

        if not self._enabled or not messages:
            return False, messages

        # 提取最后一条 user 消息作为 query
        query = self._extract_last_user_text(messages)
        if not query or len(query) < 3:
            return False, messages

        # 去重：检查最近是否已调过 memory tool
        if self._skip_if_tool_called and self._has_recent_memory_tool_call(messages):
            return False, messages

        # 检索
        from .retriever import AtomSearchQuery

        q = AtomSearchQuery(
            query=query,
            top_k=self._top_k,
            session_id=session_id,
            persona_id=persona_id,
            min_score=self._min_score,
        )
        try:
            hits = await self._retriever.search(q)
        except Exception as exc:
            logger.warning(f"记忆检索失败: {exc}")
            return False, messages

        if not hits:
            return False, messages

        # 格式化
        injection_text = format_atoms_for_injection(hits)
        if len(injection_text) > self._max_chars:
            injection_text = injection_text[: self._max_chars - 50] + "\n\n...（已截断）"

        # 注入
        modified = inject_into_messages(messages, injection_text, self._mode)
        logger.info(
            f"注入记忆: query={query[:50]!r} hits={len(hits)} chars={len(injection_text)}"
        )
        return True, modified

    def _extract_last_user_text(self, messages: list[dict[str, Any]]) -> str:
        """提取最后一条 user 消息文本。"""

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                return " ".join(texts).strip()
        return ""

    def _has_recent_memory_tool_call(self, messages: list[dict[str, Any]]) -> bool:
        """检查最近是否调过记忆相关 tool。"""

        if not messages:
            return False
        recent = messages[-self._dedup_lookback :] if len(messages) > self._dedup_lookback else messages
        for msg in recent:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls or not isinstance(tool_calls, list):
                continue
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        name = fn.get("name", "")
                        if name in ("memory_search", "recall_long_term_memory", "memory_memorize"):
                            return True
        return False


__all__ = [
    "MemoryInjector",
    "format_atoms_for_injection",
    "inject_into_messages",
]

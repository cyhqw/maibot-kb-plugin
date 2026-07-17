"""maikb.injector

自动召回 + 注入器 — 在 LLM 调用前自动检索知识库并注入到 prompt。

机制：
- 注册 `maisaka.replyer.before_model_request` Hook（BLOCKING）
- 这是官方文档指定的"改写最终发给模型的消息列表"的 Hook 点
- 在 replyer 构建完最终 messages 之后、真正请求模型之前触发
- 拿到 messages 列表中最后一条 user message 作为 query
- 调用本插件的 HybridSearcher 检索知识库
- 命中高置信度结果时，把检索结果作为一条 user message 插入到最后一条
  user message 之前（标准 RAG 模式：上下文在问题之前），让 LLM 基于它回答
- 通过 modified_kwargs 返回新 messages，真正影响本次 LLM 请求
- 该 Hook 只改写本次临时请求，不写回聊天历史、不影响中期记忆插入

注意：不要用 `maisaka.planner.before_request` 注入 messages。planner 阶段
messages 尚未最终构建，replyer 之后会重建列表，planner Hook 里改的 messages
可能被丢弃。改最终消息列表必须用 replyer.before_model_request。

去重逻辑：
- 检查最近 N 轮 messages 中是否已包含 "knowledge_search" tool call
- 如果 LLM 已经主动调过 tool，本轮不再自动注入（避免重复）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from maibot_sdk import HookHandler
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder


logger = logging.getLogger("maikb.injector")


# 注入文本模板
_INJECTION_TEMPLATE = """【知识库参考】
以下是从本地知识库检索到的与用户问题相关的内容，供你回答时参考：

{chunks}

请基于以上材料回答用户问题。如果材料中没有答案，请说明知识库中无相关信息。"""


def _format_chunks(hits: list) -> str:
    """格式化检索结果为 LLM 易读文本。"""

    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits, 1):
        title_path = " > ".join(h.title_path) if h.title_path else "<无标题>"
        source = h.source_name or "未知来源"
        lines.append(f"### {i}. {h.heading or title_path}")
        lines.append(f"来源: {source} | 章节: {title_path}")
        lines.append("")
        lines.append(h.content)
        lines.append("")
        lines.append("---")
    return "\n".join(lines)


def _extract_last_user_text(messages: list) -> str:
    """从 messages 列表中提取最后一条 user 消息文本。

    messages 可能是 list of dict 或 list of object。
    """

    if not messages:
        return ""

    # 倒序找最后一条 role=user
    for msg in reversed(messages):
        role = None
        content = None

        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

        if role == "user":
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                # OpenAI 多模态格式：content 是 list of {type, text}
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
                return " ".join(texts).strip()
    return ""


def _has_recent_tool_call(messages: list, tool_name: str = "knowledge_search", look_back: int = 6) -> bool:
    """检查最近 N 条消息中是否有指定 tool 的调用。

    避免与 LLM 主动调 tool 重复注入。
    """

    if not messages:
        return False

    recent = messages[-look_back:] if len(messages) > look_back else messages
    for msg in recent:
        # tool_calls 在 assistant 消息里
        if isinstance(msg, dict):
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")
        else:
            role = getattr(msg, "role", None)
            tool_calls = getattr(msg, "tool_calls", None)

        if role != "assistant" or not tool_calls:
            continue

        # tool_calls 格式：[{id, type, function: {name, arguments}}]
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict) and fn.get("name") == tool_name:
                        return True
                # 也支持简单字符串形式
                elif isinstance(tc, str) and tool_name in tc:
                    return True
    return False


def _truncate_messages_if_needed(messages: list, max_tokens: int, current_tokens: int) -> list:
    """如果注入后超 token，从前面删 user/assistant 消息（保留 system 和最后一条 user）。

    简化实现：不真的截断，只警告。完整实现需要 token 计数器。
    """

    return messages


class InjectorMixin:
    """自动召回 + 注入器 Mixin。

    配置项（在 [injector] section）：
    - enabled: bool = True
    - min_score: float = 0.01  # RRF 融合分数阈值
    - min_vector_score: float = 0.3  # 向量相似度阈值
    - top_k: int = 3  # 注入几条
    - max_chars: int = 2000  # 注入文本最大字符数
    - dedup_lookback: int = 6  # 检查最近 N 条消息避免重复
    - skip_if_tool_called: bool = True  # LLM 已调过 knowledge_search 时跳过
    """

    @HookHandler(
        "maisaka.replyer.before_model_request",
        name="maikb_kb_auto_inject",
        description="在 replyer 构建最终消息后、请求模型前，自动检索知识库并注入相关内容",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        error_policy=ErrorPolicy.SKIP,
    )
    async def hook_auto_inject(
        self,
        messages: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """自动召回 + 注入 Hook。

        使用 replyer.before_model_request：这是改写"真正发给模型的消息列表"的
        官方 Hook 点，插入的 user message 会进入本次 LLM 请求。
        """

        try:
            cfg = self.config.injector  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            return {"action": "continue"}

        if not cfg.enabled:
            return {"action": "continue"}

        # KB 模块未初始化时跳过
        from .kb.api import _kb_searcher
        if _kb_searcher is None:
            return {"action": "continue"}

        # messages 必须是 list
        if not isinstance(messages, list) or not messages:
            return {"action": "continue"}

        # 提取最后一条 user 消息作为 query
        query = _extract_last_user_text(messages)
        if not query or len(query) < 3:
            # 查询太短，向量检索意义不大
            return {"action": "continue"}

        # 去重：如果 LLM 最近调过 knowledge_search，跳过
        if cfg.skip_if_tool_called and _has_recent_tool_call(
            messages, "knowledge_search", cfg.dedup_lookback
        ):
            return {"action": "continue"}

        # 检索
        try:
            from .kb import SearchQuery
            # 从配置读取融合模式，默认 vector_ranked（BM25 仅召回不排序）
            fusion_mode = getattr(cfg, "fusion_mode", "vector_ranked")
            valid_modes = ("hybrid", "vector_ranked", "vector_only")
            if fusion_mode not in valid_modes:
                fusion_mode = "vector_ranked"
            q = SearchQuery(
                query=query,
                top_k=cfg.top_k,
                use_vector=True,
                use_bm25=True,
                fusion_mode=fusion_mode,  # type: ignore[arg-type]
            )
            hits = await _kb_searcher.search(q)
        except Exception as exc:
            logger.warning(f"自动召回失败: {exc}")
            return {"action": "continue"}

        if not hits:
            return {"action": "continue"}

        # 过滤低置信度
        filtered = []
        for h in hits:
            if h.score < cfg.min_score:
                continue
            if h.vector_score < cfg.min_vector_score and h.bm25_score <= 0:
                continue
            filtered.append(h)

        if not filtered:
            return {"action": "continue"}

        # 格式化注入文本
        injection_text = _INJECTION_TEMPLATE.format(chunks=_format_chunks(filtered))

        # 截断到 max_chars
        if len(injection_text) > cfg.max_chars:
            injection_text = injection_text[: cfg.max_chars - 50] + "\n\n...（已截断）"

        try:
            logger_info = self.ctx.logger  # type: ignore[attr-defined]
        except Exception:
            logger_info = logger
        logger_info.info(
            f"自动注入知识库参考: query={query[:50]!r} hits={len(filtered)} "
            f"chars={len(injection_text)}"
        )

        # 注入：把知识库参考作为一条 user message 插入到最后一条 user message 之前
        # 标准 RAG 模式：上下文在问题之前，LLM 据此回答当前问题
        new_message = {"role": "user", "content": injection_text}
        modified_messages = list(messages)

        insert_at = len(modified_messages)
        for i in range(len(modified_messages) - 1, -1, -1):
            msg = modified_messages[i]
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user":
                insert_at = i
                break
        modified_messages.insert(insert_at, new_message)

        # 返回改写后的 messages；保留其余 kwargs 不变
        modified_kwargs = dict(kwargs)
        modified_kwargs["messages"] = modified_messages
        return {
            "action": "continue",
            "modified_kwargs": modified_kwargs,
        }


__all__ = ["InjectorMixin"]

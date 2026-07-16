"""astrdb.memory.processor

MemoryProcessor — 对话总结 + atom 抽取。

移植自 LivingMemory `core/processors/memory_processor.py`，简化为：
1. 把对话消息拼接成文本
2. 调 MaiBot LLM 总结
3. 解析 JSON 输出（summary / key_facts / topics / importance）
4. AtomClassifier 把 key_facts 分类成 atoms
5. 双通道摘要：canonical_summary（检索用）+ persona_summary（注入用）
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from ..database import AstrBotDatabase
from .atom_classifier import classify_atoms
from .atom_store import AtomStore
from .models import AtomType, MemoryAtom


logger = logging.getLogger("astrdb.memory.processor")


# ----------------------------------------------------------------------
# Prompt 模板（移植自 LivingMemory，简化）
# ----------------------------------------------------------------------

_PRIVATE_CHAT_PROMPT = """你是一个记忆管理系统。请分析以下私聊对话，提取关键信息。

对话内容：
{conversation}

请返回 JSON 格式（不要 markdown 代码块），包含以下字段：
{{
  "summary": "对话的简洁总结（50-150字）",
  "key_facts": ["关键事实1", "关键事实2", ...],
  "topics": ["主题1", "主题2", ...],
  "participants": ["参与者1", ...],
  "sentiment": "positive | neutral | negative",
  "interaction_type": "chat | question | task | emotional",
  "importance": 0.0-1.0
}}

要求：
1. key_facts 每条不超过 50 字，独立可读
2. importance: 0-0.3 琐碎闲聊；0.3-0.6 一般信息；0.6-0.8 重要事实；0.8-1.0 关键记忆
3. 只提取有价值的信息，无意义的寒暄可以 importance < 0.3
4. 当前日期：{date}
"""

_GROUP_CHAT_PROMPT = """你是一个记忆管理系统。请分析以下群聊对话，提取关键信息。

对话内容：
{conversation}

请返回 JSON 格式（不要 markdown 代码块），包含以下字段：
{{
  "summary": "对话的简洁总结（50-150字）",
  "key_facts": ["关键事实1", ...],
  "topics": ["主题1", ...],
  "participants": ["参与者昵称1", ...],
  "sentiment": "positive | neutral | negative",
  "interaction_type": "chat | question | task | emotional",
  "importance": 0.0-1.0
}}

要求：
1. key_facts 每条不超过 50 字，独立可读
2. 群聊场景注意区分不同参与者
3. importance: 0-0.3 琐碎闲聊；0.3-0.6 一般信息；0.6-0.8 重要事实；0.8-1.0 关键记忆
4. 当前日期：{date}
"""


# ----------------------------------------------------------------------
# 消息格式化
# ----------------------------------------------------------------------

def format_messages_for_llm(
    messages: list[dict[str, Any]],
    is_group_chat: bool = False,
) -> str:
    """把消息列表格式化为 LLM 可读文本。

    群聊格式：[昵称 | 时间] 内容
    私聊格式：[user/bot] 内容
    """

    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # 多模态消息，提取文本
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        sender_name = msg.get("sender_name") or msg.get("name") or role
        ts = msg.get("timestamp")

        if is_group_chat:
            time_str = ""
            if ts:
                try:
                    time_str = datetime.fromtimestamp(ts).strftime("%H:%M")
                except Exception:
                    pass
            prefix = f"[{sender_name} | {time_str}]" if time_str else f"[{sender_name}]"
        else:
            prefix = f"[{role}]"

        lines.append(f"{prefix} {content}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# JSON 解析（容错）
# ----------------------------------------------------------------------

def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    """尝试解析 JSON，支持容错修复。"""

    if not text:
        return None

    # 1. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 markdown 代码块中的 JSON
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 找第一个 { 到最后一个 } 的子串
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            # 4. 修复常见错误：移除尾逗号
            candidate = text[start : end + 1]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return None


def _extract_by_regex(text: str) -> Optional[dict[str, Any]]:
    """正则提取关键字段（最后兜底）。"""

    result: dict[str, Any] = {}

    m = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
    if m:
        result["summary"] = m.group(1)

    m = re.search(r'"importance"\s*:\s*([\d.]+)', text)
    if m:
        try:
            result["importance"] = float(m.group(1))
        except ValueError:
            pass

    # key_facts 数组
    m = re.search(r'"key_facts"\s*:\s*\[([^\]]+)\]', text, re.DOTALL)
    if m:
        facts = re.findall(r'"([^"]+)"', m.group(1))
        if facts:
            result["key_facts"] = facts

    # topics 数组
    m = re.search(r'"topics"\s*:\s*\[([^\]]+)\]', text, re.DOTALL)
    if m:
        topics = re.findall(r'"([^"]+)"', m.group(1))
        if topics:
            result["topics"] = topics

    return result if result else None


# ----------------------------------------------------------------------
# MemoryProcessor
# ----------------------------------------------------------------------

class MemoryProcessor:
    """对话总结 + atom 抽取处理器。"""

    def __init__(
        self,
        db: AstrBotDatabase,
        atom_store: AtomStore,
        llm_generate_fn=None,
        *,
        summary_trigger_rounds: int = 10,
    ) -> None:
        self._db = db
        self._store = atom_store
        self._llm_generate = llm_generate_fn  # async (prompt, system_prompt) -> str
        self._summary_trigger_rounds = summary_trigger_rounds

    def set_llm_fn(self, fn) -> None:
        """设置 LLM 调用函数（延迟注入，避免初始化时序问题）。"""

        self._llm_generate = fn

    async def process_conversation(
        self,
        messages: list[dict[str, Any]],
        *,
        parent_memory_id: str,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        is_group_chat: bool = False,
        persona_system_prompt: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """处理一段对话，生成总结 + atoms。

        Args:
            messages: 对话消息列表
            parent_memory_id: 父记忆 ID（conversation_id）
            session_id: 会话 ID
            persona_id: 人格 ID
            is_group_chat: 是否群聊
            persona_system_prompt: 人格的 system prompt

        Returns:
            {
                "summary": str,
                "key_facts": list[str],
                "topics": list[str],
                "importance": float,
                "atoms": list[MemoryAtom],
                "canonical_summary": str,  # 检索用
                "persona_summary": str,    # 注入用
            }
            失败返回 None。
        """

        if not messages:
            return None

        if self._llm_generate is None:
            logger.warning("LLM 函数未注入，无法处理对话")
            return None

        # 1. 格式化对话
        conversation_text = format_messages_for_llm(messages, is_group_chat)
        if not conversation_text.strip():
            return None

        # 2. 构造 prompt
        template = _GROUP_CHAT_PROMPT if is_group_chat else _PRIVATE_CHAT_PROMPT
        prompt = template.format(
            conversation=conversation_text,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        system_prompt = "你是一个专业的记忆管理系统，擅长从对话中提取关键信息。"
        if persona_system_prompt:
            system_prompt += f"\n\n当前人格设定：\n{persona_system_prompt}"

        # 3. 调 LLM（带重试）
        response_text = None
        for attempt in range(3):
            try:
                response_text = await self._llm_generate(
                    prompt=prompt, system_prompt=system_prompt
                )
                if response_text:
                    break
            except Exception as exc:
                logger.warning(f"LLM 调用失败 (attempt {attempt+1}): {exc}")
                import asyncio
                await asyncio.sleep(2 ** attempt)  # 指数退避

        if not response_text:
            logger.error("LLM 调用 3 次都失败，放弃本次总结")
            return None

        # 4. 解析 JSON
        parsed = _try_parse_json(response_text)
        if parsed is None:
            parsed = _extract_by_regex(response_text)
        if parsed is None:
            logger.error(f"无法解析 LLM 输出为 JSON: {response_text[:200]}")
            return None

        # 5. 校验质量
        summary = str(parsed.get("summary", "")).strip()
        key_facts = parsed.get("key_facts", []) or []
        topics = parsed.get("topics", []) or []
        importance = float(parsed.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        if not summary or len(summary) < 10:
            logger.warning(f"总结质量低（太短）: {summary!r}")
            return None

        if not key_facts:
            logger.info("无 key_facts，跳过 atom 生成")
            atoms: list[MemoryAtom] = []
        else:
            # 6. 分类 atoms
            classifications = classify_atoms(key_facts, parent_importance=importance)
            atoms = []
            for (content, (atom_type, confidence, event_time, atom_importance)) in zip(
                key_facts, classifications
            ):
                # 提取实体（简单：用 key_facts 中的名词短语）
                entities = _extract_entities(content, topics)
                atom = MemoryAtom(
                    parent_memory_id=parent_memory_id,
                    atom_type=atom_type.value,
                    content=content,
                    entities=entities,
                    importance=atom_importance,
                    confidence=confidence,
                    session_id=session_id,
                    persona_id=persona_id,
                    event_time_ts=event_time,
                    source="auto",
                )
                atoms.append(atom)

            # 7. 批量写入 atoms
            if atoms:
                await self._store.insert_atoms(atoms)

        # 8. 双通道摘要
        canonical_summary = summary
        if key_facts:
            canonical_summary = summary + " | " + "；".join(key_facts[:5])
        persona_summary = summary

        return {
            "summary": summary,
            "key_facts": key_facts,
            "topics": topics,
            "importance": importance,
            "atoms": atoms,
            "canonical_summary": canonical_summary,
            "persona_summary": persona_summary,
            "sentiment": parsed.get("sentiment", "neutral"),
            "interaction_type": parsed.get("interaction_type", "chat"),
        }


def _extract_entities(content: str, topics: list[str]) -> list[str]:
    """简单实体提取：从 content 中找 topics 命中的词。"""

    if not topics:
        return []
    return [t for t in topics if t and t in content]


__all__ = ["MemoryProcessor", "format_messages_for_llm"]

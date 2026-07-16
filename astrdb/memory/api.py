"""astrdb.memory.api

Memory 模块对外 API 的实现（Mixin）。

API 列表：
- astrdb.memorize          主动记忆（手动写入 atom）
- astrdb.mem.search        检索记忆 atoms
- astrdb.mem.list          列出 atoms
- astrdb.mem.stats         记忆统计
- astrdb.mem.reinforce     强化某 atom
- astrdb.mem.forget        删除某 atom
- astrdb.mem.decay_now     手动触发衰减
- astrdb.mem.process       处理对话生成 atoms（高级用法）

LLM Tool:
- memory_search            让 LLM 主动检索记忆
- memory_memorize          让 LLM 主动记忆

HookHandler:
- maisaka.planner.before_request  自动注入记忆到 prompt
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from maibot_sdk import API, HookHandler, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

from .. import get_db
from .atom_classifier import classify_atom
from .atom_store import AtomStore
from .injection import MemoryInjector
from .lifecycle import AtomLifecycleManager
from .models import AtomStatus, AtomType, MemoryAtom
from .retriever import AtomRetriever, AtomSearchQuery
from .processor import MemoryProcessor


logger = logging.getLogger("astrdb.memory.api")


# 全局单例（plugin 在 on_load 中初始化）
_mem_atom_store: Optional[AtomStore] = None
_mem_retriever: Optional[AtomRetriever] = None
_mem_lifecycle: Optional[AtomLifecycleManager] = None
_mem_processor: Optional[MemoryProcessor] = None
_mem_injector: Optional[MemoryInjector] = None


def init_memory(
    db,
    llm_generate_fn=None,
    *,
    injector_enabled: bool = True,
    injector_mode: str = "extra_user_content",
    injector_top_k: int = 3,
    injector_min_score: float = 0.1,
    injector_max_chars: int = 2000,
    summary_trigger_rounds: int = 10,
) -> None:
    """初始化 memory 模块（plugin.on_load 中调用）。"""

    global _mem_atom_store, _mem_retriever, _mem_lifecycle, _mem_processor, _mem_injector

    _mem_atom_store = AtomStore(db)
    _mem_retriever = AtomRetriever(db, _mem_atom_store)
    _mem_lifecycle = AtomLifecycleManager(_mem_atom_store)
    _mem_processor = MemoryProcessor(
        db,
        _mem_atom_store,
        llm_generate_fn=llm_generate_fn,
        summary_trigger_rounds=summary_trigger_rounds,
    )
    _mem_injector = MemoryInjector(
        _mem_retriever,
        enabled=injector_enabled,
        injection_mode=injector_mode,
        top_k=injector_top_k,
        min_score=injector_min_score,
        max_chars=injector_max_chars,
    )

    logger.info(
        f"Memory 模块初始化完成: injector_enabled={injector_enabled} mode={injector_mode}"
    )


async def init_memory_async(db) -> None:
    """异步初始化（创建 FTS 表）。"""

    if _mem_atom_store is None:
        raise RuntimeError("Memory 模块未初始化，请先调用 init_memory()")
    await _mem_atom_store.ensure_fts_table()


def set_llm_fn(fn) -> None:
    """延迟设置 LLM 调用函数。"""

    if _mem_processor is not None:
        _mem_processor.set_llm_fn(fn)


def close_memory() -> None:
    """清理 memory 模块状态。"""

    global _mem_atom_store, _mem_retriever, _mem_lifecycle, _mem_processor, _mem_injector
    _mem_atom_store = None
    _mem_retriever = None
    _mem_lifecycle = None
    _mem_processor = None
    _mem_injector = None


class MemoryApiMixin:
    """Memory API 方法集合，由 AstrBotDbPlugin 多重继承。"""

    # ==================================================================
    # 自动注入 Hook — 在 LLM 调用前自动检索记忆并注入
    # ==================================================================

    @HookHandler(
        "maisaka.planner.before_request",
        name="astrdb_memory_auto_inject",
        description="在 LLM 调用前自动检索记忆 atoms 并注入到 prompt",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,  # 在 KB 注入之后执行
        error_policy=ErrorPolicy.SKIP,
    )
    async def hook_memory_auto_inject(
        self,
        messages: Any = None,
        session_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """自动注入记忆 Hook。"""

        if _mem_injector is None:
            return {"action": "continue"}

        if not isinstance(messages, list) or not messages:
            return {"action": "continue"}

        # 提取 session_id（UMO 格式）用于记忆隔离
        # MaiBot 的 session_id 可能是 stream_id，需要适配
        sid = session_id or None

        try:
            injected, modified = await _mem_injector.inject(
                messages,
                session_id=sid,
            )
        except Exception as exc:
            logger.warning(f"Memory 自动注入失败: {exc}")
            return {"action": "continue"}

        if not injected:
            return {"action": "continue"}

        return {
            "action": "continue",
            "modified_kwargs": {
                "messages": modified,
                "session_id": session_id,
                **{k: v for k, v in kwargs.items() if k not in ("messages", "session_id")},
            },
        }

    # ==================================================================
    # 记忆 API
    # ==================================================================

    @API(
        "astrdb.memorize",
        description="主动记忆一条事实（手动写入 atom）",
        version="1",
        public=True,
    )
    async def api_memorize(
        self,
        content: str,
        *,
        parent_memory_id: str = "manual",
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        importance: float = 0.7,
        atom_type: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """主动记忆一条事实。

        Args:
            content: 记忆内容
            parent_memory_id: 父记忆 ID（默认 "manual"）
            session_id: 会话 ID
            persona_id: 人格 ID
            importance: 重要性 [0,1]
            atom_type: 指定类型（留空自动分类）
        """

        if _mem_atom_store is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        if not content or not content.strip():
            return {"success": False, "error": "content 不能为空"}

        # 自动分类
        if atom_type:
            try:
                a_type = AtomType(atom_type)
                confidence = 0.8
                event_time = None
            except ValueError:
                a_type, confidence, event_time = classify_atom(content)
        else:
            a_type, confidence, event_time = classify_atom(content)

        atom = await _mem_atom_store.insert_one(
            parent_memory_id=parent_memory_id,
            atom_type=a_type,
            content=content.strip(),
            entities=[],
            importance=importance,
            confidence=confidence,
            session_id=session_id,
            persona_id=persona_id,
            event_time_ts=event_time,
            source="manual",
        )

        return {
            "success": True,
            "atom_id": atom.atom_id,
            "atom_type": atom.atom_type,
            "ttl_days": atom.ttl_days,
            "expires_at_ts": atom.expires_at_ts,
        }

    @API(
        "astrdb.mem.search",
        description="检索记忆 atoms",
        version="1",
        public=True,
    )
    async def api_mem_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        atom_type: Optional[str] = None,
        min_score: float = 0.0,
        **_: Any,
    ) -> dict[str, Any]:
        """检索记忆。"""

        if _mem_retriever is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        a_type = None
        if atom_type:
            try:
                a_type = AtomType(atom_type)
            except ValueError:
                pass

        q = AtomSearchQuery(
            query=query,
            top_k=top_k,
            session_id=session_id,
            persona_id=persona_id,
            atom_type=a_type,
            min_score=min_score,
        )
        hits = await _mem_retriever.search(q)
        return {
            "success": True,
            "query": query,
            "count": len(hits),
            "items": [h.to_dict() for h in hits],
        }

    @API(
        "astrdb.mem.list",
        description="列出记忆 atoms",
        version="1",
        public=True,
    )
    async def api_mem_list(
        self,
        *,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        atom_type: Optional[str] = None,
        limit: int = 50,
        **_: Any,
    ) -> dict[str, Any]:
        """列出 atoms（按重要性降序）。"""

        if _mem_atom_store is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        a_type = None
        if atom_type:
            try:
                a_type = AtomType(atom_type)
            except ValueError:
                pass

        atoms = await _mem_atom_store.list_active(
            session_id=session_id,
            persona_id=persona_id,
            atom_type=a_type,
            limit=limit,
        )
        return {
            "success": True,
            "count": len(atoms),
            "items": [
                {
                    "atom_id": a.atom_id,
                    "content": a.content,
                    "atom_type": a.atom_type,
                    "importance": a.importance,
                    "confidence": a.confidence,
                    "entities": a.entities,
                    "session_id": a.session_id,
                    "persona_id": a.persona_id,
                    "parent_memory_id": a.parent_memory_id,
                    "ttl_days": a.ttl_days,
                    "expires_at_ts": a.expires_at_ts,
                    "reinforcement_count": a.reinforcement_count,
                    "created_at_ts": a.created_at_ts,
                    "last_accessed_at_ts": a.last_accessed_at_ts,
                    "source": a.source,
                }
                for a in atoms
            ],
        }

    @API(
        "astrdb.mem.stats",
        description="记忆统计",
        version="1",
        public=True,
    )
    async def api_mem_stats(self, **_: Any) -> dict[str, Any]:
        """记忆统计。"""

        if _mem_atom_store is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        by_status = await _mem_atom_store.count_by_status()
        by_type = await _mem_atom_store.count_by_type()
        total = await _mem_atom_store.total_count()

        return {
            "success": True,
            "total": total,
            "by_status": by_status,
            "by_type": by_type,
        }

    @API(
        "astrdb.mem.reinforce",
        description="强化某条记忆（增加置信度 + 续期 TTL）",
        version="1",
        public=True,
    )
    async def api_mem_reinforce(
        self,
        atom_id: str,
        *,
        new_confidence: float = 0.8,
        **_: Any,
    ) -> dict[str, Any]:
        """强化 atom。"""

        if _mem_atom_store is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        ok = await _mem_atom_store.reinforce(atom_id, new_confidence)
        return {"success": ok}

    @API(
        "astrdb.mem.forget",
        description="删除某条记忆",
        version="1",
        public=True,
    )
    async def api_mem_forget(self, atom_id: str, **_: Any) -> dict[str, Any]:
        """删除 atom。"""

        if _mem_atom_store is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        # 软删除：标记为 FORGOTTEN 并从 FTS 移除
        from sqlalchemy import text as sa_text

        async with _mem_atom_store._db.get_db() as session:
            async with session.begin():
                # 标记 FORGOTTEN
                from sqlmodel import select
                stmt = select(MemoryAtom).where(MemoryAtom.atom_id == atom_id)
                result = await session.execute(stmt)
                atom = result.scalar_one_or_none()
                if atom is None:
                    return {"success": False, "error": "atom not found"}
                atom.status = AtomStatus.FORGOTTEN.value
                # 从 FTS 移除
                await session.execute(
                    sa_text("DELETE FROM memory_atoms_fts WHERE atom_id = :aid"),
                    {"aid": atom_id},
                )
        return {"success": True}

    @API(
        "astrdb.mem.decay_now",
        description="手动触发一次衰减+清理",
        version="1",
        public=True,
    )
    async def api_mem_decay_now(self, **_: Any) -> dict[str, Any]:
        """手动触发衰减。"""

        if _mem_atom_store is None or _mem_lifecycle is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        decayed = await _mem_atom_store.apply_daily_decay(decay_rate=0.01, days=1)
        cleaned = await _mem_atom_store.cleanup_low_importance()
        lifecycle_result = await _mem_lifecycle.run_maintenance()

        return {
            "success": True,
            "decayed": decayed,
            "cleaned": cleaned,
            **lifecycle_result,
        }

    @API(
        "astrdb.mem.process",
        description="处理对话生成 atoms（高级用法，需 LLM）",
        version="1",
        public=True,
    )
    async def api_mem_process(
        self,
        messages: list[dict[str, Any]],
        *,
        parent_memory_id: str,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        is_group_chat: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """处理对话，生成总结 + atoms。"""

        if _mem_processor is None:
            return {"success": False, "error": "Memory 模块未初始化"}

        result = await _mem_processor.process_conversation(
            messages,
            parent_memory_id=parent_memory_id,
            session_id=session_id,
            persona_id=persona_id,
            is_group_chat=is_group_chat,
        )

        if result is None:
            return {"success": False, "error": "处理失败（LLM 未配置或返回无效）"}

        return {
            "success": True,
            "summary": result["summary"],
            "key_facts": result["key_facts"],
            "topics": result["topics"],
            "importance": result["importance"],
            "atoms_count": len(result["atoms"]),
            "atom_ids": [a.atom_id for a in result["atoms"]],
            "canonical_summary": result["canonical_summary"],
            "persona_summary": result["persona_summary"],
        }

    # ==================================================================
    # LLM Tools — 让 MaiBot 在对话中主动调用
    # ==================================================================

    @Tool(
        "memory_search",
        description="检索长期记忆（过往对话中提取的事实）。当用户询问之前说过的事、个人偏好、关系、计划等时调用。",
        brief_description="检索长期记忆（过往对话事实）",
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="检索查询，如 '用户喜欢什么' 或 '上次提到的会议'",
                required=True,
            ),
            ToolParameterInfo(
                name="k",
                param_type=ToolParamType.INTEGER,
                description="返回数量，默认 5",
                required=False,
            ),
        ],
    )
    async def tool_memory_search(
        self,
        query: str,
        k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        """LLM 工具：检索记忆。"""

        if _mem_retriever is None:
            return {"content": "记忆系统未初始化", "found": False}

        q = AtomSearchQuery(query=query, top_k=k)
        hits = await _mem_retriever.search(q)

        if not hits:
            return {
                "content": f"未在记忆中找到与 '{query}' 相关的内容",
                "found": False,
                "count": 0,
            }

        from .injection import format_atoms_for_injection
        return {
            "content": format_atoms_for_injection(hits),
            "found": True,
            "count": len(hits),
            "hits": [h.to_dict() for h in hits],
        }

    @Tool(
        "memory_memorize",
        description="主动记忆一条事实到长期记忆。当用户告知重要信息（偏好、计划、关系等）时调用。",
        brief_description="主动记忆事实到长期记忆",
        parameters=[
            ToolParameterInfo(
                name="content",
                param_type=ToolParamType.STRING,
                description="要记忆的事实，简洁可读，不超过 50 字",
                required=True,
            ),
            ToolParameterInfo(
                name="importance",
                param_type=ToolParamType.NUMBER,
                description="重要性 0-1，默认 0.7",
                required=False,
            ),
        ],
    )
    async def tool_memory_memorize(
        self,
        content: str,
        importance: float = 0.7,
        **_: Any,
    ) -> dict[str, Any]:
        """LLM 工具：主动记忆。"""

        if _mem_atom_store is None:
            return {"content": "记忆系统未初始化", "memorized": False}

        a_type, confidence, event_time = classify_atom(content)
        atom = await _mem_atom_store.insert_one(
            parent_memory_id="agent_tool",
            atom_type=a_type,
            content=content,
            importance=importance,
            confidence=confidence,
            event_time_ts=event_time,
            source="agent_tool",
        )

        return {
            "content": f"已记忆: {content} (类型: {atom.atom_type}, TTL: {atom.ttl_days:.1f}天)",
            "memorized": True,
            "atom_id": atom.atom_id,
            "atom_type": atom.atom_type,
            "ttl_days": atom.ttl_days,
        }


__all__ = [
    "MemoryApiMixin",
    "init_memory",
    "init_memory_async",
    "set_llm_fn",
    "close_memory",
]

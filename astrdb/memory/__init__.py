"""astrdb.memory — MemoryAtom 记忆系统

移植自 LivingMemory 的核心创新，包括：

- MemoryAtom 数据模型（细粒度事实单元 + TTL + 衰减）
- AtomClassifier 纯规则分类（5 种类型，零 LLM 调用）
- AtomStore 存储 + FTS 检索
- AtomLifecycleManager 四级状态机 + 强化机制
- DecayScheduler 每日衰减 + 自动清理
- MemoryProcessor 对话总结 + atom 抽取（调 LLM）
- AtomRetriever 多维加权 + MMR 去重 + LRU 缓存

典型用法：

    from astrdb.memory import (
        AtomStore, AtomRetriever, MemoryProcessor,
        AtomLifecycleManager, DecayScheduler,
        AtomType, AtomSearchQuery,
    )

    # 1. 初始化
    atom_store = AtomStore(db)
    await atom_store.ensure_fts_table()
    retriever = AtomRetriever(db, atom_store)
    processor = MemoryProcessor(db, atom_store, llm_generate_fn=...)
    lifecycle = AtomLifecycleManager(atom_store)
    scheduler = DecayScheduler(atom_store, lifecycle)
    scheduler.start()

    # 2. 处理对话生成 atoms
    result = await processor.process_conversation(
        messages=[{"role": "user", "content": "明天3点开会"}],
        parent_memory_id="conv-xxx",
    )

    # 3. 检索
    hits = await retriever.search(AtomSearchQuery(query="明天开会"))
"""

from .models import (
    ATOM_TYPE_CONFIG,
    AtomStatus,
    AtomType,
    DecayType,
    MemoryAtom,
    atom_status_from_str,
    atom_type_from_str,
    compute_decay_factor,
    compute_ttl,
    decay_type_from_str,
    now_ts,
)
from .atom_store import AtomStore
from .atom_classifier import classify_atom, classify_atoms, parse_event_time
from .lifecycle import AtomLifecycleManager
from .decay_scheduler import DecayScheduler
from .processor import MemoryProcessor, format_messages_for_llm
from .retriever import AtomRetriever, AtomSearchHit, AtomSearchQuery
from .injection import MemoryInjector, format_atoms_for_injection


__all__ = [
    # 模型
    "AtomType",
    "AtomStatus",
    "DecayType",
    "MemoryAtom",
    "ATOM_TYPE_CONFIG",
    "compute_ttl",
    "compute_decay_factor",
    "atom_type_from_str",
    "atom_status_from_str",
    "decay_type_from_str",
    "now_ts",
    # 存储
    "AtomStore",
    # 分类器
    "classify_atom",
    "classify_atoms",
    "parse_event_time",
    # 生命周期
    "AtomLifecycleManager",
    # 调度器
    "DecayScheduler",
    # 处理器
    "MemoryProcessor",
    "format_messages_for_llm",
    # 检索器
    "AtomRetriever",
    "AtomSearchQuery",
    "AtomSearchHit",
    # 注入
    "MemoryInjector",
    "format_atoms_for_injection",
]

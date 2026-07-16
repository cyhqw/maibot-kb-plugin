"""tests.test_memory — Memory 模块测试"""

import pytest

from astrdb import close_db, get_db, init_db
from astrdb.memory import (
    AtomLifecycleManager,
    AtomRetriever,
    AtomSearchQuery,
    AtomStore,
    AtomStatus,
    AtomType,
    DecayType,
    MemoryAtom,
    classify_atom,
    compute_decay_factor,
    compute_ttl,
    now_ts,
    parse_event_time,
)


# ----------------------------------------------------------------------
# 模型测试
# ----------------------------------------------------------------------

def test_atom_type_config():
    """每种类型有正确的 TTL 和衰减配置。"""

    cfg = AtomType.EPISODIC
    from astrdb.memory.models import ATOM_TYPE_CONFIG
    assert ATOM_TYPE_CONFIG[AtomType.EPISODIC]["base_ttl_days"] == 7.0
    assert ATOM_TYPE_CONFIG[AtomType.FACTUAL]["base_ttl_days"] == 180.0
    assert ATOM_TYPE_CONFIG[AtomType.PLANNED]["decay_type"] == DecayType.STEP


def test_compute_ttl():
    """TTL 计算公式正确。"""

    # EPISODIC, importance=0.5, reinforcement=0
    # ttl = 7.0 × (0.5 + 0.5) × 1.0 = 7.0
    ttl, decay = compute_ttl(AtomType.EPISODIC, importance=0.5, reinforcement_count=0)
    assert ttl == pytest.approx(7.0)
    assert decay == DecayType.EXPONENTIAL

    # FACTUAL, importance=1.0, reinforcement=5
    # ttl = 180 × 1.5 × 1.5 = 405
    ttl, _ = compute_ttl(AtomType.FACTUAL, importance=1.0, reinforcement_count=5)
    assert ttl == pytest.approx(405.0)

    # reinforcement 上限 0.5
    ttl, _ = compute_ttl(AtomType.FACTUAL, importance=1.0, reinforcement_count=100)
    assert ttl == pytest.approx(180 * 1.5 * 1.5)


def test_compute_decay_factor():
    """衰减因子计算。"""

    # LINEAR: 半 TTL 时衰减到 0.5
    assert compute_decay_factor(DecayType.LINEAR, 5, 10) == pytest.approx(0.5)
    assert compute_decay_factor(DecayType.LINEAR, 10, 10) == pytest.approx(0.0)
    assert compute_decay_factor(DecayType.LINEAR, 0, 10) == pytest.approx(1.0)

    # EXPONENTIAL: 0 时为 1
    assert compute_decay_factor(DecayType.EXPONENTIAL, 0, 10) == pytest.approx(1.0)

    # STEP: TTL 内为 1，过期骤降至 0.05
    assert compute_decay_factor(DecayType.STEP, 0.5, 1) == 1.0
    assert compute_decay_factor(DecayType.STEP, 2, 1) == 0.05


# ----------------------------------------------------------------------
# 分类器测试
# ----------------------------------------------------------------------

def test_classify_planned():
    """计划性记忆识别。"""

    atom_type, conf, event_time = classify_atom("明天下午3点开会讨论项目")
    assert atom_type == AtomType.PLANNED
    assert conf > 0.7
    assert event_time is not None


def test_classify_preference():
    """偏好性记忆识别。"""

    atom_type, conf, _ = classify_atom("张三喜欢吃火锅")
    assert atom_type == AtomType.PREFERENCE
    assert conf > 0.7


def test_classify_relational():
    """关系性记忆识别。"""

    atom_type, conf, _ = classify_atom("李四是王五的同事")
    assert atom_type == AtomType.RELATIONAL


def test_classify_factual():
    """事实性记忆识别。"""

    atom_type, conf, _ = classify_atom("法涅斯是原初之人")
    assert atom_type == AtomType.FACTUAL


def test_classify_episodic():
    """事件性记忆识别（有动作无时间）。"""

    atom_type, conf, _ = classify_atom("我去了超市买菜")
    assert atom_type == AtomType.EPISODIC


def test_classify_empty():
    """空内容返回 UNKNOWN。"""

    atom_type, _, _ = classify_atom("")
    assert atom_type == AtomType.UNKNOWN


def test_parse_event_time():
    """事件时间解析。"""

    ts = parse_event_time("明天开会")
    assert ts is not None
    assert ts > now_ts()

    ts = parse_event_time("没有时间词的文本")
    assert ts is None


# ----------------------------------------------------------------------
# AtomStore 测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_atom_store_crud(tmp_path):
    """Atom CRUD 测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()

    # 插入
    atom = await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.FACTUAL,
        content="法涅斯是原初之人",
        importance=0.8,
        session_id="aiocqhttp:FriendMessage:user1",
    )
    assert atom.atom_id
    assert atom.status == AtomStatus.ACTIVE.value
    assert atom.ttl_days > 0

    # 读取
    loaded = await store.get_by_id(atom.atom_id)
    assert loaded is not None
    assert loaded.content == "法涅斯是原初之人"

    # 列表
    atoms = await store.list_by_parent("conv-001")
    assert len(atoms) == 1

    await close_db()


@pytest.mark.asyncio
async def test_atom_store_fts_search(tmp_path):
    """FTS 检索测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()

    # 插入多个 atoms
    contents = [
        ("法涅斯是原初之人", AtomType.FACTUAL),
        ("张三喜欢火锅", AtomType.PREFERENCE),
        ("明天开会", AtomType.PLANNED),
    ]
    for content, a_type in contents:
        await store.insert_one(
            parent_memory_id="conv-001",
            atom_type=a_type,
            content=content,
            session_id="session1",
        )

    # 检索
    hits = await store.fts_search("法涅斯", session_id="session1")
    assert len(hits) >= 1
    assert hits[0][0]  # atom_id

    # 短查询
    hits = await store.fts_search("火锅", session_id="session1")
    assert len(hits) >= 1

    await close_db()


@pytest.mark.asyncio
async def test_atom_reinforce(tmp_path):
    """强化机制测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()

    atom = await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.FACTUAL,
        content="法涅斯是原初之人",
        importance=0.5,
        confidence=0.7,
    )

    # 强化
    ok = await store.reinforce(atom.atom_id, new_confidence=0.9)
    assert ok is True

    loaded = await store.get_by_id(atom.atom_id)
    assert loaded.reinforcement_count == 1
    # EMA: 0.7 × 0.7 + 0.9 × 0.3 = 0.49 + 0.27 = 0.76
    assert loaded.confidence == pytest.approx(0.76, rel=0.01)
    # TTL 应该增加
    assert loaded.ttl_days > 0

    await close_db()


@pytest.mark.asyncio
async def test_atom_lifecycle(tmp_path):
    """生命周期测试：过期 → 遗忘 → 清理。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()
    lifecycle = AtomLifecycleManager(store, forget_delay_days=0, purge_delay_days=0)

    # 插入一个已过期的 atom
    atom = await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.EPISODIC,
        content="测试过期",
        importance=0.5,
    )
    # 手动设置过期
    async with db.get_db() as session:
        async with session.begin():
            from sqlmodel import select
            stmt = select(MemoryAtom).where(MemoryAtom.atom_id == atom.atom_id)
            result = await session.execute(stmt)
            a = result.scalar_one()
            a.expires_at_ts = now_ts() - 1  # 已过期

    # expire_stale
    expired = await store.expire_stale_atoms()
    assert expired == 1

    # forget_expired
    forgotten = await store.forget_expired_atoms(forget_delay_days=0)
    assert forgotten == 1

    # cleanup_forgotten
    purged = await store.cleanup_forgotten(purge_delay_days=0)
    assert purged == 1

    # 确认已删除
    loaded = await store.get_by_id(atom.atom_id)
    assert loaded is None

    await close_db()


@pytest.mark.asyncio
async def test_atom_decay(tmp_path):
    """衰减测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()

    atom = await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.FACTUAL,
        content="测试衰减",
        importance=0.8,
    )

    # 衰减 1 天
    # 注意：刚插入的 atom 有 last_accessed_at_ts = now，
    # 所以 effective_rate = decay_rate × 0.5 = 0.05
    # 0.8 × (1-0.05)^1 = 0.76
    decayed = await store.apply_daily_decay(decay_rate=0.1, days=1)
    assert decayed == 1

    loaded = await store.get_by_id(atom.atom_id)
    assert loaded.importance == pytest.approx(0.76, rel=0.05)

    await close_db()


# ----------------------------------------------------------------------
# 检索器测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retriever_search(tmp_path):
    """检索器测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()
    retriever = AtomRetriever(db, store)

    # 插入
    await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.FACTUAL,
        content="法涅斯是原初之人",
        importance=0.8,
        session_id="session1",
    )
    await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.PREFERENCE,
        content="张三喜欢吃火锅",
        importance=0.7,
        session_id="session1",
    )

    # 检索
    hits = await retriever.search(
        AtomSearchQuery(query="法涅斯", top_k=3, session_id="session1")
    )
    assert len(hits) >= 1
    assert "法涅斯" in hits[0].content
    assert hits[0].atom_type == AtomType.FACTUAL.value

    # 验证分数明细
    assert hits[0].bm25_score > 0
    assert hits[0].importance_score > 0
    assert hits[0].recency_score > 0

    await close_db()


@pytest.mark.asyncio
async def test_retriever_mmr(tmp_path):
    """MMR 去重测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()
    retriever = AtomRetriever(db, store)

    # 插入相似内容
    for content in [
        "法涅斯是原初之人",
        "法涅斯是原初的那一位",
        "法涅斯生着羽翼头戴王冠",
        "张三喜欢火锅",
        "明天开会",
    ]:
        await store.insert_one(
            parent_memory_id="conv-001",
            atom_type=AtomType.FACTUAL,
            content=content,
            importance=0.7,
            session_id="session1",
        )

    # 检索（启用 MMR）
    hits_mmr = await retriever.search(
        AtomSearchQuery(query="法涅斯", top_k=3, session_id="session1", apply_mmr=True)
    )

    # 检索（禁用 MMR）
    hits_no_mmr = await retriever.search(
        AtomSearchQuery(query="法涅斯", top_k=3, session_id="session1", apply_mmr=False)
    )

    # MMR 应该返回更多样化的结果
    assert len(hits_mmr) <= 3
    assert len(hits_no_mmr) <= 3

    await close_db()


@pytest.mark.asyncio
async def test_retriever_cache(tmp_path):
    """LRU 缓存测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()
    retriever = AtomRetriever(db, store, cache_enabled=True, cache_ttl=45.0)

    await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.FACTUAL,
        content="法涅斯是原初之人",
        importance=0.8,
        session_id="session1",
    )

    # 第一次查询
    hits1 = await retriever.search(
        AtomSearchQuery(query="法涅斯", top_k=3, session_id="session1")
    )
    # 第二次相同查询（应命中缓存）
    hits2 = await retriever.search(
        AtomSearchQuery(query="法涅斯", top_k=3, session_id="session1")
    )

    assert len(hits1) == len(hits2)
    # 缓存应该有 1 条
    assert len(retriever._cache) == 1

    # 清空缓存
    retriever.invalidate_cache()
    assert len(retriever._cache) == 0

    await close_db()


# ----------------------------------------------------------------------
# 注入测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_injection(tmp_path):
    """记忆注入测试。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    store = AtomStore(db)
    await store.ensure_fts_table()
    retriever = AtomRetriever(db, store)

    from astrdb.memory import MemoryInjector

    injector = MemoryInjector(
        retriever,
        enabled=True,
        injection_mode="extra_user_content",
        top_k=3,
        min_score=0.0,
    )

    # 插入记忆
    await store.insert_one(
        parent_memory_id="conv-001",
        atom_type=AtomType.PREFERENCE,
        content="用户喜欢吃火锅",
        importance=0.8,
        session_id="session1",
    )

    # 注入 — 用包含"喜欢吃"的查询，确保 FTS 命中
    messages = [{"role": "user", "content": "用户喜欢吃什么"}]
    injected, modified = await injector.inject(messages, session_id="session1")

    assert injected is True
    assert len(modified) == 2  # 原始 + 注入
    assert "Memory-Reference" in modified[1]["content"]

    await close_db()

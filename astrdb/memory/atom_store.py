"""astrdb.memory.atom_store

MemoryAtom 存储层 — CRUD + FTS 全文检索 + 衰减/强化/过期清理。

移植自 LivingMemory `storage/atom_store.py`，简化为同步 SQLModel 操作。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlmodel import select

from ..database import AstrBotDatabase
from .models import (
    AtomStatus,
    AtomType,
    MemoryAtom,
    compute_ttl,
    now_ts,
)


logger = logging.getLogger("astrdb.memory.atom_store")


class AtomStore:
    """MemoryAtom 存储与检索。"""

    def __init__(self, db: AstrBotDatabase) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # 创建 FTS 表（在 database.initialize 时已创建 kb_chunks_fts，
    # 这里再创建 memory_atoms_fts）
    # ------------------------------------------------------------------

    async def ensure_fts_table(self) -> None:
        """创建 memory_atoms_fts 虚拟表（如果不存在）。"""

        async with self._db.engine.begin() as conn:
            await conn.execute(text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_atoms_fts USING fts5(
                    atom_id UNINDEXED,
                    content,
                    entities_text,
                    tokenize = 'trigram'
                )
                """
            ))

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    async def insert_atoms(self, atoms: list[MemoryAtom]) -> int:
        """批量插入 atoms，同步到 FTS。"""

        if not atoms:
            return 0

        async def _op(session):
            for a in atoms:
                # 填充时间戳
                ts = now_ts()
                if a.created_at_ts == 0:
                    a.created_at_ts = ts
                if a.last_accessed_at_ts == 0:
                    a.last_accessed_at_ts = ts
                # 计算 TTL 和 expires_at
                ttl, decay = compute_ttl(
                    AtomType(a.atom_type) if a.atom_type else AtomType.UNKNOWN,
                    a.importance,
                    a.reinforcement_count,
                )
                if a.ttl_days == 30.0:  # 默认值，覆盖
                    a.ttl_days = ttl
                if a.decay_type == "exponential":  # 默认值，覆盖
                    a.decay_type = decay.value
                if a.expires_at_ts == 0:
                    a.expires_at_ts = a.created_at_ts + a.ttl_days * 86400

                session.add(a)

            await session.flush()

            # 同步 FTS
            for a in atoms:
                await session.execute(
                    text(
                        "INSERT INTO memory_atoms_fts(atom_id, content, entities_text) "
                        "VALUES (:aid, :content, :entities)"
                    ),
                    {
                        "aid": a.atom_id,
                        "content": a.content,
                        "entities": " ".join(a.entities or []),
                    },
                )
            return len(atoms)

        return await self._db._run_in_tx(_op)

    async def insert_one(
        self,
        *,
        parent_memory_id: str,
        atom_type: AtomType,
        content: str,
        entities: Optional[list[str]] = None,
        importance: float = 0.5,
        confidence: float = 0.7,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        event_time_ts: Optional[float] = None,
        source: str = "auto",
        metadata_json: Optional[dict[str, Any]] = None,
    ) -> MemoryAtom:
        """插入单个 atom。"""

        atom = MemoryAtom(
            parent_memory_id=parent_memory_id,
            atom_type=atom_type.value,
            content=content,
            entities=entities or [],
            importance=importance,
            confidence=confidence,
            session_id=session_id,
            persona_id=persona_id,
            event_time_ts=event_time_ts,
            source=source,
            metadata_json=metadata_json or {},
        )
        await self.insert_atoms([atom])
        return atom

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    async def get_by_id(self, atom_id: str) -> Optional[MemoryAtom]:
        async with self._db.get_db() as session:
            stmt = select(MemoryAtom).where(MemoryAtom.atom_id == atom_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_by_parent(self, parent_memory_id: str) -> list[MemoryAtom]:
        async with self._db.get_db() as session:
            stmt = (
                select(MemoryAtom)
                .where(MemoryAtom.parent_memory_id == parent_memory_id)
                .order_by(MemoryAtom.id.asc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def list_active(
        self,
        *,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        atom_type: Optional[AtomType] = None,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        """列出活跃 atoms。"""

        async with self._db.get_db() as session:
            stmt = (
                select(MemoryAtom)
                .where(MemoryAtom.status == AtomStatus.ACTIVE.value)
                .order_by(MemoryAtom.importance.desc(), MemoryAtom.created_at_ts.desc())
                .limit(limit)
            )
            if session_id:
                stmt = stmt.where(MemoryAtom.session_id == session_id)
            if persona_id:
                stmt = stmt.where(MemoryAtom.persona_id == persona_id)
            if atom_type:
                stmt = stmt.where(MemoryAtom.atom_type == atom_type.value)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # 检索（FTS）
    # ------------------------------------------------------------------

    async def fts_search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """FTS 检索 atoms。

        Returns:
            List of (atom_id, bm25_score)
        """

        if not query or not query.strip():
            return []

        q = query.strip()
        # 短查询回退 LIKE
        if len(q) < 3:
            sql = """
                SELECT a.atom_id, 1.0 AS score
                FROM memory_atoms a
                JOIN memory_atoms_fts f ON a.atom_id = f.atom_id
                WHERE (f.content LIKE :pattern OR f.entities_text LIKE :pattern)
                  AND a.status = 'active'
            """
            params: dict[str, Any] = {"pattern": f"%{q}%"}
            if session_id:
                sql += " AND a.session_id = :sid"
                params["sid"] = session_id
            if persona_id:
                sql += " AND a.persona_id = :pid"
                params["pid"] = persona_id
            sql += " LIMIT :limit"
            params["limit"] = limit
            async with self._db.get_db() as session:
                result = await session.execute(text(sql), params)
                return [(row[0], float(row[1])) for row in result.fetchall()]

        safe_query = q.replace('"', '""')
        # 拆分为多个 3-gram 子串，用 OR 连接提升召回
        # trigram 分词器对长查询会把整个查询当成 phrase 匹配，
        # 拆成 3-gram 后任一命中即可召回
        if len(q) >= 3:
            grams = []
            for i in range(len(q) - 2):
                gram = q[i : i + 3].replace('"', '""')
                if gram not in grams:
                    grams.append(gram)
            if grams:
                # 用 OR 连接，每个 gram 用引号包成 phrase
                match_expr = " OR ".join(f'"{g}"' for g in grams)
            else:
                match_expr = f'"{safe_query}"'
        else:
            match_expr = f'"{safe_query}"'

        async with self._db.get_db() as session:
            try:
                sql = """
                    SELECT a.atom_id, bm25(memory_atoms_fts) AS score
                    FROM memory_atoms a
                    JOIN memory_atoms_fts f ON a.atom_id = f.atom_id
                    WHERE memory_atoms_fts MATCH :q AND a.status = 'active'
                """
                params: dict[str, Any] = {"q": match_expr}
                if session_id:
                    sql += " AND a.session_id = :sid"
                    params["sid"] = session_id
                if persona_id:
                    sql += " AND a.persona_id = :pid"
                    params["pid"] = persona_id
                sql += " ORDER BY score LIMIT :limit"
                params["limit"] = limit
                result = await session.execute(text(sql), params)
                return [(row[0], -float(row[1])) for row in result.fetchall()]
            except Exception as exc:
                logger.warning(f"Atom FTS 查询失败: {exc}")
                return []

    # ------------------------------------------------------------------
    # 更新（访问 / 强化）
    # ------------------------------------------------------------------

    async def touch(self, atom_id: str) -> bool:
        """更新最后访问时间。"""

        async def _op(session):
            stmt = select(MemoryAtom).where(MemoryAtom.atom_id == atom_id)
            result = await session.execute(stmt)
            a = result.scalar_one_or_none()
            if a is None:
                return False
            a.last_accessed_at_ts = now_ts()
            return True

        return await self._db._run_in_tx(_op)

    async def reinforce(
        self,
        atom_id: str,
        new_confidence: float = 0.8,
    ) -> bool:
        """强化 atom：增加 reinforcement_count，EMA 更新 confidence，重算 TTL 续期。

        EMA 公式（移植自 LivingMemory）：
            confidence = old × 0.7 + new × 0.3
        """

        async def _op(session):
            stmt = select(MemoryAtom).where(MemoryAtom.atom_id == atom_id)
            result = await session.execute(stmt)
            a = result.scalar_one_or_none()
            if a is None:
                return False

            a.reinforcement_count += 1
            a.confidence = a.confidence * 0.7 + new_confidence * 0.3
            a.last_reinforced_at_ts = now_ts()

            # 重算 TTL 并续期
            new_ttl, _ = compute_ttl(
                AtomType(a.atom_type) if a.atom_type else AtomType.UNKNOWN,
                a.importance,
                a.reinforcement_count,
            )
            a.ttl_days = new_ttl
            a.expires_at_ts = now_ts() + new_ttl * 86400

            # 如果是 EXPIRED/FORGOTTEN 状态，重新激活
            if a.status in (AtomStatus.EXPIRED.value, AtomStatus.FORGOTTEN.value):
                a.status = AtomStatus.ACTIVE.value

            return True

        return await self._db._run_in_tx(_op)

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    async def expire_stale_atoms(self) -> int:
        """把过期但未标记的 atoms 标记为 EXPIRED。"""

        ts = now_ts()
        async def _op(session):
            stmt = select(MemoryAtom).where(
                MemoryAtom.status == AtomStatus.ACTIVE.value,
                MemoryAtom.expires_at_ts < ts,
            )
            result = await session.execute(stmt)
            atoms = list(result.scalars().all())
            for a in atoms:
                a.status = AtomStatus.EXPIRED.value
            return len(atoms)

        return await self._db._run_in_tx(_op)

    async def forget_expired_atoms(self, forget_delay_days: float = 7.0) -> int:
        """EXPIRED + forget_delay → FORGOTTEN（从 FTS 移除）。"""

        threshold = now_ts() - forget_delay_days * 86400
        async def _op(session):
            stmt = select(MemoryAtom).where(
                MemoryAtom.status == AtomStatus.EXPIRED.value,
                MemoryAtom.expires_at_ts < threshold,
            )
            result = await session.execute(stmt)
            atoms = list(result.scalars().all())
            for a in atoms:
                a.status = AtomStatus.FORGOTTEN.value
                # 从 FTS 移除
                await session.execute(
                    text("DELETE FROM memory_atoms_fts WHERE atom_id = :aid"),
                    {"aid": a.atom_id},
                )
            return len(atoms)

        return await self._db._run_in_tx(_op)

    async def cleanup_forgotten(self, purge_delay_days: float = 30.0) -> int:
        """FORGOTTEN + purge_delay → 物理删除。"""

        threshold = now_ts() - purge_delay_days * 86400
        async def _op(session):
            stmt = select(MemoryAtom).where(
                MemoryAtom.status == AtomStatus.FORGOTTEN.value,
                MemoryAtom.expires_at_ts < threshold,
            )
            result = await session.execute(stmt)
            atoms = list(result.scalars().all())
            for a in atoms:
                await session.delete(a)
                # 清理 FTS 残留
                await session.execute(
                    text("DELETE FROM memory_atoms_fts WHERE atom_id = :aid"),
                    {"aid": a.atom_id},
                )
            return len(atoms)

        return await self._db._run_in_tx(_op)

    # ------------------------------------------------------------------
    # 衰减
    # ------------------------------------------------------------------

    async def apply_daily_decay(
        self,
        decay_rate: float = 0.01,
        days: int = 1,
    ) -> int:
        """批量衰减：importance × (1-effective_decay_rate)^days。

        访问强化降权：最近 access_decay_window_days 内访问过的 atom 衰减率减半。

        Returns:
            衰减的 atom 数
        """

        access_window_start = now_ts() - 30 * 86400  # 30 天窗口
        async def _op(session):
            stmt = select(MemoryAtom).where(
                MemoryAtom.status == AtomStatus.ACTIVE.value
            )
            result = await session.execute(stmt)
            atoms = list(result.scalars().all())
            for a in atoms:
                # 访问强化降权
                if a.last_accessed_at_ts >= access_window_start:
                    effective_rate = decay_rate * 0.5
                else:
                    effective_rate = decay_rate

                decay_factor = (1 - effective_rate) ** days
                a.importance = max(0.01, a.importance * decay_factor)

                # 重要性低于阈值 → DORMANT
                if a.importance < 0.1:
                    a.status = AtomStatus.DORMANT.value

            return len(atoms)

        return await self._db._run_in_tx(_op)

    async def cleanup_low_importance(
        self,
        days_threshold: int = 30,
        importance_threshold: float = 0.3,
    ) -> int:
        """删除超过 N 天且重要性低于阈值的 atoms。"""

        threshold_ts = now_ts() - days_threshold * 86400
        async def _op(session):
            stmt = select(MemoryAtom).where(
                MemoryAtom.created_at_ts < threshold_ts,
                MemoryAtom.importance < importance_threshold,
            )
            result = await session.execute(stmt)
            atoms = list(result.scalars().all())
            for a in atoms:
                await session.delete(a)
                await session.execute(
                    text("DELETE FROM memory_atoms_fts WHERE atom_id = :aid"),
                    {"aid": a.atom_id},
                )
            return len(atoms)

        return await self._db._run_in_tx(_op)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    async def count_by_status(self) -> dict[str, int]:
        """按状态统计 atom 数。"""

        async with self._db.get_db() as session:
            result = await session.execute(
                text("SELECT status, COUNT(*) FROM memory_atoms GROUP BY status")
            )
            return {row[0]: int(row[1]) for row in result.fetchall()}

    async def count_by_type(self) -> dict[str, int]:
        """按类型统计 atom 数。"""

        async with self._db.get_db() as session:
            result = await session.execute(
                text("SELECT atom_type, COUNT(*) FROM memory_atoms GROUP BY atom_type")
            )
            return {row[0]: int(row[1]) for row in result.fetchall()}

    async def total_count(self) -> int:
        return await self._db.count_rows("memory_atoms")


__all__ = ["AtomStore"]

"""astrdb.memory.lifecycle

Atom 生命周期管理器 — 定期维护 atom 状态。

移植自 LivingMemory `core/managers/atom_lifecycle_manager.py`。

状态机：
    ACTIVE ──[expires_at 到期]──→ EXPIRED ──[+7天]──→ FORGOTTEN ──[+30天]──→ 物理删除

强化机制（reinforce）：
    新 atom 内容分词 → FTS 搜现有 atoms → Jaccard 相似度 ≥ 0.6 → reinforce
"""

from __future__ import annotations

import logging
from typing import Optional

from ..database import AstrBotDatabase
from .atom_store import AtomStore
from .models import AtomType, now_ts


logger = logging.getLogger("astrdb.memory.lifecycle")


class AtomLifecycleManager:
    """Atom 生命周期管理。

    每 24 小时执行一次维护 pass：
    1. expire_stale_atoms: ACTIVE → EXPIRED
    2. forget_expired_atoms: EXPIRED + 7d → FORGOTTEN
    3. cleanup_forgotten: FORGOTTEN + 30d → 物理删除
    """

    def __init__(
        self,
        atom_store: AtomStore,
        *,
        maintenance_interval_hours: float = 24.0,
        forget_delay_days: float = 7.0,
        purge_delay_days: float = 30.0,
    ) -> None:
        self._store = atom_store
        self._interval = maintenance_interval_hours
        self._forget_delay = forget_delay_days
        self._purge_delay = purge_delay_days

    async def run_maintenance(self) -> dict[str, int]:
        """执行一次维护 pass。"""

        result: dict[str, int] = {}
        result["expired"] = await self._store.expire_stale_atoms()
        result["forgotten"] = await self._store.forget_expired_atoms(self._forget_delay)
        result["purged"] = await self._store.cleanup_forgotten(self._purge_delay)
        logger.info(f"Atom 维护完成: {result}")
        return result

    async def reinforce_similar(self, content: str, threshold: float = 0.6) -> int:
        """查找与 content 相似的 atoms 并强化。

        用简单 token Jaccard 相似度（避免引入 jieba）。

        Returns:
            强化的 atom 数
        """

        if not content or not content.strip():
            return 0

        # 简单分词：中文按字，英文按词
        tokens = _tokenize(content)
        if not tokens:
            return 0

        # FTS 搜索候选
        candidates = await self._store.fts_search(content, limit=20)
        if not candidates:
            return 0

        reinforced = 0
        for atom_id, _ in candidates:
            atom = await self._store.get_by_id(atom_id)
            if atom is None:
                continue
            atom_tokens = _tokenize(atom.content)
            sim = _jaccard(tokens, atom_tokens)
            if sim >= threshold:
                await self._store.reinforce(atom_id, new_confidence=0.8)
                reinforced += 1

        return reinforced


def _tokenize(text: str) -> set[str]:
    """简单分词：CJK 按字，其他按空白。"""

    tokens = set()
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.add(ch)
        elif ch.isalnum():
            # 累积连续字母数字
            pass
    # 简单处理：英文按空白分
    for word in text.split():
        if word and not all("\u4e00" <= c <= "\u9fff" for c in word):
            tokens.add(word.lower())
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard 相似度。"""

    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


__all__ = ["AtomLifecycleManager"]

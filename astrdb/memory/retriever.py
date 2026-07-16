"""astrdb.memory.retriever

Atom 检索器 — 多维加权 + MMR 去重 + LRU 缓存。

移植自 LivingMemory `core/retrieval/hybrid_retriever.py` 的核心思想：
- final_score = bm25_score × 0.5 + importance × 0.25 + recency × 0.25
- MMR 去重：避免返回内容相似的 atoms
- LRU 缓存：相同查询 45s 内复用结果
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from sqlmodel import select

from ..database import AstrBotDatabase
from .atom_store import AtomStore
from .models import (
    AtomStatus,
    AtomType,
    DecayType,
    MemoryAtom,
    compute_decay_factor,
    decay_type_from_str,
    now_ts,
)


logger = logging.getLogger("astrdb.memory.retriever")


@dataclass
class AtomSearchHit:
    """单条检索结果。"""

    atom_id: str
    score: float
    content: str
    atom_type: str
    importance: float
    confidence: float
    entities: list[str]
    session_id: Optional[str]
    persona_id: Optional[str]
    created_at_ts: float
    last_accessed_at_ts: float
    expires_at_ts: float
    parent_memory_id: str

    # 分数明细（用于调试）
    bm25_score: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0
    decay_factor: float = 1.0

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id,
            "score": self.score,
            "content": self.content,
            "atom_type": self.atom_type,
            "importance": self.importance,
            "confidence": self.confidence,
            "entities": self.entities,
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "parent_memory_id": self.parent_memory_id,
            "score_breakdown": {
                "bm25": self.bm25_score,
                "importance": self.importance_score,
                "recency": self.recency_score,
                "decay": self.decay_factor,
            },
        }


@dataclass
class AtomSearchQuery:
    """检索请求。"""

    query: str
    top_k: int = 5
    session_id: Optional[str] = None
    persona_id: Optional[str] = None
    atom_type: Optional[AtomType] = None
    min_score: float = 0.0
    # 加权
    bm25_weight: float = 0.5
    importance_weight: float = 0.25
    recency_weight: float = 0.25
    # MMR
    mmr_lambda: float = 0.7
    apply_mmr: bool = True


class AtomRetriever:
    """Atom 检索器。"""

    def __init__(
        self,
        db: AstrBotDatabase,
        atom_store: AtomStore,
        *,
        cache_enabled: bool = True,
        cache_ttl: float = 45.0,
        cache_max_size: int = 256,
    ) -> None:
        self._db = db
        self._store = atom_store
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl
        self._cache_max_size = cache_max_size
        self._cache: OrderedDict[tuple, tuple[float, list[AtomSearchHit]]] = OrderedDict()

    async def search(self, q: AtomSearchQuery) -> list[AtomSearchHit]:
        """执行检索。"""

        if not q.query or not q.query.strip():
            return []

        # 缓存检查
        cache_key = (q.query, q.top_k, q.session_id, q.persona_id, q.atom_type)
        if self._cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                ts, hits = cached
                if time.time() - ts < self._cache_ttl:
                    return hits

        # 1. BM25 检索
        bm25_hits = await self._store.fts_search(
            q.query,
            session_id=q.session_id,
            persona_id=q.persona_id,
            limit=q.top_k * 4,  # 多召回一些用于 MMR
        )

        if not bm25_hits:
            return []

        # 2. 加载完整 atom 信息
        atom_ids = [aid for aid, _ in bm25_hits]
        atoms = await self._load_atoms(atom_ids)
        if not atoms:
            return []

        # atom_type 过滤
        if q.atom_type:
            atoms = [a for a in atoms if a.atom_type == q.atom_type.value]

        # 3. 计算多维分数
        hits = []
        bm25_map = {aid: score for aid, score in bm25_hits}
        max_bm25 = max(bm25_map.values()) if bm25_map else 1.0

        for a in atoms:
            bm25_score = bm25_map.get(a.atom_id, 0.0)
            # 归一化 bm25 到 [0, 1]
            bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0

            # importance 直接用
            importance_score = max(0.0, min(1.0, a.importance))

            # recency：基于 max(created, last_accessed) 指数衰减
            ref_ts = max(a.created_at_ts, a.last_accessed_at_ts)
            days_old = max(0.0, (now_ts() - ref_ts) / 86400)
            recency_score = float(math.exp(-0.05 * days_old))  # 半衰期 ~14 天

            # decay factor（基于 TTL）
            decay = compute_decay_factor(
                decay_type_from_str(a.decay_type),
                days_old,
                a.ttl_days,
            )

            # 综合分数
            final_score = (
                q.bm25_weight * bm25_norm
                + q.importance_weight * importance_score
                + q.recency_weight * recency_score
            ) * decay

            hits.append(
                AtomSearchHit(
                    atom_id=a.atom_id,
                    score=final_score,
                    content=a.content,
                    atom_type=a.atom_type,
                    importance=a.importance,
                    confidence=a.confidence,
                    entities=a.entities or [],
                    session_id=a.session_id,
                    persona_id=a.persona_id,
                    created_at_ts=a.created_at_ts,
                    last_accessed_at_ts=a.last_accessed_at_ts,
                    expires_at_ts=a.expires_at_ts,
                    parent_memory_id=a.parent_memory_id,
                    bm25_score=bm25_norm,
                    importance_score=importance_score,
                    recency_score=recency_score,
                    decay_factor=decay,
                )
            )

        # 4. 排序
        hits.sort(key=lambda h: h.score, reverse=True)

        # 5. MMR 去重
        if q.apply_mmr and len(hits) > q.top_k:
            hits = self._mmr_rerank(hits, q.top_k, q.mmr_lambda)
        else:
            hits = hits[: q.top_k]

        # 6. 过滤低分
        hits = [h for h in hits if h.score >= q.min_score]

        # 7. 异步更新访问时间（fire-and-forget）
        for h in hits:
            try:
                await self._store.touch(h.atom_id)
            except Exception:
                pass

        # 8. 写入缓存
        if self._cache_enabled:
            self._cache[cache_key] = (time.time(), hits)
            while len(self._cache) > self._cache_max_size:
                self._cache.popitem(last=False)

        return hits

    async def _load_atoms(self, atom_ids: list[str]) -> list[MemoryAtom]:
        """批量加载 atoms。"""

        if not atom_ids:
            return []
        async with self._db.get_db() as session:
            stmt = select(MemoryAtom).where(
                MemoryAtom.atom_id.in_(atom_ids),
                MemoryAtom.status == AtomStatus.ACTIVE.value,
            )
            result = await session.execute(stmt)
            atoms = list(result.scalars().all())
            # 保持 atom_ids 顺序
            atom_map = {a.atom_id: a for a in atoms}
            return [atom_map[aid] for aid in atom_ids if aid in atom_map]

    def _mmr_rerank(
        self,
        hits: list[AtomSearchHit],
        top_k: int,
        lam: float,
    ) -> list[AtomSearchHit]:
        """MMR（Maximal Marginal Relevance）去重。

        score_mmr = lam × relevance - (1-lam) × max_sim_to_selected
        """

        if not hits or top_k <= 0:
            return []

        selected: list[AtomSearchHit] = [hits[0]]  # 第一个直接选
        remaining = hits[1:]

        while len(selected) < top_k and remaining:
            best = None
            best_score = -1.0
            for h in remaining:
                # 与已选中的最大相似度（Jaccard 词袋）
                max_sim = max(
                    _jaccard_tokens(h.content, s.content) for s in selected
                )
                mmr = lam * h.score - (1 - lam) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best = h
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)

        return selected

    def invalidate_cache(self) -> None:
        """清空缓存（写入/删除/衰减后调用）。"""

        self._cache.clear()


def _jaccard_tokens(a: str, b: str) -> float:
    """Jaccard 词袋相似度（中文按字，英文按词）。"""

    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _tokenize(text: str) -> set[str]:
    """简单分词。"""

    tokens = set()
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.add(ch)
    for word in text.split():
        if word and not all("\u4e00" <= c <= "\u9fff" for c in word):
            tokens.add(word.lower())
    return tokens


__all__ = ["AtomRetriever", "AtomSearchQuery", "AtomSearchHit"]

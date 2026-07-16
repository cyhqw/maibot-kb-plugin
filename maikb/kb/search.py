"""kb.search

混合检索 + RRF（Reciprocal Rank Fusion）融合。

设计：
- 向量检索：cosine similarity top-K
- BM25 检索：SQLite FTS5 top-K
- 融合：RRF 公式 score = sum(1 / (k + rank))，避免两路分数尺度不一致问题
- 最终 top-N 输出，附 metadata（来源、标题路径、原文片段）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..database import MaiKBDatabase
from .vector_store import VectorIndex


logger = logging.getLogger("maikb.kb.search")


@dataclass
class SearchHit:
    """单条检索结果。"""

    chunk_id: str
    score: float
    content: str
    heading: str
    title_path: list[str] = field(default_factory=list)
    file_id: Optional[str] = None
    source_name: Optional[str] = None  # 文件名，便于用户识别来源
    vector_score: float = 0.0
    bm25_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
            "heading": self.heading,
            "title_path": self.title_path,
            "file_id": self.file_id,
            "source_name": self.source_name,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
        }


@dataclass
class SearchQuery:
    """检索请求。"""

    query: str
    top_k: int = 5
    """最终返回的结果数。"""

    vector_top_k: int = 20
    """向量检索召回数。"""

    bm25_top_k: int = 20
    """BM25 检索召回数。"""

    rrf_k: int = 60
    """RRF 公式中的 k 值，标准值 60。"""

    min_score: float = 0.0
    """最低融合分数阈值。"""

    category: Optional[str] = None
    """按文件 category 过滤。"""

    file_ids: Optional[list[str]] = None
    """限定在某几个文件内检索。"""

    use_vector: bool = True
    use_bm25: bool = True
    """分别控制是否走向量/BM25 路径。"""


class HybridSearcher:
    """混合检索器。

    用法：
        searcher = HybridSearcher(db, vector_index, embedder)
        hits = await searcher.search(SearchQuery(query="法涅斯是什么"))
    """

    def __init__(
        self,
        db: MaiKBDatabase,
        vector_index: VectorIndex,
        embedder,  # Embedder protocol
    ) -> None:
        self._db = db
        self._index = vector_index
        self._embedder = embedder

    async def search(self, q: SearchQuery) -> list[SearchHit]:
        """执行混合检索。"""

        if not q.query or not q.query.strip():
            return []

        # ---------- 向量路 ----------
        vector_hits: list[tuple[str, float]] = []
        if q.use_vector and self._index.size > 0:
            try:
                query_vec = await self._embedder.embed(q.query)
                vector_hits = self._index.search(
                    query_vec, top_k=q.vector_top_k, min_score=0.0
                )
            except Exception as exc:
                logger.warning(f"向量检索失败: {exc}")
                vector_hits = []

        # ---------- BM25 路 ----------
        bm25_hits: list[tuple[str, float]] = []
        if q.use_bm25:
            try:
                bm25_hits = await self._db.fts_search(q.query, limit=q.bm25_top_k)
            except Exception as exc:
                logger.warning(f"BM25 检索失败: {exc}")
                bm25_hits = []

        # ---------- RRF 融合 ----------
        fused = self._rrf_fuse(vector_hits, bm25_hits, k=q.rrf_k)

        if not fused:
            return []

        # ---------- 取 top-N ----------
        fused.sort(key=lambda x: x[1], reverse=True)
        top_chunk_ids = [cid for cid, _ in fused[: q.top_k * 2]]  # 多取一些备过滤

        # ---------- 查完整 chunk ----------
        chunks = await self._db.get_chunks_by_ids(top_chunk_ids)
        chunk_map = {c.chunk_id: c for c in chunks}

        # ---------- 过滤 ----------
        # 应用 category / file_ids 过滤
        # 需要查文件元数据
        file_ids_needed = {c.file_id for c in chunks}
        files_map: dict[str, str] = {}  # file_id → file_name
        if file_ids_needed:
            for fid in file_ids_needed:
                f = await self._db.get_kb_file_by_id(fid)
                if f:
                    files_map[fid] = f.file_name

        results: list[SearchHit] = []
        for cid, score in fused:
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue

            # 过滤
            if q.file_ids and chunk.file_id not in q.file_ids:
                continue
            if q.category:
                f = await self._db.get_kb_file_by_id(chunk.file_id)
                if not f or f.category != q.category:
                    continue
            if score < q.min_score:
                continue

            # 找向量/BM25 原始分数
            v_score = next((s for cid_, s in vector_hits if cid_ == cid), 0.0)
            b_score = next((s for cid_, s in bm25_hits if cid_ == cid), 0.0)

            results.append(
                SearchHit(
                    chunk_id=cid,
                    score=score,
                    content=chunk.content,
                    heading=chunk.heading or "",
                    title_path=chunk.title_path or [],
                    file_id=chunk.file_id,
                    source_name=files_map.get(chunk.file_id, ""),
                    vector_score=v_score,
                    bm25_score=b_score,
                )
            )

            if len(results) >= q.top_k:
                break

        return results

    @staticmethod
    def _rrf_fuse(
        vector_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """RRF 融合两路检索结果。

        score(d) = sum(1 / (k + rank_i(d)))

        优点：不需要两路分数归一化，只用排名。
        """

        scores: dict[str, float] = {}

        for rank, (cid, _) in enumerate(vector_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        for rank, (cid, _) in enumerate(bm25_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        return [(cid, s) for cid, s in scores.items()]


__all__ = ["SearchHit", "SearchQuery", "HybridSearcher"]

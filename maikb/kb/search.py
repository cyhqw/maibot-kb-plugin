"""kb.search

混合检索 + RRF（Reciprocal Rank Fusion）融合。

设计：
- 向量检索：cosine similarity top-K
- BM25 检索：SQLite FTS5 top-K
- 融合：RRF 公式 score = sum(1 / (k + rank))，避免两路分数尺度不一致问题
- 最终 top-N 输出，附 metadata（来源、标题路径、原文片段）

融合模式（fusion_mode）：
- "hybrid"        : 标准 RRF，向量与 BM25 都参与排序（原始行为）
- "vector_ranked" : BM25 仅参与召回，不影响排序。向量路给出完整 RRF 分数，
                    BM25 命中但向量路未命中的 chunk 以 0 分进入候选池，
                    仅在向量候选不足时填补 top-N。对中文专有名词场景更稳。
- "vector_only"   : 完全忽略 BM25（等价于 use_bm25=False）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from ..database import MaiKBDatabase
from .vector_store import VectorIndex


# 融合模式类型
FusionMode = Literal["hybrid", "vector_ranked", "vector_only"]


logger = logging.getLogger("maikb.kb.search")

# 省略号、单字符标点、纯对话标签等低信息量模式
_ELLIPSIS_RE = re.compile(r"[…\.{2,}。、，！？；：""''（）()\[\]【】—\-]")
_DIALOGUE_TAG_RE = re.compile(r"^[^：:]{1,10}[：:]")  # "角色名：" 开头的对话行


def _info_density(content: str) -> float:
    """计算 chunk 的信息密度分数（0.0 ~ 1.0）。

    惩罚以下情况：
    - 大量省略号/标点（"……" "。" 等）
    - 短行对话碎片（"角色：……"）
    - 实质文本占比低

    用于对 RRF 分数加权，避免"角色名出现多次但内容空洞"的 chunk 被高估。
    """

    if not content or not content.strip():
        return 0.0

    total = len(content)
    # 去掉标点和省略号后的实质字符数
    meaningful = _ELLIPSIS_RE.sub("", content)
    meaningful_chars = len(meaningful.strip())
    if meaningful_chars == 0:
        return 0.0

    # 实质字符占比（基础分）
    ratio = meaningful_chars / total

    # 检查对话碎片密度：如果大部分行是 "角色名：短句" 格式，降低分数
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if lines:
        dialogue_lines = sum(1 for l in lines if _DIALOGUE_TAG_RE.match(l))
        dialogue_ratio = dialogue_lines / len(lines)
        # 对话占比越高，信息密度越低（对话碎片通常是角色念台词，缺乏描述性信息）
        ratio *= (1.0 - dialogue_ratio * 0.5)

    # 检查省略号密度
    ellipsis_count = content.count("…") + content.count("...")
    if ellipsis_count > 0:
        ellipsis_ratio = min(ellipsis_count / max(len(lines), 1), 1.0)
        ratio *= (1.0 - ellipsis_ratio * 0.4)

    return max(0.1, min(1.0, ratio))


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

    fusion_mode: FusionMode = "vector_ranked"
    """融合模式：
    - "hybrid"        : 标准 RRF，向量与 BM25 都参与排序
    - "vector_ranked" : BM25 仅参与召回，不影响排序（推荐，默认）
    - "vector_only"   : 完全忽略 BM25
    """


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

        # ---------- 融合模式与 use_bm25 协调 ----------
        # fusion_mode="vector_only" 等价于关掉 BM25
        if q.fusion_mode == "vector_only":
            q.use_bm25 = False

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
        fused = self._rrf_fuse(
            vector_hits, bm25_hits, k=q.rrf_k, mode=q.fusion_mode
        )

        if not fused:
            return []

        # ---------- 取 top-N（先多取一些，用信息密度重排）----------
        fused.sort(key=lambda x: x[1], reverse=True)
        top_chunk_ids = [cid for cid, _ in fused[: q.top_k * 3]]  # 多取一些备重排

        # ---------- 查完整 chunk ----------
        chunks = await self._db.get_chunks_by_ids(top_chunk_ids)
        chunk_map = {c.chunk_id: c for c in chunks}

        # ---------- 查文件元数据 ----------
        file_ids_needed = {c.file_id for c in chunks}
        files_map: dict[str, str] = {}
        if file_ids_needed:
            for fid in file_ids_needed:
                f = await self._db.get_kb_file_by_id(fid)
                if f:
                    files_map[fid] = f.file_name

        # ---------- 信息密度加权 + 过滤 ----------
        scored: list[tuple[str, float, float, float, float]] = []
        for cid, rrf_score in fused:
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            if q.file_ids and chunk.file_id not in q.file_ids:
                continue
            if q.category:
                f = await self._db.get_kb_file_by_id(chunk.file_id)
                if not f or f.category != q.category:
                    continue
            if rrf_score < q.min_score:
                continue

            v_score = next((s for cid_, s in vector_hits if cid_ == cid), 0.0)
            b_score = next((s for cid_, s in bm25_hits if cid_ == cid), 0.0)
            density = _info_density(chunk.content)
            # 信息密度作为 RRF 分数的乘数：密度 1.0 不变，密度 0.3 打三折
            adjusted_score = rrf_score * density

            scored.append((cid, adjusted_score, v_score, b_score, density))

        # 按加权后分数重新排序
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchHit] = []
        for cid, adj_score, v_score, b_score, density in scored:
            chunk = chunk_map[cid]
            results.append(
                SearchHit(
                    chunk_id=cid,
                    score=adj_score,
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
        mode: FusionMode = "vector_ranked",
    ) -> list[tuple[str, float]]:
        """RRF 融合两路检索结果。

        score(d) = sum(1 / (k + rank_i(d)))

        优点：不需要两路分数归一化，只用排名。

        Args:
            mode: 融合模式
                - "hybrid"        : 标准 RRF，向量与 BM25 都参与排序
                - "vector_ranked" : BM25 仅贡献"入场券"。向量路给出完整 RRF 分数，
                                    BM25 命中但向量路未命中的 chunk 以 0 分进入候选池，
                                    仅在向量候选不足时填补 top-N。
                - "vector_only"   : 完全忽略 BM25（应在调用前就置 use_bm25=False，
                                    此处兜底再过滤一次）
        """

        scores: dict[str, float] = {}

        # 向量路始终贡献完整 RRF 分数
        for rank, (cid, _) in enumerate(vector_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        if mode == "vector_only":
            # 完全忽略 BM25
            pass
        elif mode == "vector_ranked":
            # BM25 只贡献"入场券"，不加分
            # 向量路未命中的 BM25 chunk 以 0 分进入候选池
            for cid, _ in bm25_hits:
                if cid not in scores:
                    scores[cid] = 0.0
        else:  # "hybrid"
            # 标准 RRF：BM25 也加分
            for rank, (cid, _) in enumerate(bm25_hits):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        return [(cid, s) for cid, s in scores.items()]


__all__ = ["SearchHit", "SearchQuery", "HybridSearcher", "FusionMode"]

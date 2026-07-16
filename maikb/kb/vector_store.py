"""kb.vector_store

基于 numpy 的内存向量索引。

设计：
- 启动时从 SQLite 加载所有 chunk 的 embedding 到内存
- 检索用 numpy 矩阵乘法（cosine similarity，极快）
- 增量更新：新增/删除 chunk 时同步内存索引
- 数据量评估：1 万 chunk × 1024 维 float32 = 40MB，完全可接受

不做：
- FAISS / sqlite-vec 等外部依赖（保持零依赖）
- HNSW 等近似检索（数据量不大时暴力检索足够）
- 持久化文件（embedding 已存在 SQLite BLOB 字段，每次启动从 DB 加载即可）
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np


logger = logging.getLogger("maikb.kb.vector_store")


class VectorIndex:
    """内存向量索引（cosine similarity）。

    线程安全：所有写操作加锁，读操作无锁（numpy 切片是只读的）。
    异步安全：因为是单进程 asyncio，加锁主要是为了防止 reload 时的并发写入。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunk_ids: list[str] = []
        self._chunk_meta: dict[str, dict] = {}  # chunk_id → {content, heading, title_path}
        self._matrix: Optional[np.ndarray] = None  # shape (N, D), float32, 已归一化
        self._dim: int = 0

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    @property
    def dimension(self) -> int:
        return self._dim

    def load_from_records(
        self,
        records: list[tuple[str, bytes, str, str, list[str]]],
    ) -> None:
        """从数据库记录加载全部向量。

        Args:
            records: [(chunk_id, embedding_bytes, content, heading, title_path), ...]
        """

        with self._lock:
            if not records:
                self._chunk_ids = []
                self._chunk_meta = {}
                self._matrix = None
                self._dim = 0
                logger.info("VectorIndex 加载完成：0 个向量")
                return

            vectors = []
            chunk_ids = []
            meta = {}
            for chunk_id, emb_bytes, content, heading, title_path in records:
                if not emb_bytes:
                    continue
                try:
                    vec = np.frombuffer(emb_bytes, dtype=np.float32)
                except Exception as exc:
                    logger.warning(f"chunk {chunk_id} 向量反序列化失败: {exc}")
                    continue
                if vec.size == 0:
                    continue
                vectors.append(vec)
                chunk_ids.append(chunk_id)
                meta[chunk_id] = {
                    "content": content,
                    "heading": heading,
                    "title_path": title_path,
                }

            if not vectors:
                self._chunk_ids = []
                self._chunk_meta = {}
                self._matrix = None
                self._dim = 0
                return

            self._matrix = np.vstack(vectors).astype(np.float32)
            # 归一化（cosine similarity 需要）
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-8
            self._matrix = self._matrix / norms
            self._chunk_ids = chunk_ids
            self._chunk_meta = meta
            self._dim = self._matrix.shape[1]
            logger.info(
                f"VectorIndex 加载完成：{len(chunk_ids)} 个向量，维度 {self._dim}"
            )

    def add(self, chunk_id: str, vector: np.ndarray, content: str, heading: str = "", title_path: list[str] | None = None) -> None:
        """增量添加一个向量。"""

        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm

        with self._lock:
            if self._matrix is None:
                self._matrix = vec
                self._dim = vec.shape[1]
            else:
                if vec.shape[1] != self._dim:
                    raise ValueError(
                        f"向量维度不匹配：期望 {self._dim}，实际 {vec.shape[1]}"
                    )
                self._matrix = np.vstack([self._matrix, vec])
            self._chunk_ids.append(chunk_id)
            self._chunk_meta[chunk_id] = {
                "content": content,
                "heading": heading,
                "title_path": title_path or [],
            }

    def remove(self, chunk_id: str) -> bool:
        """删除一个向量。"""

        with self._lock:
            if chunk_id not in self._chunk_meta:
                return False
            idx = self._chunk_ids.index(chunk_id)
            self._chunk_ids.pop(idx)
            self._chunk_meta.pop(chunk_id)
            if self._matrix is not None:
                # numpy 删除行
                self._matrix = np.delete(self._matrix, idx, axis=0)
                if len(self._chunk_ids) == 0:
                    self._matrix = None
                    self._dim = 0
            return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """检索 top-K 最相似的 chunk。

        Args:
            query_vector: 查询向量（未归一化也行，内部会归一化）
            top_k: 返回前 K 个
            min_score: 最低相似度阈值

        Returns:
            List of (chunk_id, score)，按 score 降序。
        """

        if self._matrix is None or len(self._chunk_ids) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.size != self._dim:
            raise ValueError(
                f"查询向量维度不匹配：期望 {self._dim}，实际 {q.size}"
            )

        # 归一化查询向量
        q_norm = np.linalg.norm(q) + 1e-8
        q = q / q_norm

        # 读不加锁（避免检索慢）
        matrix = self._matrix
        chunk_ids = self._chunk_ids

        # cosine = matrix @ q
        scores = matrix @ q

        # 取 top-K
        if len(scores) > top_k:
            # 用 argpartition 加速
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            top_indices = np.argsort(-scores)

        result = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            result.append((chunk_ids[idx], score))
        return result

    def get_meta(self, chunk_id: str) -> Optional[dict]:
        return self._chunk_meta.get(chunk_id)

    def clear(self) -> None:
        with self._lock:
            self._chunk_ids = []
            self._chunk_meta = {}
            self._matrix = None
            self._dim = 0


__all__ = ["VectorIndex"]

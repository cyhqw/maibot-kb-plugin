"""kb.embedder

Embedding 抽象层 — 把"获取向量"这件事抽象成统一接口。

支持三种模式：
1. MaiBot 模式：通过 self.ctx.llm.embed 调用 host 的 embedding 服务（推荐，零配置）
2. OpenAI 兼容模式：直接调 OpenAI / DeepSeek / Moonshot / 智谱等兼容接口
3. 哑元模式：用随机向量占位（仅用于测试）

MaiBot 模式下，plugin.py 会把 plugin.ctx.llm.embed 注入到这里。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from typing import Any, Awaitable, Callable, Optional, Protocol

import numpy as np


logger = logging.getLogger("maikb.kb.embedder")


class EmbeddingError(Exception):
    """Embedding 失败。"""


class Embedder(Protocol):
    """Embedder 协议：把文本转成 float32 向量。"""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, text: str) -> np.ndarray: ...

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


# ----------------------------------------------------------------------
# MaiBot 模式：通过 plugin.ctx.llm.embed 调 host 的 embedding
# ----------------------------------------------------------------------

class MaiBotEmbedder:
    """通过 MaiBot 的 LLMCapability.embed 获取向量。

    用法：
        embedder = MaiBotEmbedder(embed_fn=plugin.ctx.llm.embed, model_name="text-embedding-3-small")
        vec = await embedder.embed("你好")
    """

    def __init__(
        self,
        embed_fn: Callable[..., Awaitable[Any]],
        *,
        model_name: str = "default",
        dimension: int = 1024,
        batch_size: int = 32,
    ) -> None:
        self._embed_fn = embed_fn
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        result = await self._embed_fn(text=text, model=self._model_name)
        vec = _extract_vector(result)
        if vec is None:
            raise EmbeddingError(f"MaiBot embed 返回空，原始结果: {result!r}")
        return vec.astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        # MaiBot 的 embed 当前是单条接口，做并发批处理
        sem = asyncio.Semaphore(self._batch_size)

        async def _one(t: str) -> np.ndarray:
            async with sem:
                try:
                    return await self.embed(t)
                except Exception as exc:
                    logger.warning(f"Embed 失败（用零向量代替）: {exc}")
                    return np.zeros(self._dimension, dtype=np.float32)

        return await asyncio.gather(*[_one(t) for t in texts])


def _extract_vector(result: Any) -> Optional[np.ndarray]:
    """从 MaiBot embed 返回值中提取向量（兼容多种返回格式）。"""

    if result is None:
        return None
    # 直接是 list / ndarray
    if isinstance(result, (list, tuple)):
        return np.asarray(result, dtype=np.float32)
    if isinstance(result, np.ndarray):
        return result
    # dict 形式：{"embedding": [...]} / {"vector": [...]} / {"data": [...]}
    if isinstance(result, dict):
        for key in ("embedding", "vector", "data", "embeddings", "result"):
            v = result.get(key)
            if v is not None:
                if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                    # data: [[...], [...]] 取第一条
                    return np.asarray(v[0], dtype=np.float32)
                return np.asarray(v, dtype=np.float32)
    return None


# ----------------------------------------------------------------------
# OpenAI 兼容模式
# ----------------------------------------------------------------------

class OpenAICompatibleEmbedder:
    """直接调 OpenAI / DeepSeek / Moonshot / 智谱等兼容接口。

    适合不依赖 MaiBot 的独立部署场景（如 CLI 导入脚本）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 32,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        vecs = await self.embed_batch([text])
        return vecs[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        # 懒加载 httpx
        try:
            import httpx
        except ImportError as exc:
            raise EmbeddingError("OpenAICompatibleEmbedder 需要 httpx，请 pip install httpx") from exc

        results: list[np.ndarray] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                try:
                    resp = await client.post(
                        f"{self._base_url}/embeddings",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "model": self._model_name,
                            "input": batch,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data["data"]:
                        results.append(np.asarray(item["embedding"], dtype=np.float32))
                except Exception as exc:
                    logger.warning(f"OpenAI embed batch 失败（用零向量代替）: {exc}")
                    results.extend(
                        np.zeros(self._dimension, dtype=np.float32) for _ in batch
                    )
        return results


# ----------------------------------------------------------------------
# 哑元模式：仅用于测试
# ----------------------------------------------------------------------

class DummyEmbedder:
    """哑元 embedder：用确定性哈希生成伪随机向量，仅用于测试。

    同一个文本永远生成同一个向量，保证测试可重现。
    """

    def __init__(self, dimension: int = 256, model_name: str = "dummy") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        # 用 SHA256 做种子，确定性生成
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        vec = np.array(
            [rng.gauss(0, 1) for _ in range(self._dimension)],
            dtype=np.float32,
        )
        # 归一化
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


__all__ = [
    "Embedder",
    "EmbeddingError",
    "MaiBotEmbedder",
    "OpenAICompatibleEmbedder",
    "DummyEmbedder",
]

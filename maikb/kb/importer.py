"""kb.importer

批量知识库导入器。

功能：
- 扫描目录下所有 .md / .txt 文件
- 增量更新：基于 file_hash 判断是否变更
- 切分（调用 chunker） + embedding（调 embedder） + 入库
- 启动时加载已有向量到内存索引

用法（在 plugin.py 中通过 API 调用）：

    from kb.importer import KnowledgeBaseImporter
    importer = KnowledgeBaseImporter(db, vector_index, embedder, knowledge_dir)
    result = await importer.ingest_directory()
    # → {"scanned": 55, "new": 5, "updated": 2, "unchanged": 48, "failed": 0, "chunks": 234}
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..database import MaiKBDatabase
from ..models import KnowledgeChunk, KnowledgeFile
from .chunker import Chunk, chunk_file
from .vector_store import VectorIndex


logger = logging.getLogger("maikb.kb.importer")


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


@dataclass
class IngestResult:
    """导入结果。"""

    scanned: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    chunks: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "new": self.new,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "failed": self.failed,
            "chunks": self.chunks,
            "failures": self.failures,
        }


class KnowledgeBaseImporter:
    """知识库导入器。"""

    def __init__(
        self,
        db: MaiKBDatabase,
        vector_index: VectorIndex,
        embedder,
        knowledge_dir: Path | str,
        *,
        target_chars: int = 500,
        max_chars: int = 1500,
        min_chars: int = 80,
        embed_batch_size: int = 16,
        default_category: Optional[str] = None,
    ) -> None:
        self._db = db
        self._index = vector_index
        self._embedder = embedder
        self._knowledge_dir = Path(knowledge_dir)
        self._target_chars = target_chars
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._embed_batch_size = embed_batch_size
        self._default_category = default_category

    async def load_existing_index(self) -> int:
        """从数据库加载已有向量到内存索引。返回加载的向量数。"""

        records = await self._db.get_all_chunk_embeddings()
        self._index.load_from_records(records)
        return self._index.size

    async def ingest_directory(self, *, force_rebuild: bool = False) -> IngestResult:
        """扫描目录并增量导入所有知识库文件。

        Args:
            force_rebuild: True 时忽略 file_hash，全部重新切分+嵌入
        """

        result = IngestResult()

        if not self._knowledge_dir.exists():
            logger.warning(f"知识库目录不存在: {self._knowledge_dir}")
            return result

        # 扫描所有支持的文件
        files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(self._knowledge_dir.rglob(f"*{ext}"))
        files.sort()

        result.scanned = len(files)
        logger.info(f"扫描到 {result.scanned} 个文件")

        for file_path in files:
            try:
                status = await self._ingest_one(file_path, force_rebuild=force_rebuild)
                if status == "new":
                    result.new += 1
                elif status == "updated":
                    result.updated += 1
                elif status == "unchanged":
                    result.unchanged += 1
            except Exception as exc:
                logger.error(f"导入失败 {file_path}: {exc}", exc_info=True)
                result.failed += 1
                result.failures.append((str(file_path), str(exc)))

        # 统计总 chunk 数
        all_files = await self._db.list_kb_files(status="ready")
        result.chunks = sum(f.chunk_count for f in all_files)

        logger.info(
            f"导入完成: scanned={result.scanned} new={result.new} "
            f"updated={result.updated} unchanged={result.unchanged} "
            f"failed={result.failed} chunks={result.chunks}"
        )
        return result

    async def ingest_file(self, file_path: Path | str, *, force_rebuild: bool = False) -> str:
        """导入单个文件。返回 new/updated/unchanged。"""

        return await self._ingest_one(Path(file_path), force_rebuild=force_rebuild)

    async def _ingest_one(self, file_path: Path, *, force_rebuild: bool) -> str:
        """导入单个文件。"""

        # 1. 读文件 + 算 hash
        try:
            raw_bytes = file_path.read_bytes()
        except Exception as exc:
            raise RuntimeError(f"读取失败: {exc}")

        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        # 相对路径
        try:
            rel_path = str(file_path.relative_to(self._knowledge_dir))
        except ValueError:
            rel_path = file_path.name

        # 2. 查是否已存在
        existing = await self._db.get_kb_file_by_path(rel_path)
        if existing and existing.file_hash == file_hash and not force_rebuild:
            logger.debug(f"跳过未变更文件: {rel_path}")
            return "unchanged"

        # 3. 解码文本
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("gbk", errors="replace")

        # 4. 切分
        chunks = chunk_file(
            file_path.name,
            text,
            target_chars=self._target_chars,
            max_chars=self._max_chars,
            min_chars=self._min_chars,
        )

        if not chunks:
            logger.warning(f"文件无内容，跳过: {rel_path}")
            return "unchanged"

        # 5. 提取文件标题（第一个 # 标题）
        title = self._extract_title(text) or file_path.stem

        # 6. 删旧 chunks（如果是更新）
        if existing:
            # 先从 DB 查出旧 chunk_id，再删 DB + 内存索引
            from sqlmodel import select as _sel
            from ..models import KnowledgeChunk as _KC
            async with self._db.get_db() as session:
                stmt = _sel(_KC.chunk_id).where(_KC.file_id == existing.file_id)
                rows = (await session.execute(stmt)).fetchall()
            old_chunk_ids = [r[0] for r in rows]
            await self._db.delete_chunks_by_file(existing.file_id)
            for cid in old_chunk_ids:
                self._index.remove(cid)

        # 7. upsert 文件元数据
        file_record = await self._db.upsert_kb_file(
            file_path=rel_path,
            file_name=file_path.name,
            file_hash=file_hash,
            file_size=len(raw_bytes),
            encoding="utf-8",
            title=title,
            category=self._default_category,
            tags=[],
            chunk_count=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
            last_ingested_at=datetime.now(timezone.utc),
            status="processing",
            error=None,
        )

        # 8. 入库 chunks
        chunk_models = [
            KnowledgeChunk(
                file_id=file_record.file_id,
                chunk_index=c.chunk_index,
                title_path=c.title_path,
                heading=c.heading,
                content=c.content,
                content_hash=c.content_hash,
                token_count=c.token_count,
                char_count=c.char_count,
            )
            for c in chunks
        ]
        await self._db.insert_kb_chunks(chunk_models)

        # 9. 批量 embedding
        texts = [c.content for c in chunks]
        try:
            vectors = await self._embedder.embed_batch(texts)
        except Exception as exc:
            # embedding 失败，把文件标记为 failed 但保留 chunks（可被 BM25 检索）
            await self._db.upsert_kb_file(
                file_path=rel_path,
                file_name=file_path.name,
                file_hash=file_hash,
                file_size=len(raw_bytes),
                title=title,
                category=self._default_category,
                chunk_count=len(chunks),
                status="failed",
                error=f"embedding failed: {exc}",
            )
            raise RuntimeError(f"embedding 失败: {exc}")

        # 10. 写回向量
        for chunk_model, vec in zip(chunk_models, vectors):
            vec_f32 = np.asarray(vec, dtype=np.float32)
            await self._db.update_chunk_embedding(
                chunk_model.chunk_id,
                vec_f32.tobytes(),
                self._embedder.model_name,
            )
            # 同步加到内存索引
            self._index.add(
                chunk_model.chunk_id,
                vec_f32,
                content=chunk_model.content,
                heading=chunk_model.heading or "",
                title_path=chunk_model.title_path or [],
            )

        # 11. 标记为 ready
        await self._db.upsert_kb_file(
            file_path=rel_path,
            file_name=file_path.name,
            file_hash=file_hash,
            file_size=len(raw_bytes),
            title=title,
            category=self._default_category,
            chunk_count=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
            last_ingested_at=datetime.now(timezone.utc),
            status="ready",
            error=None,
        )

        return "updated" if existing else "new"

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        """从 markdown 提取第一个 # 标题。"""

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                return stripped[2:].strip()
        return None


def _lazy_np():
    """延迟导入 numpy（保留为兼容旧调用）。"""

    import numpy
    return numpy


__all__ = ["KnowledgeBaseImporter", "IngestResult", "SUPPORTED_EXTENSIONS"]

"""maikb.database

异步 DAO 类 — 移植自 AstrBot `astrbot/core/db/sqlite.py` 的 SQLiteDatabase。

设计要点：
- SQLModel + SQLAlchemy[asyncio] + aiosqlite
- async_sessionmaker + expire_on_commit=False
- PRAGMA 调优套餐（WAL + busy_timeout + mmap 等）
- 每次启动都跑幂等的 _ensure_xxx_column
- 提供 ~30 个常用 DAO 方法（AstrBot 原版有 80+，这里取核心子集）
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import select

from .models import (
    Attachment,
    CommandConfig,
    ConversationV2,
    CronJob,
    KnowledgeChunk,
    KnowledgeFile,
    Preference,
    Persona,
    PersonaFolder,
    PlatformMessageHistory,
    PlatformSession,
    PlatformStat,
    ProviderStat,
    UmoAlias,
    build_umo,
)


logger = logging.getLogger("maikb.database")


class MaiKBDatabase:
    """异步 SQLite DAO（移植自 AstrBot SQLiteDatabase）。

    用法：

        db = MaiKBDatabase("/path/to/maikb.db")
        await db.initialize()
        async with db.get_db() as session:
            ...

    也可以通过 plugin.py 中的全局单例访问：

        from maikb import get_db
        db = await get_db()
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path: Path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 复刻 AstrBot 的 PRAGMA 调优套餐
        self.database_url = f"sqlite+aiosqlite:///{self.db_path}"
        self.engine: AsyncEngine = create_async_engine(
            self.database_url,
            echo=False,
            future=True,
            connect_args={"timeout": 30},
        )
        self.AsyncSessionLocal = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self.inited: bool = False

    # ------------------------------------------------------------------
    # 启动 / 关闭
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """启动初始化：建表 + PRAGMA + 幂等列补齐。"""

        if self.inited:
            return

        async with self.engine.begin() as conn:
            # 1) 建表（已存在的表会被跳过）
            from sqlmodel import SQLModel
            await conn.run_sync(SQLModel.metadata.create_all)

            # 2) SQLite PRAGMA 调优套餐（移植自 AstrBot）
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.execute(text("PRAGMA cache_size=20000"))
            await conn.execute(text("PRAGMA temp_store=MEMORY"))
            await conn.execute(text("PRAGMA mmap_size=134217728"))
            await conn.execute(text("PRAGMA optimize"))

            # 3) 幂等列补齐（双保险）
            await self._ensure_persona_skills_column(conn)
            await self._ensure_persona_custom_error_message_column(conn)
            await self._ensure_platform_message_history_checkpoint_column(conn)

            # 4) FTS5 虚拟表（知识库全文检索）
            await self._ensure_kb_fts_table(conn)

        self.inited = True
        logger.info(f"MaiKBDatabase 初始化完成: {self.db_path}")

    async def close(self) -> None:
        """关闭引擎，释放连接池。"""

        await self.engine.dispose()
        self.inited = False
        logger.info("MaiKBDatabase 已关闭")

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def get_db(self) -> AsyncIterator[AsyncSession]:
        """获取一个异步 Session（推荐用法）。

        用法：

            async with db.get_db() as session:
                async with session.begin():
                    session.add(some_model)
        """

        async with self.AsyncSessionLocal() as session:
            yield session

    async def _run_in_tx(self, fn):
        """统一的事务包装：自动 begin / commit / rollback。"""

        async with self.get_db() as session:
            async with session.begin():
                return await fn(session)

    # ------------------------------------------------------------------
    # 幂等列补齐（移植自 AstrBot _ensure_xxx_column 系列）
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_columns(conn, table_name: str) -> set[str]:
        result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        return {row[1] for row in result.fetchall()}

    async def _ensure_persona_skills_column(self, conn) -> None:
        """补齐 personas.skills 列。"""

        cols = await self._get_columns(conn, "personas")
        if "skills" not in cols:
            await conn.execute(text("ALTER TABLE personas ADD COLUMN skills JSON DEFAULT '[]'"))
            logger.info("已补齐 personas.skills 列")

    async def _ensure_persona_custom_error_message_column(self, conn) -> None:
        """补齐 personas.custom_error_message 列。"""

        cols = await self._get_columns(conn, "personas")
        if "custom_error_message" not in cols:
            await conn.execute(text("ALTER TABLE personas ADD COLUMN custom_error_message TEXT"))
            logger.info("已补齐 personas.custom_error_message 列")

    async def _ensure_platform_message_history_checkpoint_column(self, conn) -> None:
        """补齐 platform_message_history.llm_checkpoint_id 列。"""

        cols = await self._get_columns(conn, "platform_message_history")
        if "llm_checkpoint_id" not in cols:
            await conn.execute(text("ALTER TABLE platform_message_history ADD COLUMN llm_checkpoint_id TEXT"))
            logger.info("已补齐 platform_message_history.llm_checkpoint_id 列")

    @staticmethod
    async def _ensure_kb_fts_table(conn) -> None:
        """创建 FTS5 虚拟表（如不存在）。

        使用 trigram 分词器（SQLite 3.34+ 自带），对中文友好：
        - 把文本按 3-gram 切分，可以匹配任意 ≥3 字符的子串
        - 短查询（< 3 字符）会走 LIKE 兜底

        替代方案 unicode61 对中文按字切分时，会把整句当单 token，无法匹配短语；
        trigram 牺牲一些存储空间换可用性，对 RAG 场景是正确取舍。
        """

        await conn.execute(text(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                tokenize = 'trigram'
            )
            """
        ))

    # ==================================================================
    # Conversation 相关 DAO
    # ==================================================================

    async def get_conversation_by_id(self, cid: str) -> Optional[ConversationV2]:
        async with self.get_db() as session:
            stmt = select(ConversationV2).where(ConversationV2.conversation_id == cid)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_conversations_by_user(self, umo: str, limit: int = 50) -> list[ConversationV2]:
        async with self.get_db() as session:
            stmt = (
                select(ConversationV2)
                .where(ConversationV2.user_id == umo)
                .order_by(ConversationV2.updated_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def create_conversation(
        self,
        *,
        user_id: str,
        platform_id: str,
        content: Optional[list[Any]] = None,
        title: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> ConversationV2:
        async def _op(session: AsyncSession) -> ConversationV2:
            new_conv = ConversationV2(
                user_id=user_id,
                platform_id=platform_id,
                content=content or [],
                title=title,
                persona_id=persona_id,
            )
            session.add(new_conv)
            await session.flush()
            await session.refresh(new_conv)
            return new_conv

        return await self._run_in_tx(_op)

    async def update_conversation_content(
        self, cid: str, content: list[Any], token_usage_delta: int = 0
    ) -> bool:
        async def _op(session: AsyncSession) -> bool:
            stmt = select(ConversationV2).where(ConversationV2.conversation_id == cid)
            result = await session.execute(stmt)
            conv = result.scalar_one_or_none()
            if conv is None:
                return False
            conv.content = content
            conv.token_usage = (conv.token_usage or 0) + token_usage_delta
            return True

        return await self._run_in_tx(_op)

    async def delete_conversation(self, cid: str) -> bool:
        async def _op(session: AsyncSession) -> bool:
            stmt = select(ConversationV2).where(ConversationV2.conversation_id == cid)
            result = await session.execute(stmt)
            conv = result.scalar_one_or_none()
            if conv is None:
                return False
            await session.delete(conv)
            return True

        return await self._run_in_tx(_op)

    # ==================================================================
    # Preference（万能 KV 表）相关 DAO
    # ==================================================================

    async def get_preference(self, scope: str, scope_id: str, key: str) -> Optional[Any]:
        """读取 KV 值（自动解包 value["val"]）。"""

        async with self.get_db() as session:
            stmt = select(Preference).where(
                Preference.scope == scope,
                Preference.scope_id == scope_id,
                Preference.key == key,
            )
            result = await session.execute(stmt)
            pref = result.scalar_one_or_none()
            if pref is None:
                return None
            return pref.value.get("val")

    async def upsert_preference(
        self, scope: str, scope_id: str, key: str, value: Any
    ) -> None:
        """Upsert KV（先查后插/更新）。"""

        async def _op(session: AsyncSession) -> None:
            stmt = select(Preference).where(
                Preference.scope == scope,
                Preference.scope_id == scope_id,
                Preference.key == key,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            payload = {"val": value}
            if existing is None:
                session.add(
                    Preference(
                        scope=scope, scope_id=scope_id, key=key, value=payload
                    )
                )
            else:
                existing.value = payload

        await self._run_in_tx(_op)

    async def delete_preference(self, scope: str, scope_id: str, key: str) -> bool:
        async def _op(session: AsyncSession) -> bool:
            stmt = select(Preference).where(
                Preference.scope == scope,
                Preference.scope_id == scope_id,
                Preference.key == key,
            )
            result = await session.execute(stmt)
            pref = result.scalar_one_or_none()
            if pref is None:
                return False
            await session.delete(pref)
            return True

        return await self._run_in_tx(_op)

    async def list_preferences(
        self, scope: str, scope_id: str, key_prefix: str = ""
    ) -> dict[str, Any]:
        """列出某 scope + scope_id 下所有 KV（支持前缀过滤）。"""

        async with self.get_db() as session:
            stmt = select(Preference).where(
                Preference.scope == scope,
                Preference.scope_id == scope_id,
            )
            if key_prefix:
                stmt = stmt.where(Preference.key.like(f"{key_prefix}%"))
            result = await session.execute(stmt)
            return {
                pref.key: pref.value.get("val")
                for pref in result.scalars().all()
            }

    # ==================================================================
    # Persona 相关 DAO
    # ==================================================================

    async def get_persona(self, persona_id: str) -> Optional[Persona]:
        async with self.get_db() as session:
            stmt = select(Persona).where(Persona.persona_id == persona_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_personas(self, folder_id: Optional[str] = None) -> list[Persona]:
        async with self.get_db() as session:
            stmt = select(Persona).order_by(Persona.sort_order.asc())
            if folder_id is not None:
                stmt = stmt.where(Persona.folder_id == folder_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def create_persona(self, **kwargs: Any) -> Persona:
        async def _op(session: AsyncSession) -> Persona:
            persona = Persona(**kwargs)
            session.add(persona)
            await session.flush()
            await session.refresh(persona)
            return persona

        return await self._run_in_tx(_op)

    async def delete_persona(self, persona_id: str) -> bool:
        async def _op(session: AsyncSession) -> bool:
            stmt = select(Persona).where(Persona.persona_id == persona_id)
            result = await session.execute(stmt)
            persona = result.scalar_one_or_none()
            if persona is None:
                return False
            await session.delete(persona)
            return True

        return await self._run_in_tx(_op)

    # ==================================================================
    # PlatformMessageHistory 相关 DAO
    # ==================================================================

    async def add_message_history(
        self,
        *,
        platform_id: str,
        user_id: str,
        content: dict[str, Any],
        sender_id: Optional[str] = None,
        sender_name: Optional[str] = None,
        llm_checkpoint_id: Optional[str] = None,
    ) -> PlatformMessageHistory:
        async def _op(session: AsyncSession) -> PlatformMessageHistory:
            record = PlatformMessageHistory(
                platform_id=platform_id,
                user_id=user_id,
                sender_id=sender_id,
                sender_name=sender_name,
                content=content,
                llm_checkpoint_id=llm_checkpoint_id,
            )
            session.add(record)
            await session.flush()
            await session.refresh(record)
            return record

        return await self._run_in_tx(_op)

    async def get_message_history(
        self, umo: str, limit: int = 50, before_id: Optional[int] = None
    ) -> list[PlatformMessageHistory]:
        async with self.get_db() as session:
            stmt = (
                select(PlatformMessageHistory)
                .where(PlatformMessageHistory.user_id == umo)
                .order_by(PlatformMessageHistory.id.desc())
                .limit(limit)
            )
            if before_id is not None:
                stmt = stmt.where(PlatformMessageHistory.id < before_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ==================================================================
    # CronJob 相关 DAO
    # ==================================================================

    async def create_cron_job(self, **kwargs: Any) -> CronJob:
        async def _op(session: AsyncSession) -> CronJob:
            job = CronJob(**kwargs)
            session.add(job)
            await session.flush()
            await session.refresh(job)
            return job

        return await self._run_in_tx(_op)

    async def get_pending_cron_jobs(self, before_ts: int) -> list[CronJob]:
        async with self.get_db() as session:
            stmt = (
                select(CronJob)
                .where(CronJob.enabled == True)  # noqa: E712
                .where(CronJob.status == "pending")
                .where(CronJob.next_run_time <= before_ts)
                .order_by(CronJob.next_run_time.asc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def mark_cron_done(self, job_id: str, *, error: Optional[str] = None) -> bool:
        async def _op(session: AsyncSession) -> bool:
            stmt = select(CronJob).where(CronJob.job_id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None:
                return False
            job.last_run_at = int(datetime.now(timezone.utc).timestamp())
            if error:
                job.status = "failed"
                job.last_error = error
            else:
                job.status = "done" if job.run_once else "pending"
                job.last_error = None
            return True

        return await self._run_in_tx(_op)

    # ==================================================================
    # PlatformStat / ProviderStat 相关 DAO（统计聚合）
    # ==================================================================

    async def incr_platform_stat(
        self, timestamp: int, platform_id: str, platform_type: str, count: int = 1
    ) -> None:
        """原子自增平台统计（用 SQLite UPSERT 语法）。"""

        async with self.get_db() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        INSERT INTO platform_stats (timestamp, platform_id, platform_type, count, created_at, updated_at)
                        VALUES (:timestamp, :platform_id, :platform_type, :count, :now, :now)
                        ON CONFLICT(timestamp, platform_id, platform_type) DO UPDATE SET
                            count = platform_stats.count + EXCLUDED.count,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {
                        "timestamp": timestamp,
                        "platform_id": platform_id,
                        "platform_type": platform_type,
                        "count": count,
                        "now": datetime.now(timezone.utc),
                    },
                )

    async def record_provider_stat(self, **kwargs: Any) -> ProviderStat:
        async def _op(session: AsyncSession) -> ProviderStat:
            stat = ProviderStat(**kwargs)
            session.add(stat)
            await session.flush()
            await session.refresh(stat)
            return stat

        return await self._run_in_tx(_op)

    # ==================================================================
    # CommandConfig 相关 DAO
    # ==================================================================

    async def upsert_command_config(self, **kwargs: Any) -> CommandConfig:
        handler_full_name = kwargs.get("handler_full_name")
        if not handler_full_name:
            raise ValueError("handler_full_name is required")

        async def _op(session: AsyncSession) -> CommandConfig:
            stmt = select(CommandConfig).where(
                CommandConfig.handler_full_name == handler_full_name
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is None:
                cfg = CommandConfig(**kwargs)
                session.add(cfg)
                await session.flush()
                await session.refresh(cfg)
                return cfg
            for k, v in kwargs.items():
                setattr(existing, k, v)
            await session.flush()
            await session.refresh(existing)
            return existing

        return await self._run_in_tx(_op)

    # ==================================================================
    # UmoAlias 相关 DAO
    # ==================================================================

    async def set_umo_alias(self, umo: str, alias: str) -> UmoAlias:
        async def _op(session: AsyncSession) -> UmoAlias:
            stmt = select(UmoAlias).where(UmoAlias.umo == umo)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is None:
                record = UmoAlias(umo=umo, user_alias=alias)
                session.add(record)
                await session.flush()
                await session.refresh(record)
                return record
            existing.user_alias = alias
            await session.flush()
            await session.refresh(existing)
            return existing

        return await self._run_in_tx(_op)

    # ==================================================================
    # 诊断 / 维护
    # ==================================================================

    async def count_rows(self, table_name: str) -> int:
        async with self.get_db() as session:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return int(result.scalar() or 0)

    async def list_tables(self) -> list[str]:
        async with self.get_db() as session:
            result = await session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
            return [row[0] for row in result.fetchall()]

    # ==================================================================
    # KnowledgeFile 相关 DAO
    # ==================================================================

    async def upsert_kb_file(self, **kwargs: Any) -> KnowledgeFile:
        """Upsert 知识库文件元数据（按 file_path 唯一）。"""

        file_path = kwargs.get("file_path")
        if not file_path:
            raise ValueError("file_path is required")

        async def _op(session: AsyncSession) -> KnowledgeFile:
            stmt = select(KnowledgeFile).where(KnowledgeFile.file_path == file_path)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is None:
                f = KnowledgeFile(**kwargs)
                session.add(f)
                await session.flush()
                await session.refresh(f)
                return f
            for k, v in kwargs.items():
                setattr(existing, k, v)
            await session.flush()
            await session.refresh(existing)
            return existing

        return await self._run_in_tx(_op)

    async def get_kb_file_by_path(self, file_path: str) -> Optional[KnowledgeFile]:
        async with self.get_db() as session:
            stmt = select(KnowledgeFile).where(KnowledgeFile.file_path == file_path)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_kb_file_by_id(self, file_id: str) -> Optional[KnowledgeFile]:
        async with self.get_db() as session:
            stmt = select(KnowledgeFile).where(KnowledgeFile.file_id == file_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_kb_files(
        self, status: Optional[str] = None, category: Optional[str] = None
    ) -> list[KnowledgeFile]:
        async with self.get_db() as session:
            stmt = select(KnowledgeFile).order_by(KnowledgeFile.file_path.asc())
            if status:
                stmt = stmt.where(KnowledgeFile.status == status)
            if category:
                stmt = stmt.where(KnowledgeFile.category == category)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def delete_kb_file(self, file_id: str) -> int:
        """删除文件元数据 + 对应 chunks + FTS 索引。返回删除的 chunk 数。"""

        async def _op(session: AsyncSession) -> int:
            # 先查 chunk 数
            stmt_chunks = select(KnowledgeChunk).where(KnowledgeChunk.file_id == file_id)
            chunks = (await session.execute(stmt_chunks)).scalars().all()
            chunk_ids = [c.chunk_id for c in chunks]
            chunk_count = len(chunks)

            # 删 chunks
            for c in chunks:
                await session.delete(c)

            # 删 FTS 索引
            if chunk_ids:
                placeholders = ",".join(f":id{i}" for i in range(len(chunk_ids)))
                params = {f"id{i}": cid for i, cid in enumerate(chunk_ids)}
                await session.execute(
                    text(f"DELETE FROM kb_chunks_fts WHERE chunk_id IN ({placeholders})"),
                    params,
                )

            # 删 file
            stmt_file = select(KnowledgeFile).where(KnowledgeFile.file_id == file_id)
            f = (await session.execute(stmt_file)).scalar_one_or_none()
            if f:
                await session.delete(f)

            return chunk_count

        return await self._run_in_tx(_op)

    # ==================================================================
    # KnowledgeChunk 相关 DAO
    # ==================================================================

    async def insert_kb_chunks(self, chunks: list[KnowledgeChunk]) -> int:
        """批量插入 chunks（同时同步到 FTS 索引）。"""

        if not chunks:
            return 0

        async def _op(session: AsyncSession) -> int:
            for c in chunks:
                session.add(c)
            await session.flush()
            # 同步 FTS
            for c in chunks:
                await session.execute(
                    text("INSERT INTO kb_chunks_fts(chunk_id, content) VALUES (:cid, :content)"),
                    {"cid": c.chunk_id, "content": c.content},
                )
            return len(chunks)

        return await self._run_in_tx(_op)

    async def delete_chunks_by_file(self, file_id: str) -> int:
        """删除某文件的所有 chunks + FTS 索引。"""

        async def _op(session: AsyncSession) -> int:
            stmt = select(KnowledgeChunk).where(KnowledgeChunk.file_id == file_id)
            chunks = (await session.execute(stmt)).scalars().all()
            for c in chunks:
                await session.delete(c)
                await session.execute(
                    text("DELETE FROM kb_chunks_fts WHERE chunk_id = :cid"),
                    {"cid": c.chunk_id},
                )
            return len(chunks)

        return await self._run_in_tx(_op)

    async def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding: bytes,
        embedding_model: str,
    ) -> bool:
        """更新某 chunk 的向量。"""

        async def _op(session: AsyncSession) -> bool:
            stmt = select(KnowledgeChunk).where(KnowledgeChunk.chunk_id == chunk_id)
            result = await session.execute(stmt)
            c = result.scalar_one_or_none()
            if c is None:
                return False
            c.embedding = embedding
            c.embedding_model = embedding_model
            c.embedded_at = datetime.now(timezone.utc)
            return True

        return await self._run_in_tx(_op)

    async def get_all_chunk_embeddings(self) -> list[tuple[str, bytes, str, str, list[str]]]:
        """加载所有 chunk 的向量到内存（启动时用）。

        返回 [(chunk_id, embedding_bytes, content, heading, title_path), ...]
        """

        async with self.get_db() as session:
            stmt = select(
                KnowledgeChunk.chunk_id,
                KnowledgeChunk.embedding,
                KnowledgeChunk.content,
                KnowledgeChunk.heading,
                KnowledgeChunk.title_path,
            ).where(KnowledgeChunk.embedding.is_not(None))
            result = await session.execute(stmt)
            return [
                (row[0], row[1], row[2], row[3] or "", row[4] or [])
                for row in result.fetchall()
            ]

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """按 chunk_id 列表批量查询完整 chunk 信息。"""

        if not chunk_ids:
            return []
        async with self.get_db() as session:
            stmt = select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(chunk_ids))
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def fts_search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """BM25 全文检索。

        使用 trigram 分词器，对中文友好。
        - 查询 ≥ 3 字符：走 FTS5 MATCH，返回 BM25 分数
        - 查询 < 3 字符：回退 LIKE 兜底（短查询 trigram 无法匹配）

        Returns:
            List of (chunk_id, score). 分数越高越相关。
        """

        if not query or not query.strip():
            return []

        q = query.strip()

        # 短查询回退 LIKE。
        # 注意：必须查真实表 kb_chunks，不能查 FTS5 虚拟表 kb_chunks_fts——
        # FTS5 虚拟表的列不支持 LIKE 子串过滤（会静默返回空），这是历史 bug。
        if len(q) < 3:
            async with self.get_db() as session:
                stmt = text(
                    """
                    SELECT chunk_id, 1.0 AS score
                    FROM kb_chunks
                    WHERE content LIKE :pattern
                    LIMIT :limit
                    """
                )
                result = await session.execute(
                    stmt, {"pattern": f"%{q}%", "limit": limit}
                )
                return [(row[0], float(row[1])) for row in result.fetchall()]

        # 正常 FTS5 MATCH（trigram 支持 ≥3 字符的子串匹配）
        safe_query = q.replace('"', '""')
        async with self.get_db() as session:
            try:
                stmt = text(
                    """
                    SELECT chunk_id, bm25(kb_chunks_fts) AS score
                    FROM kb_chunks_fts
                    WHERE kb_chunks_fts MATCH :q
                    ORDER BY score
                    LIMIT :limit
                    """
                )
                result = await session.execute(stmt, {"q": safe_query, "limit": limit})
                # bm25 分数越小越好（负数），统一转成越大越好
                return [(row[0], -float(row[1])) for row in result.fetchall()]
            except Exception as exc:
                logger.warning(f"FTS 查询失败: {exc}, 回退 LIKE")
                # FTS 失败时兜底 LIKE（同样查真实表 kb_chunks）
                async with self.get_db() as session2:
                    stmt = text(
                        """
                        SELECT chunk_id, 1.0 AS score
                        FROM kb_chunks
                        WHERE content LIKE :pattern
                        LIMIT :limit
                        """
                    )
                    result = await session2.execute(
                        stmt, {"pattern": f"%{q}%", "limit": limit}
                    )
                    return [(row[0], float(row[1])) for row in result.fetchall()]


__all__ = ["MaiKBDatabase"]

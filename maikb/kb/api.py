"""kb.api

KB 模块对外 API 的实现（Mixin）。

设计：把 KB 相关的 @API 方法抽到独立 Mixin，避免 plugin.py 过长。
plugin.py 中的 MaiKBPlugin 通过多重继承把这个 Mixin 拉进来。

API 列表：
- maikb.kb.ingest_directory  批量扫描目录导入
- maikb.kb.ingest_file       单文件导入
- maikb.kb.list_files        列出已入库文件
- maikb.kb.search            混合检索
- maikb.kb.search_vector     仅向量检索
- maikb.kb.search_bm25       仅 BM25 检索
- maikb.kb.delete_file       删除某文件及其 chunks
- maikb.kb.stats             知识库统计
- maikb.kb.reload_index      重载内存索引
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from maibot_sdk import API, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType

import maikb
from maikb import get_db
from maikb.kb import (
    DummyEmbedder,
    HybridSearcher,
    KnowledgeBaseImporter,
    MaiBotEmbedder,
    OpenAICompatibleEmbedder,
    SearchQuery,
    VectorIndex,
)


logger = logging.getLogger("maikb.kb.api")


# 全局单例（plugin 在 on_load 中初始化）
_kb_importer: Optional[KnowledgeBaseImporter] = None
_kb_searcher: Optional[HybridSearcher] = None
_kb_vector_index: Optional[VectorIndex] = None
_kb_embedder: Any = None  # Embedder protocol


def init_kb(
    knowledge_dir: Path | str,
    embedding_config: dict,
    maibot_embed_fn=None,
) -> None:
    """初始化 KB 模块（plugin.on_load 中调用）。

    Args:
        knowledge_dir: 知识库源文件目录
        embedding_config: embedding 配置 dict
            {
                "provider": "maibot" | "openai" | "dummy",
                "model": "text-embedding-3-small",
                "dimension": 1536,
                "api_key": "...",       # openai 模式必填
                "base_url": "...",      # openai 模式可选
                "batch_size": 16,
            }
        maibot_embed_fn: MaiBot 模式下传入 plugin.ctx.llm.embed
    """

    global _kb_importer, _kb_searcher, _kb_vector_index, _kb_embedder

    knowledge_dir = Path(knowledge_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    provider = str(embedding_config.get("provider", "dummy")).lower()
    model_name = str(embedding_config.get("model", "default"))
    dimension = int(embedding_config.get("dimension", 1536))
    batch_size = int(embedding_config.get("batch_size", 16))

    if provider == "maibot":
        if maibot_embed_fn is None:
            logger.warning("MaiBot embedder 模式但未提供 embed_fn，降级到 dummy")
            _kb_embedder = DummyEmbedder(dimension=dimension, model_name=model_name)
        else:
            _kb_embedder = MaiBotEmbedder(
                embed_fn=maibot_embed_fn,
                model_name=model_name,
                dimension=dimension,
                batch_size=batch_size,
            )
    elif provider == "openai":
        api_key = embedding_config.get("api_key", "")
        base_url = embedding_config.get("base_url", "https://api.openai.com/v1")
        if not api_key:
            logger.warning("OpenAI embedder 模式但未配置 api_key，降级到 dummy")
            _kb_embedder = DummyEmbedder(dimension=dimension, model_name=model_name)
        else:
            _kb_embedder = OpenAICompatibleEmbedder(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                dimension=dimension,
                batch_size=batch_size,
            )
    else:
        logger.info("使用 DummyEmbedder（仅用于测试，检索质量很差）")
        _kb_embedder = DummyEmbedder(dimension=dimension, model_name=model_name)

    _kb_vector_index = VectorIndex()
    _kb_importer = KnowledgeBaseImporter(
        db=get_db(),
        vector_index=_kb_vector_index,
        embedder=_kb_embedder,
        knowledge_dir=knowledge_dir,
        default_category=embedding_config.get("default_category"),
    )
    _kb_searcher = HybridSearcher(get_db(), _kb_vector_index, _kb_embedder)

    logger.info(
        f"KB 模块初始化完成: dir={knowledge_dir} provider={provider} "
        f"model={model_name} dim={dimension}"
    )


async def load_kb_index() -> int:
    """从数据库加载已有向量到内存索引（on_load 中调用）。"""

    if _kb_importer is None:
        raise RuntimeError("KB 未初始化，请先调用 init_kb()")
    return await _kb_importer.load_existing_index()


def close_kb() -> None:
    """清理 KB 模块状态（on_unload 中调用）。"""

    global _kb_importer, _kb_searcher, _kb_vector_index, _kb_embedder
    if _kb_vector_index is not None:
        _kb_vector_index.clear()
    _kb_importer = None
    _kb_searcher = None
    _kb_vector_index = None
    _kb_embedder = None


class KbApiMixin:
    """KB API 方法集合，由 MaiKBPlugin 多重继承。"""

    # ==================================================================
    # 导入 API
    # ==================================================================

    @API(
        "maikb.kb.ingest_directory",
        description="扫描知识库目录并增量导入所有 markdown/txt 文件",
        version="1",
        public=True,
    )
    async def api_kb_ingest_directory(
        self,
        force_rebuild: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """批量扫描目录导入。

        Args:
            force_rebuild: True 时忽略 file_hash 全部重新切分+嵌入
        """

        if _kb_importer is None:
            return {"success": False, "error": "KB 模块未初始化"}

        result = await _kb_importer.ingest_directory(force_rebuild=force_rebuild)
        return {"success": True, **result.to_dict()}

    @API(
        "maikb.kb.ingest_file",
        description="导入单个知识库文件",
        version="1",
        public=True,
    )
    async def api_kb_ingest_file(
        self,
        file_path: str,
        force_rebuild: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """导入单文件。file_path 可以是绝对路径或相对于 knowledge_dir 的相对路径。"""

        if _kb_importer is None:
            return {"success": False, "error": "KB 模块未初始化"}

        try:
            status = await _kb_importer.ingest_file(file_path, force_rebuild=force_rebuild)
            return {"success": True, "status": status}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ==================================================================
    # 检索 API
    # ==================================================================

    @API(
        "maikb.kb.search",
        description="混合检索知识库（向量 + BM25 + RRF 融合）",
        version="1",
        public=True,
    )
    async def api_kb_search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        file_ids: Optional[list[str]] = None,
        use_vector: bool = True,
        use_bm25: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """混合检索。

        Args:
            query: 查询文本
            top_k: 返回结果数
            category: 按文件 category 过滤
            file_ids: 限定在某几个文件内检索
            use_vector: 是否走向量
            use_bm25: 是否走 BM25
        """

        if _kb_searcher is None:
            return {"success": False, "error": "KB 模块未初始化"}

        q = SearchQuery(
            query=query,
            top_k=top_k,
            category=category,
            file_ids=file_ids,
            use_vector=use_vector,
            use_bm25=use_bm25,
        )
        hits = await _kb_searcher.search(q)
        return {
            "success": True,
            "query": query,
            "count": len(hits),
            "items": [h.to_dict() for h in hits],
        }

    @API(
        "maikb.kb.search_vector",
        description="仅向量检索",
        version="1",
        public=True,
    )
    async def api_kb_search_vector(
        self,
        query: str,
        top_k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        if _kb_searcher is None:
            return {"success": False, "error": "KB 模块未初始化"}

        q = SearchQuery(
            query=query, top_k=top_k, use_vector=True, use_bm25=False
        )
        hits = await _kb_searcher.search(q)
        return {
            "success": True,
            "query": query,
            "count": len(hits),
            "items": [h.to_dict() for h in hits],
        }

    @API(
        "maikb.kb.search_bm25",
        description="仅 BM25 全文检索",
        version="1",
        public=True,
    )
    async def api_kb_search_bm25(
        self,
        query: str,
        top_k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        if _kb_searcher is None:
            return {"success": False, "error": "KB 模块未初始化"}

        q = SearchQuery(
            query=query, top_k=top_k, use_vector=False, use_bm25=True
        )
        hits = await _kb_searcher.search(q)
        return {
            "success": True,
            "query": query,
            "count": len(hits),
            "items": [h.to_dict() for h in hits],
        }

    # ==================================================================
    # 管理 API
    # ==================================================================

    @API(
        "maikb.kb.list_files",
        description="列出知识库中所有文件",
        version="1",
        public=True,
    )
    async def api_kb_list_files(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        db = get_db()
        files = await db.list_kb_files(status=status, category=category)
        return {
            "success": True,
            "count": len(files),
            "items": [
                {
                    "file_id": f.file_id,
                    "file_path": f.file_path,
                    "file_name": f.file_name,
                    "title": f.title,
                    "category": f.category,
                    "status": f.status,
                    "chunk_count": f.chunk_count,
                    "total_tokens": f.total_tokens,
                    "file_size": f.file_size,
                    "last_ingested_at": f.last_ingested_at.isoformat() if f.last_ingested_at else None,
                    "error": f.error,
                }
                for f in files
            ],
        }

    @API(
        "maikb.kb.delete_file",
        description="删除某个知识库文件及其所有 chunks",
        version="1",
        public=True,
    )
    async def api_kb_delete_file(self, file_id: str, **_: Any) -> dict[str, Any]:
        if _kb_vector_index is None:
            return {"success": False, "error": "KB 模块未初始化"}

        db = get_db()
        # 先查 chunk_ids 用于从内存索引删除
        from sqlmodel import select
        from maikb.models import KnowledgeChunk
        async with db.get_db() as session:
            stmt = select(KnowledgeChunk.chunk_id).where(KnowledgeChunk.file_id == file_id)
            rows = (await session.execute(stmt)).fetchall()
        chunk_ids = [r[0] for r in rows]

        deleted = await db.delete_kb_file(file_id)
        for cid in chunk_ids:
            _kb_vector_index.remove(cid)

        return {"success": True, "deleted_chunks": deleted}

    @API(
        "maikb.kb.stats",
        description="知识库统计信息",
        version="1",
        public=True,
    )
    async def api_kb_stats(self, **_: Any) -> dict[str, Any]:
        db = get_db()
        all_files = await db.list_kb_files()
        ready_files = [f for f in all_files if f.status == "ready"]
        total_chunks = sum(f.chunk_count for f in ready_files)
        total_tokens = sum(f.total_tokens for f in ready_files)
        total_size = sum(f.file_size for f in ready_files)
        return {
            "success": True,
            "files_total": len(all_files),
            "files_ready": len(ready_files),
            "files_failed": sum(1 for f in all_files if f.status == "failed"),
            "chunks_total": total_chunks,
            "tokens_total": total_tokens,
            "size_bytes": total_size,
            "vector_index_size": _kb_vector_index.size if _kb_vector_index else 0,
            "embedding_model": _kb_embedder.model_name if _kb_embedder else None,
            "embedding_dimension": _kb_embedder.dimension if _kb_embedder else 0,
        }

    @API(
        "maikb.kb.reload_index",
        description="从数据库重新加载向量索引到内存",
        version="1",
        public=True,
    )
    async def api_kb_reload_index(self, **_: Any) -> dict[str, Any]:
        if _kb_importer is None:
            return {"success": False, "error": "KB 模块未初始化"}
        n = await _kb_importer.load_existing_index()
        return {"success": True, "loaded": n}

    # ==================================================================
    # LLM Tool — 让 MaiBot 在对话中主动检索知识库
    # ==================================================================

    @Tool(
        "knowledge_search",
        description="检索本地知识库（原神/游戏/小说世界观等已导入的文档）。当用户询问世界观、剧情、角色、设定相关问题时调用。",
        brief_description="检索本地知识库（原神/游戏/小说世界观等）",
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="检索查询，例如 '法涅斯是谁' 或 '提瓦特七国'",
                required=True,
            ),
            ToolParameterInfo(
                name="top_k",
                param_type=ToolParamType.INTEGER,
                description="返回结果数量，默认 5",
                required=False,
            ),
        ],
    )
    async def tool_knowledge_search(
        self,
        query: str,
        top_k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        """LLM 工具：检索知识库，返回格式化文本供 LLM 引用。"""

        if _kb_searcher is None:
            return {
                "content": "知识库未初始化，无法检索",
                "found": False,
            }

        q = SearchQuery(query=query, top_k=top_k)
        hits = await _kb_searcher.search(q)

        if not hits:
            return {
                "content": f"未在知识库中找到与 '{query}' 相关的内容",
                "found": False,
                "count": 0,
            }

        # 格式化为 LLM 易读的文本
        lines = [f"找到 {len(hits)} 条与 '{query}' 相关的知识：", ""]
        for i, h in enumerate(hits, 1):
            source_path = " > ".join(h.title_path) if h.title_path else h.source_name or "未知来源"
            lines.append(f"### {i}. {h.heading or source_path}")
            lines.append(f"来源: {h.source_name or '未知'} | 章节: {source_path}")
            lines.append(f"相关度: vector={h.vector_score:.3f} bm25={h.bm25_score:.3f}")
            lines.append("")
            lines.append(h.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return {
            "content": "\n".join(lines),
            "found": True,
            "count": len(hits),
            "hits": [h.to_dict() for h in hits],
        }


__all__ = [
    "KbApiMixin",
    "init_kb",
    "load_kb_index",
    "close_kb",
]

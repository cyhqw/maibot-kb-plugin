"""kb — 知识库 RAG 模块

提供：
- Markdown / 纯文本语义切分（按标题层级，不靠字符数硬切）
- Embedding 抽象（MaiBot / OpenAI 兼容 / 哑元）
- 内存向量索引（numpy cosine）
- SQLite FTS5 全文检索（BM25）
- 混合检索 + RRF 融合
- 批量导入器（增量更新，基于 file_hash）

典型用法：

    from kb import (
        KnowledgeBaseImporter,
        HybridSearcher,
        VectorIndex,
        DummyEmbedder,
        SearchQuery,
    )

    # 1. 初始化
    index = VectorIndex()
    embedder = DummyEmbedder(dimension=256)
    importer = KnowledgeBaseImporter(db, index, embedder, "/path/to/kb")
    await importer.load_existing_index()  # 加载已有向量
    await importer.ingest_directory()     # 增量导入新文件

    # 2. 检索
    searcher = HybridSearcher(db, index, embedder)
    hits = await searcher.search(SearchQuery(query="法涅斯是什么"))
"""

from .chunker import Chunk, chunk_file, chunk_markdown, chunk_plain_text
from .embedder import (
    DummyEmbedder,
    Embedder,
    EmbeddingError,
    MaiBotEmbedder,
    OpenAICompatibleEmbedder,
)
from .importer import IngestResult, KnowledgeBaseImporter, SUPPORTED_EXTENSIONS
from .search import FusionMode, HybridSearcher, SearchHit, SearchQuery
from .vector_store import VectorIndex

# API mixin 延迟导入，避免循环引用
def _get_api_mixin():
    from .api import KbApiMixin, init_kb, load_kb_index, close_kb
    return KbApiMixin, init_kb, load_kb_index, close_kb


__all__ = [
    # chunker
    "Chunk",
    "chunk_file",
    "chunk_markdown",
    "chunk_plain_text",
    # embedder
    "Embedder",
    "EmbeddingError",
    "MaiBotEmbedder",
    "OpenAICompatibleEmbedder",
    "DummyEmbedder",
    # vector_store
    "VectorIndex",
    # search
    "SearchHit",
    "SearchQuery",
    "HybridSearcher",
    "FusionMode",
    # importer
    "KnowledgeBaseImporter",
    "IngestResult",
    "SUPPORTED_EXTENSIONS",
]

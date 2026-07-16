"""tests.test_kb — 端到端 KB 集成测试"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from maikb import close_db, get_db, init_db
from maikb.kb import (
    DummyEmbedder,
    HybridSearcher,
    KnowledgeBaseImporter,
    SearchQuery,
    VectorIndex,
    chunk_markdown,
)


@pytest.fixture
def kb_dir(tmp_path):
    d = tmp_path / "kb"
    d.mkdir()
    return d


@pytest.mark.asyncio
async def test_end_to_end_ingest_and_search(tmp_path, kb_dir):
    """端到端：导入 → 索引 → 检索。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    # 写入测试 markdown
    md = """# 蒙德

## 第二幕 月宫与葬火

法涅斯是原初之人之一。他降临提瓦特，创造了人类。

### 法涅斯的诞生

法涅斯从蛋中诞生，是第一个原初之人。他带着四个发光的影子一起降临。

### 法涅斯与龙族

法涅斯击败了七位龙王，包括尼伯龙根。尼伯龙根离开提瓦特去寻找外界的答案。

## 第三幕 高塔孤王

温迪在 2600 年前推翻了高塔孤王安德留斯，从此蒙德成为自由之城。
"""
    (kb_dir / "test.md").write_text(md)

    # 初始化 KB
    embedder = DummyEmbedder(dimension=128)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)

    # 导入
    result = await importer.ingest_directory()
    assert result.scanned == 1
    assert result.new == 1
    assert result.failed == 0
    assert result.chunks > 0

    # 索引大小应等于 chunk 数
    assert index.size == result.chunks

    # 检索
    searcher = HybridSearcher(db, index, embedder)
    hits = await searcher.search(SearchQuery(query="法涅斯", top_k=3))
    assert len(hits) >= 1
    # 至少一条结果的 content 包含 "法涅斯"
    assert any("法涅斯" in h.content for h in hits)

    await close_db()


@pytest.mark.asyncio
async def test_incremental_ingest(tmp_path, kb_dir):
    """增量导入：未变更的文件应被跳过。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    md = "# 章节\n\n这是一些内容。"
    f = kb_dir / "a.md"
    f.write_text(md)

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)

    # 第一次导入
    r1 = await importer.ingest_directory()
    assert r1.new == 1
    assert r1.unchanged == 0

    # 第二次导入（无变更）
    r2 = await importer.ingest_directory()
    assert r2.new == 0
    assert r2.unchanged == 1

    # 修改文件后第三次导入
    f.write_text("# 章节\n\n内容变了。")
    r3 = await importer.ingest_directory()
    assert r3.updated == 1
    assert r3.unchanged == 0

    await close_db()


@pytest.mark.asyncio
async def test_force_rebuild(tmp_path, kb_dir):
    """force_rebuild=True 时强制重新切分。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "a.md").write_text("# 章节\n\n内容。")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)

    await importer.ingest_directory()
    r = await importer.ingest_directory(force_rebuild=True)
    assert r.updated == 1
    assert r.unchanged == 0

    await close_db()


@pytest.mark.asyncio
async def test_multiple_files(tmp_path, kb_dir):
    """多个文件同时导入。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    for i in range(5):
        (kb_dir / f"f{i}.md").write_text(f"# 文件{i}\n\n内容 {i}。")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)

    r = await importer.ingest_directory()
    assert r.scanned == 5
    assert r.new == 5
    assert r.failed == 0

    # 检索应能命中
    searcher = HybridSearcher(db, index, embedder)
    hits = await searcher.search(SearchQuery(query="文件3", top_k=3))
    assert len(hits) >= 1

    await close_db()


@pytest.mark.asyncio
async def test_load_existing_index(tmp_path, kb_dir):
    """重新加载数据库时，向量索引应能从 DB 恢复。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "a.md").write_text("# 章节\n\n内容。")

    embedder = DummyEmbedder(dimension=64)
    index1 = VectorIndex()
    importer1 = KnowledgeBaseImporter(db, index1, embedder, kb_dir)
    await importer1.ingest_directory()
    assert index1.size > 0

    # 模拟"重启"：新建一个空索引，从 DB 加载
    index2 = VectorIndex()
    importer2 = KnowledgeBaseImporter(db, index2, embedder, kb_dir)
    loaded = await importer2.load_existing_index()
    assert loaded == index1.size
    assert index2.size == index1.size

    # 检索应该一样能用
    searcher = HybridSearcher(db, index2, embedder)
    hits = await searcher.search(SearchQuery(query="内容", top_k=3))
    assert len(hits) >= 1

    await close_db()


@pytest.mark.asyncio
async def test_bm25_search_chinese(tmp_path, kb_dir):
    """BM25 对中文短查询的兜底（trigram + LIKE）。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "test.md").write_text("""# 提瓦特

## 法涅斯

法涅斯是原初之人，降临提瓦特创造了人类。
""")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)
    await importer.ingest_directory()

    # 测试 BM25 路径（短查询走 LIKE）
    hits = await db.fts_search("法涅斯", limit=5)
    assert len(hits) >= 1

    # 测试 BM25 路径（长查询走 FTS5 MATCH）
    hits = await db.fts_search("法涅斯是原初之人", limit=5)
    assert len(hits) >= 1

    await close_db()


@pytest.mark.asyncio
async def test_search_only_vector(tmp_path, kb_dir):
    """关闭 BM25，仅走向量。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "test.md").write_text("# 章节\n\n法涅斯是原初之人。")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)
    await importer.ingest_directory()

    searcher = HybridSearcher(db, index, embedder)
    hits = await searcher.search(
        SearchQuery(query="法涅斯", top_k=3, use_vector=True, use_bm25=False)
    )
    assert len(hits) >= 1

    await close_db()


@pytest.mark.asyncio
async def test_search_only_bm25(tmp_path, kb_dir):
    """关闭向量，仅走 BM25。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "test.md").write_text("# 章节\n\n法涅斯是原初之人。")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)
    await importer.ingest_directory()

    searcher = HybridSearcher(db, index, embedder)
    hits = await searcher.search(
        SearchQuery(query="法涅斯", top_k=3, use_vector=False, use_bm25=True)
    )
    assert len(hits) >= 1
    # 全部 hits 应该有 bm25_score > 0
    assert all(h.bm25_score > 0 for h in hits)

    await close_db()


@pytest.mark.asyncio
async def test_delete_file(tmp_path, kb_dir):
    """删除文件后，chunks 和索引都应清除。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "a.md").write_text("# 章节\n\n内容。")
    (kb_dir / "b.md").write_text("# 章节2\n\n内容2。")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)
    await importer.ingest_directory()

    files = await db.list_kb_files(status="ready")
    assert len(files) == 2
    size_before = index.size

    # 删一个
    deleted = await db.delete_kb_file(files[0].file_id)
    assert deleted > 0

    files_after = await db.list_kb_files(status="ready")
    assert len(files_after) == 1

    await close_db()


@pytest.mark.asyncio
async def test_empty_query_returns_empty(tmp_path, kb_dir):
    """空查询返回空结果。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)
    db = get_db()

    (kb_dir / "test.md").write_text("# 章节\n\n内容。")

    embedder = DummyEmbedder(dimension=64)
    index = VectorIndex()
    importer = KnowledgeBaseImporter(db, index, embedder, kb_dir)
    await importer.ingest_directory()

    searcher = HybridSearcher(db, index, embedder)
    hits = await searcher.search(SearchQuery(query="", top_k=3))
    assert hits == []

    hits = await searcher.search(SearchQuery(query="   ", top_k=3))
    assert hits == []

    await close_db()

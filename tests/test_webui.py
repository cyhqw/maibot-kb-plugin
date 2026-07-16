"""tests.test_webui — Web UI 端到端测试"""

import pytest

from maikb import close_db, get_db, init_db
from maikb.kb import (
    DummyEmbedder,
    HybridSearcher,
    KnowledgeBaseImporter,
    SearchQuery,
    VectorIndex,
)
from maikb.kb.api import (
    _kb_embedder,
    _kb_importer,
    _kb_searcher,
    _kb_vector_index,
)
from maikb.webui import WebServer


@pytest.fixture
async def setup_kb(tmp_path):
    """初始化数据库 + KB + 启动 Web server。"""

    db_path = tmp_path / "test.db"
    await init_db(db_path)

    # 写一些测试文件
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("# 测试\n\n法涅斯是原初之人。")

    # 初始化 KB 模块（绕过 plugin.py，直接用 kb.api 的全局变量）
    import maikb.kb.api as kb_api

    embedder = DummyEmbedder(dimension=64, model_name="dummy-test")
    index = VectorIndex()
    importer = KnowledgeBaseImporter(get_db(), index, embedder, kb_dir)
    searcher = HybridSearcher(get_db(), index, embedder)

    # 注入到 kb.api 全局变量
    kb_api._kb_importer = importer
    kb_api._kb_searcher = searcher
    kb_api._kb_vector_index = index
    kb_api._kb_embedder = embedder

    await importer.ingest_directory()

    # 启动 Web server（用随机端口避免冲突）
    import socket

    def free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = free_port()

    # 构造一个 fake plugin 对象
    class FakePlugin:
        class ctx:
            class paths:
                data_dir = tmp_path

        def get_plugin_config_data(self):
            return {"database": {"enabled": True}, "knowledge_base": {"enabled": True}}

    server = WebServer(plugin=FakePlugin(), host="127.0.0.1", port=port, token="")
    await server.start()

    # 等 server 完全启动
    import asyncio as _asyncio
    await _asyncio.sleep(0.5)

    yield server, port

    await server.stop()
    # 清理 kb.api 全局变量
    kb_api._kb_importer = None
    kb_api._kb_searcher = None
    kb_api._kb_vector_index = None
    kb_api._kb_embedder = None
    await close_db()


@pytest.mark.asyncio
async def test_health_endpoint(setup_kb):
    """测试 /health 端点。"""

    import httpx

    server, port = setup_kb
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["service"] == "maikb-webui"


@pytest.mark.asyncio
async def test_index_page(setup_kb):
    """测试首页 HTML 返回。"""

    import httpx

    server, port = setup_kb
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/")
        assert resp.status_code == 200
        assert "MaiBot" in resp.text
        assert "知识库管理" in resp.text


@pytest.mark.asyncio
async def test_stats_endpoint(setup_kb):
    """测试 /api/stats 端点。"""

    import httpx

    server, port = setup_kb
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_total"] == 1
        assert data["files_ready"] == 1
        assert data["chunks_total"] > 0
        assert data["vector_index_size"] > 0
        assert data["embedding_model"] == "dummy-test"


@pytest.mark.asyncio
async def test_files_endpoint(setup_kb):
    """测试 /api/files 端点。"""

    import httpx

    server, port = setup_kb
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/api/files")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["items"][0]["file_name"] == "a.md"
        assert data["items"][0]["status"] == "ready"


@pytest.mark.asyncio
async def test_search_endpoint(setup_kb):
    """测试 /api/search 端点。"""

    import httpx

    server, port = setup_kb
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/api/search",
            json={"query": "法涅斯", "top_k": 3, "use_vector": True, "use_bm25": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert "法涅斯" in data["items"][0]["content"]


@pytest.mark.asyncio
async def test_config_endpoint(setup_kb):
    """测试 /api/config GET 端点。"""

    import httpx

    server, port = setup_kb
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "database" in data
        assert "knowledge_base" in data


@pytest.mark.asyncio
async def test_token_auth(tmp_path):
    """测试 token 认证。"""

    db_path = tmp_path / "auth.db"
    await init_db(db_path)

    import socket

    def free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = free_port()

    class FakePlugin:
        def get_plugin_config_data(self):
            return {}

    server = WebServer(plugin=FakePlugin(), host="127.0.0.1", port=port, token="secret123")
    await server.start()

    import asyncio as _asyncio
    await _asyncio.sleep(0.5)

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # 无 token → 401
            resp = await client.get(f"http://127.0.0.1:{port}/api/stats")
            assert resp.status_code == 401

            # 错误 token → 401
            resp = await client.get(
                f"http://127.0.0.1:{port}/api/stats",
                headers={"Authorization": "Bearer wrong"},
            )
            assert resp.status_code == 401

            # 正确 token → 200
            resp = await client.get(
                f"http://127.0.0.1:{port}/api/stats",
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200
    finally:
        await server.stop()
        await close_db()

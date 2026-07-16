"""maikb.webui.server

独立的 FastAPI Web 管理界面。

设计：
- 插件 on_load 时启动 uvicorn server（可配置端口）
- 监听 127.0.0.1 默认端口 8765（避开 MaiBot WebUI 的 8001）
- 简单 token 认证（可配置）
- 前端 HTML/JS 嵌入到 Python 字符串（避免分发多文件）

端点：
- GET  /              返回 SPA HTML 页面
- GET  /api/stats     知识库统计
- GET  /api/files     文件列表
- GET  /api/files/{file_id}/chunks  查看某文件的切片
- POST /api/search    检索测试
- POST /api/ingest    触发增量导入
- POST /api/rebuild   强制全量重建
- GET  /api/config    读取配置
- PUT  /api/config    更新配置（写 config.toml）
- GET  /api/health    健康检查
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


logger = logging.getLogger("maikb.webui")


# ----------------------------------------------------------------------
# 请求/响应模型
# ----------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_vector: bool = True
    use_bm25: bool = True
    category: Optional[str] = None


class IngestRequest(BaseModel):
    force_rebuild: bool = False


class ConfigUpdateRequest(BaseModel):
    config: dict[str, Any]


# ----------------------------------------------------------------------
# Web 服务器
# ----------------------------------------------------------------------

class WebServer:
    """独立的 FastAPI Web 管理 server。"""

    def __init__(
        self,
        plugin,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str = "",
    ) -> None:
        self._plugin = plugin
        self._host = host
        self._port = port
        self._token = token
        self._app: Optional[FastAPI] = None
        self._server: Any = None  # uvicorn.Server
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """启动 server。"""

        if self._server is not None:
            return

        # 懒加载 uvicorn
        try:
            import uvicorn
        except ImportError as exc:
            logger.error(f"启动 Web UI 失败：未安装 uvicorn，请 pip install uvicorn: {exc}")
            return

        self._app = self._build_app()

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        logger.info(f"Web UI 已启动: http://{self._host}:{self._port}")

    async def stop(self) -> None:
        """停止 server。"""

        if self._server is not None:
            self._server.should_exit = True
            if self._task is not None:
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._task.cancel()
            self._server = None
            self._task = None
            logger.info("Web UI 已停止")

    # ------------------------------------------------------------------
    # FastAPI app 构建
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        from fastapi import Depends, Header
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(
            title="MaiBot Knowledge Base - Admin",
            docs_url="/api/docs",
            openapi_url="/api/openapi.json",
        )

        # CORS（便于本地开发时前端跨域调试）
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 认证依赖
        async def verify_token(authorization: Optional[str] = Header(None)) -> None:
            if not self._token:
                return  # 未配置 token，跳过认证
            if not authorization:
                raise HTTPException(status_code=401, detail="Missing Authorization header")
            # 期望 "Bearer <token>"
            parts = authorization.split(" ", 1)
            token = parts[1] if len(parts) == 2 else parts[0]
            if token != self._token:
                raise HTTPException(status_code=401, detail="Invalid token")

        # ------------------------------------------------------------------
        # 页面路由
        # ------------------------------------------------------------------

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(content=_INDEX_HTML)

        @app.get("/health")
        async def health() -> dict:
            return {"ok": True, "service": "maikb-webui"}

        # ------------------------------------------------------------------
        # 知识库 API
        # ------------------------------------------------------------------

        @app.get("/api/stats", dependencies=[Depends(verify_token)])
        async def stats() -> dict:
            return await self._handle_stats()

        @app.get("/api/files", dependencies=[Depends(verify_token)])
        async def list_files(
            status: Optional[str] = None,
            category: Optional[str] = None,
        ) -> dict:
            return await self._handle_list_files(status, category)

        @app.get("/api/files/{file_id}/chunks", dependencies=[Depends(verify_token)])
        async def file_chunks(file_id: str) -> dict:
            return await self._handle_file_chunks(file_id)

        @app.delete("/api/files/{file_id}", dependencies=[Depends(verify_token)])
        async def delete_file(file_id: str) -> dict:
            return await self._handle_delete_file(file_id)

        @app.get("/api/categories", dependencies=[Depends(verify_token)])
        async def categories() -> dict:
            return await self._handle_categories()

        @app.post("/api/upload", dependencies=[Depends(verify_token)])
        async def upload(file: UploadFile = File(...)) -> dict:
            return await self._handle_upload(file)

        @app.post("/api/search", dependencies=[Depends(verify_token)])
        async def search(req: SearchRequest) -> dict:
            return await self._handle_search(req)

        @app.post("/api/ingest", dependencies=[Depends(verify_token)])
        async def ingest(req: IngestRequest) -> dict:
            return await self._handle_ingest(req)

        @app.post("/api/rebuild", dependencies=[Depends(verify_token)])
        async def rebuild() -> dict:
            return await self._handle_rebuild()

        @app.get("/api/config", dependencies=[Depends(verify_token)])
        async def get_config() -> dict:
            return await self._handle_get_config()

        @app.put("/api/config", dependencies=[Depends(verify_token)])
        async def update_config(req: ConfigUpdateRequest) -> dict:
            return await self._handle_update_config(req.config)

        return app

    # ------------------------------------------------------------------
    # 请求处理器
    # ------------------------------------------------------------------

    async def _handle_stats(self) -> dict:
        from ..kb.api import _kb_importer, _kb_vector_index, _kb_embedder
        from .. import get_db

        db = get_db()
        all_files = await db.list_kb_files()
        ready_files = [f for f in all_files if f.status == "ready"]
        failed_files = [f for f in all_files if f.status == "failed"]
        total_chunks = sum(f.chunk_count for f in ready_files)
        total_tokens = sum(f.total_tokens for f in ready_files)
        total_size = sum(f.file_size for f in ready_files)

        return {
            "files_total": len(all_files),
            "files_ready": len(ready_files),
            "files_failed": len(failed_files),
            "chunks_total": total_chunks,
            "tokens_total": total_tokens,
            "size_bytes": total_size,
            "size_human": _human_size(total_size),
            "vector_index_size": _kb_vector_index.size if _kb_vector_index else 0,
            "embedding_model": _kb_embedder.model_name if _kb_embedder else None,
            "embedding_dimension": _kb_embedder.dimension if _kb_embedder else 0,
        }

    async def _handle_list_files(
        self, status: Optional[str], category: Optional[str]
    ) -> dict:
        from .. import get_db

        db = get_db()
        files = await db.list_kb_files(status=status, category=category)
        return {
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
                    "size_human": _human_size(f.file_size),
                    "last_ingested_at": f.last_ingested_at.isoformat() if f.last_ingested_at else None,
                    "error": f.error,
                }
                for f in files
            ],
        }

    async def _handle_file_chunks(self, file_id: str) -> dict:
        from .. import get_db
        from sqlmodel import select
        from ..models import KnowledgeChunk

        db = get_db()
        f = await db.get_kb_file_by_id(file_id)
        if f is None:
            raise HTTPException(status_code=404, detail="File not found")

        async with db.get_db() as session:
            stmt = (
                select(KnowledgeChunk)
                .where(KnowledgeChunk.file_id == file_id)
                .order_by(KnowledgeChunk.chunk_index.asc())
            )
            result = await session.execute(stmt)
            chunks = list(result.scalars().all())

        return {
            "file_id": file_id,
            "file_path": f.file_path,
            "count": len(chunks),
            "items": [
                {
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "title_path": c.title_path,
                    "heading": c.heading,
                    "content": c.content,
                    "char_count": c.char_count,
                    "token_count": c.token_count,
                    "has_embedding": c.embedding is not None,
                    "embedding_model": c.embedding_model,
                    "embedded_at": c.embedded_at.isoformat() if c.embedded_at else None,
                }
                for c in chunks
            ],
        }

    async def _handle_delete_file(self, file_id: str) -> dict:
        """删除文件及其 chunks、FTS、内存向量索引。"""

        from ..kb.api import _kb_vector_index

        db = get_db()
        # 先查 chunk_ids 用于从内存索引删除
        from sqlmodel import select
        from ..models import KnowledgeChunk

        async with db.get_db() as session:
            stmt = select(KnowledgeChunk.chunk_id).where(KnowledgeChunk.file_id == file_id)
            rows = (await session.execute(stmt)).fetchall()
        chunk_ids = [r[0] for r in rows]

        deleted = await db.delete_kb_file(file_id)
        if _kb_vector_index is not None:
            for cid in chunk_ids:
                _kb_vector_index.remove(cid)

        return {"success": True, "deleted_chunks": deleted}

    async def _handle_categories(self) -> dict:
        """返回知识库中出现的所有 category（去重）。"""

        from .. import get_db

        db = get_db()
        all_files = await db.list_kb_files()
        cats: list[str] = []
        seen: set[str] = set()
        for f in all_files:
            c = f.category or ""
            if c and c not in seen:
                seen.add(c)
                cats.append(c)
        return {"categories": cats}

    async def _handle_upload(self, file: UploadFile) -> dict:
        """上传 .md/.txt 文件到 knowledge_dir 并增量导入。"""

        from ..kb.api import _kb_importer

        kb_cfg = self._plugin.config.knowledge_base
        kb_dir = self._plugin.ctx.paths.data_dir / kb_cfg.knowledge_dir
        kb_dir.mkdir(parents=True, exist_ok=True)

        filename = file.filename or "upload.txt"
        # 安全：仅取文件名，防止路径穿越
        safe_name = Path(filename).name
        if not safe_name:
            raise HTTPException(status_code=400, detail="非法文件名")
        lower = safe_name.lower()
        if not (lower.endswith(".md") or lower.endswith(".markdown") or lower.endswith(".txt")):
            raise HTTPException(status_code=400, detail="仅支持 .md / .markdown / .txt 文件")

        dest = kb_dir / safe_name
        content = await file.read()
        dest.write_bytes(content)

        # 触发增量导入
        if _kb_importer is not None:
            try:
                result = await _kb_importer.ingest_directory(force_rebuild=False)
                return {
                    "success": True,
                    "saved_as": safe_name,
                    "size_bytes": len(content),
                    "ingest": result.to_dict(),
                }
            except Exception as exc:
                logger.warning(f"上传后导入失败: {exc}")
                return {"success": True, "saved_as": safe_name, "ingest_error": str(exc)}

        return {"success": True, "saved_as": safe_name, "ingest": None}

    async def _handle_search(self, req: SearchRequest) -> dict:
        from ..kb.api import _kb_searcher
        from ..kb import SearchQuery

        if _kb_searcher is None:
            raise HTTPException(status_code=503, detail="KB module not initialized")

        q = SearchQuery(
            query=req.query,
            top_k=req.top_k,
            use_vector=req.use_vector,
            use_bm25=req.use_bm25,
            category=req.category,
        )
        hits = await _kb_searcher.search(q)
        return {
            "query": req.query,
            "count": len(hits),
            "items": [
                {
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "content": h.content,
                    "heading": h.heading,
                    "title_path": h.title_path,
                    "source_name": h.source_name,
                    "vector_score": h.vector_score,
                    "bm25_score": h.bm25_score,
                }
                for h in hits
            ],
        }

    async def _handle_ingest(self, req: IngestRequest) -> dict:
        from ..kb.api import _kb_importer

        if _kb_importer is None:
            raise HTTPException(status_code=503, detail="KB module not initialized")

        result = await _kb_importer.ingest_directory(force_rebuild=req.force_rebuild)
        return {"success": True, **result.to_dict()}

    async def _handle_rebuild(self) -> dict:
        from ..kb.api import _kb_importer

        if _kb_importer is None:
            raise HTTPException(status_code=503, detail="KB module not initialized")

        result = await _kb_importer.ingest_directory(force_rebuild=True)
        return {"success": True, **result.to_dict()}

    async def _handle_get_config(self) -> dict:
        """读取插件当前配置（序列化 config_model）。

        优先用 config_model.model_dump()；若 plugin 对象只暴露旧式
        get_plugin_config_data()（例如测试桩），则回退到它。
        """

        try:
            cfg = getattr(self._plugin, "config", None)
            if cfg is not None and hasattr(cfg, "model_dump"):
                return cfg.model_dump(mode="json")
            if cfg is not None and hasattr(cfg, "dict"):
                return cfg.dict()
            if hasattr(self._plugin, "get_plugin_config_data"):
                return self._plugin.get_plugin_config_data()
            return {}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    async def _handle_update_config(self, config: dict[str, Any]) -> dict:
        """更新配置：直接写 config.toml，触发 MaiBot 的 FileWatcher 热重载。"""

        try:
            import tomlkit
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="tomlkit not installed, please pip install tomlkit",
            )

        # 配置文件路径：data/plugins/<plugin_id>/config.toml
        config_path = self._plugin.ctx.paths.data_dir / "config.toml"

        # 读现有 config（保留注释和格式）
        if config_path.exists():
            doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
        else:
            doc = tomlkit.document()

        # 合并新配置（顶层 key）
        for k, v in config.items():
            # 跳过 None
            if v is None:
                continue
            # 转换 Python 类型为 tomlkit 兼容
            doc[k] = _to_toml_value(v)

        config_path.write_text(tomlkit.dumps(doc), encoding="utf-8")

        return {
            "success": True,
            "message": "配置已写入，FileWatcher 将在数百 ms 内触发热重载",
            "path": str(config_path),
        }


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _human_size(num: int) -> str:
    """字节数转人类可读。"""

    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _to_toml_value(v: Any) -> Any:
    """转换 Python 值为 tomlkit 兼容类型。"""

    if isinstance(v, dict):
        table = tomlkit.table()
        for k, sub in v.items():
            table[k] = _to_toml_value(sub)
        return table
    if isinstance(v, list):
        return [_to_toml_value(x) for x in v]
    return v


# ----------------------------------------------------------------------
# 嵌入式 HTML 前端
# ----------------------------------------------------------------------

_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>知识库管理</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
:root {
  --bg: #f5f7fa; --surface: #ffffff; --surface-2: #f8f9fc; --surface-3: #eef1f6;
  --text: #2c3e50; --text-muted: #888; --border: #e8ecf3;
  --brand: #667eea; --brand-dark: #5568d3; --danger: #e74c3c; --danger-dark: #c0392b;
  --ok: #27ae60; --warn: #f39c12;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
}
:root[data-theme="dark"] {
  --bg: #1a1b26; --surface: #24283b; --surface-2: #2d3250; --surface-3: #3a3f5c;
  --text: #c0caf5; --text-muted: #8b95b8; --border: #3a3f5c;
  --brand: #7c8ff0; --brand-dark: #667eea; --danger: #e06c75; --danger-dark: #c5505a;
  --ok: #98c379; --warn: #e5c07b;
  --shadow: 0 1px 3px rgba(0,0,0,0.3);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.6; transition: background 0.2s, color 0.2s;
}
header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white; padding: 18px 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  display: flex; align-items: center; justify-content: space-between;
}
header h1 { font-size: 21px; margin-bottom: 2px; }
header .subtitle { font-size: 13px; opacity: 0.9; }
.theme-btn {
  background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.3); color: white;
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 14px;
}
.theme-btn:hover { background: rgba(255,255,255,0.28); }
.container { max-width: 1400px; margin: 20px auto; padding: 0 20px; }
.tabs { display: flex; gap: 4px; margin-bottom: 16px; background: var(--surface); padding: 6px; border-radius: 8px; box-shadow: var(--shadow); flex-wrap: wrap; }
.tab {
  padding: 8px 18px; background: transparent; border: none; cursor: pointer; border-radius: 6px;
  font-size: 14px; color: var(--text-muted); transition: all 0.2s;
}
.tab:hover { background: var(--surface-3); }
.tab.active { background: var(--brand); color: white; }
.panel { background: var(--surface); border-radius: 8px; padding: 24px; box-shadow: var(--shadow); margin-bottom: 16px; }
.panel h2 { font-size: 16px; margin-bottom: 16px; color: var(--text); }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.stat-card { background: var(--surface-2); padding: 16px; border-radius: 8px; border-left: 3px solid var(--brand); }
.stat-card .label { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }
.stat-card .value { font-size: 24px; font-weight: 600; color: var(--text); }
.stat-card .unit { font-size: 12px; color: var(--text-muted); margin-left: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
th { background: var(--surface-2); font-weight: 600; color: var(--text-muted); }
tr:hover { background: var(--surface-2); }
.status-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.status-ready { background: rgba(39,174,96,0.15); color: var(--ok); }
.status-failed { background: rgba(231,76,60,0.15); color: var(--danger); }
.status-pending { background: rgba(243,156,18,0.15); color: var(--warn); }
.status-processing { background: rgba(102,126,234,0.15); color: var(--brand); }
.cat-badge { display: inline-block; padding: 1px 7px; border-radius: 8px; font-size: 11px; background: var(--surface-3); color: var(--text-muted); }
.btn { padding: 6px 14px; background: var(--brand); color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; transition: background 0.2s; }
.btn:hover { background: var(--brand-dark); }
.btn-danger { background: var(--danger); }
.btn-danger:hover { background: var(--danger-dark); }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.btn-ghost { background: transparent; color: var(--text-muted); border: 1px solid var(--border); }
.btn-ghost:hover { background: var(--surface-3); }
.search-box { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.search-box input, .search-box select, .filter-input, .filter-select {
  padding: 8px 12px; border: 1px solid var(--border); border-radius: 4px; font-size: 14px;
  background: var(--surface); color: var(--text);
}
.search-box input:focus, .filter-input:focus { outline: none; border-color: var(--brand); }
.hit-card { border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; background: var(--surface-2); }
.hit-card .hit-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 12px; color: var(--text-muted); }
.hit-card .hit-title { font-weight: 600; color: var(--text); font-size: 14px; }
.hit-card .hit-content { font-size: 13px; color: var(--text); white-space: pre-wrap; max-height: 200px; overflow-y: auto; background: var(--surface); padding: 8px; border-radius: 4px; border: 1px solid var(--border); }
.hit-card .hit-scores { font-family: monospace; font-size: 11px; color: var(--text-muted); }
.toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 20px; background: var(--text); color: var(--surface); border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); z-index: 1000; opacity: 0; transition: opacity 0.3s; }
.toast.show { opacity: 1; }
.toast.error { background: var(--danger); color: white; }
.toast.success { background: var(--ok); color: white; }
.loading { text-align: center; padding: 40px; color: var(--text-muted); }
.spinner { border: 3px solid var(--surface-3); border-top: 3px solid var(--brand); border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
@keyframes spin { 0% { transform: rotate(0); } 100% { transform: rotate(360deg); } }
.actions-bar { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; flex-wrap: wrap; }
.config-editor { width: 100%; min-height: 400px; padding: 12px; font-family: "SF Mono", Monaco, "Cascadia Code", monospace; font-size: 13px; border: 1px solid var(--border); border-radius: 4px; background: var(--surface-2); color: var(--text); resize: vertical; }
.muted { color: var(--text-muted); font-size: 12px; }
.modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 999; justify-content: center; align-items: flex-start; padding: 30px 20px; overflow-y: auto; }
.modal-overlay.show { display: flex; }
.modal { background: var(--surface); border-radius: 8px; max-width: 900px; width: 100%; padding: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.modal-header h2 { font-size: 16px; color: var(--text); }
.modal-close { background: none; border: none; font-size: 22px; cursor: pointer; color: var(--text-muted); }
.modal-close:hover { color: var(--text); }
.upload-wrap { display: inline-flex; align-items: center; gap: 6px; }
input[type=file] { display: none; }
</style>
</head>
<body>
<header>
  <div>
    <h1>📚 MaiBot · 知识库管理</h1>
    <div class="subtitle">MaiBot 插件 · 向量/BM25 混合检索 RAG</div>
  </div>
  <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">🌙</button>
</header>

<div class="container">
  <div class="actions-bar">
    <span class="muted">Token (如配置):</span>
    <input type="password" id="tokenInput" class="filter-input" placeholder="留空则无认证" style="width:200px">
    <button class="btn btn-sm" onclick="saveToken()">保存</button>
    <span style="flex:1"></span>
    <button class="btn btn-sm btn-ghost" onclick="refreshAll()">🔄 刷新</button>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab(this, 'stats')">📊 统计</button>
    <button class="tab" onclick="switchTab(this, 'files')">📁 文件</button>
    <button class="tab" onclick="switchTab(this, 'search')">🔍 检索测试</button>
    <button class="tab" onclick="switchTab(this, 'config')">⚙️ 配置</button>
  </div>

  <div id="panel-stats" class="panel">
    <h2>知识库统计</h2>
    <div id="statsContent" class="stats-grid"><div class="loading"><div class="spinner"></div>加载中...</div></div>
    <div style="margin-top:20px">
      <button class="btn" onclick="ingest(false)">📥 增量导入</button>
      <button class="btn btn-danger" onclick="ingest(true)" style="margin-left:8px">🔧 强制全量重建</button>
    </div>
  </div>

  <div id="panel-files" class="panel" style="display:none">
    <h2>知识库文件</h2>
    <div class="actions-bar">
      <input type="text" id="fileFilter" class="filter-input" placeholder="按文件名筛选..." style="flex:1;min-width:160px">
      <select id="fileStatusFilter" class="filter-select">
        <option value="">全部状态</option>
        <option value="ready">ready</option>
        <option value="failed">failed</option>
        <option value="pending">pending</option>
        <option value="processing">processing</option>
      </select>
      <select id="fileCategoryFilter" class="filter-select">
        <option value="">全部分类</option>
      </select>
      <button class="btn btn-sm" onclick="loadFiles()">筛选</button>
      <span style="flex:1"></span>
      <div class="upload-wrap">
        <input type="file" id="uploadInput" accept=".md,.markdown,.txt" onchange="uploadFile(event)">
        <button class="btn btn-sm" onclick="document.getElementById('uploadInput').click()">📤 上传文件</button>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr><th>文件名</th><th>分类</th><th>状态</th><th>Chunks</th><th>Tokens</th><th>大小</th><th>最后导入</th><th>操作</th></tr></thead>
        <tbody id="filesTable"><tr><td colspan="8" class="loading"><div class="spinner"></div>加载中...</td></tr></tbody>
      </table>
    </div>
  </div>

  <div id="panel-search" class="panel" style="display:none">
    <h2>检索测试</h2>
    <div class="search-box">
      <input type="text" id="searchQuery" placeholder="输入查询，如：法涅斯是谁" onkeydown="if(event.key==='Enter')doSearch()" style="flex:1;min-width:200px">
      <select id="searchMode" class="filter-select">
        <option value="hybrid">混合（推荐）</option>
        <option value="vector">仅向量</option>
        <option value="bm25">仅 BM25</option>
      </select>
      <select id="searchTopK" class="filter-select">
        <option value="3">Top 3</option>
        <option value="5" selected>Top 5</option>
        <option value="10">Top 10</option>
      </select>
      <button class="btn" onclick="doSearch()">🔍 检索</button>
    </div>
    <div id="searchResults"></div>
  </div>

  <div id="panel-config" class="panel" style="display:none">
    <h2>插件配置</h2>
    <p class="muted" style="margin-bottom:12px">编辑 JSON 后点击"保存配置"，将写入 config.toml 并触发插件热重载。</p>
    <textarea id="configEditor" class="config-editor" placeholder="加载中..."></textarea>
    <div style="margin-top:12px">
      <button class="btn" onclick="saveConfig()">💾 保存配置</button>
      <button class="btn btn-sm btn-ghost" onclick="loadConfig()" style="margin-left:8px">🔄 重新加载</button>
    </div>
  </div>
</div>

<div id="chunkModal" class="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modalTitle">切片详情</h2>
      <button class="modal-close" onclick="closeModal()">×</button>
    </div>
    <div id="modalBody"></div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
let savedToken = localStorage.getItem('maikb_token') || '';
let savedTheme = localStorage.getItem('maikb_theme') || '';
if (savedTheme === 'dark') { document.documentElement.setAttribute('data-theme','dark'); document.getElementById('themeBtn').textContent = '☀️'; }
document.getElementById('tokenInput').value = savedToken;
refreshAll();

function toggleTheme() {
  const root = document.documentElement;
  if (root.getAttribute('data-theme') === 'dark') {
    root.removeAttribute('data-theme'); localStorage.setItem('maikb_theme','light');
    document.getElementById('themeBtn').textContent = '🌙';
  } else {
    root.setAttribute('data-theme','dark'); localStorage.setItem('maikb_theme','dark');
    document.getElementById('themeBtn').textContent = '☀️';
  }
}

function saveToken() {
  savedToken = document.getElementById('tokenInput').value.trim();
  localStorage.setItem('maikb_token', savedToken);
  toast('Token 已保存', 'success');
  refreshAll();
}

function headers() {
  const h = {'Content-Type': 'application/json'};
  if (savedToken) h['Authorization'] = 'Bearer ' + savedToken;
  return h;
}

async function api(path, opts = {}) {
  const resp = await fetch(path, {...opts, headers: {...headers(), ...(opts.headers || {})}});
  if (!resp.ok) {
    let msg = resp.status + ' ' + resp.statusText;
    try { const j = await resp.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return resp.json();
}

function toast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.className = 'toast', 3000);
}

function switchTab(btn, name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('[id^=panel-]').forEach(p => p.style.display = 'none');
  btn.classList.add('active');
  document.getElementById('panel-' + name).style.display = 'block';
  if (name === 'stats') loadStats();
  if (name === 'files') { loadCategories(); loadFiles(); }
  if (name === 'config') loadConfig();
}

function refreshAll() { loadStats(); }

async function loadStats() {
  try {
    const s = await api('/api/stats');
    document.getElementById('statsContent').innerHTML = `
      <div class="stat-card"><div class="label">文件总数</div><div class="value">${s.files_total}</div></div>
      <div class="stat-card"><div class="label">成功导入</div><div class="value" style="color:var(--ok)">${s.files_ready}</div></div>
      <div class="stat-card"><div class="label">失败</div><div class="value" style="color:var(--danger)">${s.files_failed}</div></div>
      <div class="stat-card"><div class="label">总 chunks</div><div class="value">${s.chunks_total}</div></div>
      <div class="stat-card"><div class="label">总 tokens</div><div class="value">${s.tokens_total}</div></div>
      <div class="stat-card"><div class="label">总大小</div><div class="value">${s.size_human}</div></div>
      <div class="stat-card"><div class="label">内存索引</div><div class="value">${s.vector_index_size}</div></div>
      <div class="stat-card"><div class="label">Embedding</div><div class="value" style="font-size:14px">${s.embedding_model || '-'}<span class="unit">${s.embedding_dimension}d</span></div></div>
    `;
  } catch (e) {
    document.getElementById('statsContent').innerHTML = '<div class="stat-card" style="grid-column:1/-1">加载失败: ' + e.message + '</div>';
  }
}

async function ingest(force) {
  if (force && !confirm('确认要强制全量重建？这可能需要较长时间。')) return;
  toast(force ? '开始全量重建...' : '开始增量导入...');
  try {
    const r = await api('/api/' + (force ? 'rebuild' : 'ingest'), {method: 'POST', body: JSON.stringify({force_rebuild: force})});
    toast('完成: new=' + r.new + ' updated=' + r.updated + ' unchanged=' + r.unchanged + ' failed=' + r.failed + ' chunks=' + r.chunks, 'success');
    loadStats();
  } catch (e) { toast('失败: ' + e.message, 'error'); }
}

async function loadCategories() {
  try {
    const r = await api('/api/categories');
    const sel = document.getElementById('fileCategoryFilter');
    const cur = sel.value;
    sel.innerHTML = '<option value="">全部分类</option>' + r.categories.map(c => '<option value="' + esc(c) + '">' + esc(c) + '</option>').join('');
    sel.value = cur;
  } catch (e) {}
}

async function loadFiles() {
  const status = document.getElementById('fileStatusFilter').value;
  const category = document.getElementById('fileCategoryFilter').value;
  const filter = document.getElementById('fileFilter').value.toLowerCase();
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  if (category) params.set('category', category);
  try {
    const r = await api('/api/files' + (params.toString() ? '?' + params.toString() : ''));
    let items = r.items;
    if (filter) items = items.filter(f => (f.file_name || '').toLowerCase().includes(filter));
    if (items.length === 0) {
      document.getElementById('filesTable').innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--text-muted)">无文件</td></tr>';
      return;
    }
    document.getElementById('filesTable').innerHTML = items.map(f => `
      <tr>
        <td title="${esc(f.file_path)}">${esc(f.file_name)}<div class="muted">${esc(f.title || '')}</div></td>
        <td>${f.category ? '<span class="cat-badge">' + esc(f.category) + '</span>' : '<span class="muted">-</span>'}</td>
        <td><span class="status-badge status-${esc(f.status)}">${esc(f.status)}</span></td>
        <td>${f.chunk_count}</td>
        <td>${f.total_tokens}</td>
        <td>${f.size_human}</td>
        <td class="muted">${f.last_ingested_at ? new Date(f.last_ingested_at).toLocaleString() : '-'}</td>
        <td>
          <button class="btn btn-sm" onclick="viewChunks('${esc(f.file_id)}')">切片</button>
          <button class="btn btn-sm btn-danger" onclick="deleteFile('${esc(f.file_id)}','${esc(f.file_name)}')" style="margin-left:4px">删除</button>
        </td>
      </tr>
    `).join('');
  } catch (e) {
    document.getElementById('filesTable').innerHTML = '<tr><td colspan="8" style="color:var(--danger)">加载失败: ' + esc(e.message) + '</td></tr>';
  }
}

async function viewChunks(fileId) {
  try {
    const r = await api('/api/files/' + fileId + '/chunks');
    document.getElementById('modalTitle').textContent = '切片详情（' + r.count + ' 个）';
    let html = '<p class="muted" style="margin-bottom:12px">' + esc(r.file_path) + '</p>';
    if (r.count === 0) {
      html += '<p class="muted">无切片</p>';
    } else {
      r.items.forEach(c => {
        html += `
          <div class="hit-card">
            <div class="hit-header">
              <span class="hit-title">#${c.chunk_index} ${esc(c.heading || '')}</span>
              <span>${c.char_count} chars / ${c.token_count} tokens ${c.has_embedding ? '✓ embedded' : '⚠ no embedding'}</span>
            </div>
            <div class="muted" style="margin-bottom:6px">${esc((c.title_path || []).join(' > '))}</div>
            <div class="hit-content">${esc(c.content)}</div>
          </div>`;
      });
    }
    document.getElementById('modalBody').innerHTML = html;
    document.getElementById('chunkModal').classList.add('show');
  } catch (e) { toast('加载切片失败: ' + e.message, 'error'); }
}

function closeModal() { document.getElementById('chunkModal').classList.remove('show'); }

async function deleteFile(fileId, name) {
  if (!confirm('确认删除文件 "' + name + '" 及其所有切片？此操作不可撤销。')) return;
  try {
    await api('/api/files/' + fileId, {method: 'DELETE'});
    toast('已删除: ' + name, 'success');
    loadFiles(); loadStats();
  } catch (e) { toast('删除失败: ' + e.message, 'error'); }
}

async function uploadFile(event) {
  const input = event.target;
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);
  toast('上传中: ' + file.name + '...');
  try {
    const resp = await fetch('/api/upload', {method: 'POST', headers: savedToken ? {'Authorization': 'Bearer ' + savedToken} : {}, body: formData});
    if (!resp.ok) {
      let msg = resp.status; try { msg = (await resp.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    const r = await resp.json();
    toast('上传成功: ' + r.saved_as, 'success');
    loadFiles(); loadStats(); loadCategories();
  } catch (e) { toast('上传失败: ' + e.message, 'error'); }
  input.value = '';
}

function esc(s) { return (s == null ? '' : String(s)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function doSearch() {
  const q = document.getElementById('searchQuery').value.trim();
  if (!q) { toast('请输入查询', 'error'); return; }
  const mode = document.getElementById('searchMode').value;
  const topK = parseInt(document.getElementById('searchTopK').value);
  const useVector = mode !== 'bm25';
  const useBM25 = mode !== 'vector';
  document.getElementById('searchResults').innerHTML = '<div class="loading"><div class="spinner"></div>检索中...</div>';
  try {
    const r = await api('/api/search', {method: 'POST', body: JSON.stringify({query: q, top_k: topK, use_vector: useVector, use_bm25: useBM25})});
    if (r.count === 0) { document.getElementById('searchResults').innerHTML = '<p style="text-align:center;padding:20px;color:var(--text-muted)">未找到相关结果</p>'; return; }
    document.getElementById('searchResults').innerHTML = r.items.map(h => `
      <div class="hit-card">
        <div class="hit-header">
          <span class="hit-title">${esc(h.heading || (h.title_path || []).join(' > ') || '-')}</span>
          <span class="hit-scores">score=${h.score.toFixed(4)} vec=${h.vector_score.toFixed(3)} bm25=${h.bm25_score.toFixed(3)}</span>
        </div>
        <div class="muted" style="margin-bottom:6px">来源: ${esc(h.source_name || '-')} | ${esc((h.title_path || []).join(' > '))}</div>
        <div class="hit-content">${esc(h.content)}</div>
      </div>`).join('');
  } catch (e) {
    document.getElementById('searchResults').innerHTML = '<p style="color:var(--danger)">检索失败: ' + esc(e.message) + '</p>';
  }
}

async function loadConfig() {
  try {
    const c = await api('/api/config');
    document.getElementById('configEditor').value = JSON.stringify(c, null, 2);
  } catch (e) { document.getElementById('configEditor').value = '加载失败: ' + e.message; }
}

async function saveConfig() {
  const text = document.getElementById('configEditor').value;
  let config;
  try { config = JSON.parse(text); } catch (e) { toast('JSON 格式错误: ' + e.message, 'error'); return; }
  try {
    await api('/api/config', {method: 'PUT', body: JSON.stringify({config: config})});
    toast('配置已保存，插件将热重载', 'success');
  } catch (e) { toast('保存失败: ' + e.message, 'error'); }
}
</script>
</body>
</html>
"""


__all__ = ["WebServer"]

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

import tomlkit
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
        """读取插件配置。

        优先从 config.toml 文件读（保存后立即生效）；
        若文件不存在则回退到内存 config_model / get_plugin_config_data。
        """

        try:
            config_path = self._plugin.ctx.paths.data_dir / "config.toml"
            if config_path.exists():
                import tomllib
                with open(config_path, "rb") as f:
                    return tomllib.load(f)
            # 回退到内存
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
  --bg: #f0f0f0; --surface: #fff; --surface-2: #f7f7f8; --surface-3: #e8e8ea;
  --text: #1a1a1a; --text-2: #555; --text-3: #999; --border: #ddd;
  --accent: #2563eb; --accent-h: #1d4ed8; --danger: #dc2626; --danger-h: #b91c1c;
  --ok: #16a34a; --warn: #d97706;
}
:root[data-theme="dark"] {
  --bg: #16161a; --surface: #1e1e24; --surface-2: #25252c; --surface-3: #33333c;
  --text: #e4e4e7; --text-2: #a1a1aa; --text-3: #71717a; --border: #33333c;
  --accent: #3b82f6; --accent-h: #2563eb; --danger: #ef4444; --danger-h: #dc2626;
  --ok: #22c55e; --warn: #f59e0b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
header { background: #1e293b; color: #e2e8f0; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid var(--accent); }
header h1 { font-size: 16px; font-weight: 600; }
header .sub { font-size: 12px; color: #94a3b8; }
.theme-btn { background: transparent; border: 1px solid #475569; color: #94a3b8; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.theme-btn:hover { border-color: #64748b; color: #cbd5e1; }
.wrap { max-width: 1200px; margin: 16px auto; padding: 0 16px; }
.toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
.toolbar input, .toolbar select { padding: 5px 8px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; background: var(--surface); color: var(--text); }
.toolbar input:focus { outline: none; border-color: var(--accent); }
.btn { padding: 5px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; background: var(--accent); color: #fff; }
.btn:hover { background: var(--accent-h); }
.btn-danger { background: var(--danger); }
.btn-danger:hover { background: var(--danger-h); }
.btn-ghost { background: transparent; color: var(--text-2); border: 1px solid var(--border); }
.btn-ghost:hover { background: var(--surface-3); }
.btn-sm { padding: 3px 8px; font-size: 12px; }
.tabs { display: flex; gap: 2px; border-bottom: 2px solid var(--border); margin-bottom: 16px; }
.tab { padding: 8px 16px; background: transparent; border: none; cursor: pointer; font-size: 13px; color: var(--text-2); border-bottom: 2px solid transparent; margin-bottom: -2px; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 500; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 20px; margin-bottom: 16px; }
.panel-title { font-size: 14px; font-weight: 600; margin-bottom: 14px; color: var(--text); }
.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
.stat { background: var(--surface-2); padding: 12px; border-radius: 4px; border-left: 2px solid var(--accent); }
.stat .l { font-size: 11px; color: var(--text-3); }
.stat .v { font-size: 20px; font-weight: 600; }
.stat .u { font-size: 11px; color: var(--text-3); margin-left: 3px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
th { color: var(--text-3); font-weight: 500; font-size: 12px; }
tr:hover td { background: var(--surface-2); }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
.b-ready { background: rgba(22,163,74,0.12); color: var(--ok); }
.b-failed { background: rgba(220,38,38,0.12); color: var(--danger); }
.b-pending, .b-processing { background: rgba(217,119,6,0.12); color: var(--warn); }
.b-cat { background: var(--surface-3); color: var(--text-2); }
.result { border: 1px solid var(--border); border-radius: 4px; padding: 10px 14px; margin-bottom: 10px; background: var(--surface-2); }
.result .r-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.result .r-title { font-weight: 500; font-size: 13px; }
.result .r-meta { font-size: 11px; color: var(--text-3); }
.result .r-body { font-size: 13px; white-space: pre-wrap; max-height: 180px; overflow-y: auto; background: var(--surface); padding: 8px; border-radius: 3px; border: 1px solid var(--border); }
.result .r-scores { font-family: ui-monospace, monospace; font-size: 11px; color: var(--text-3); }
.muted { color: var(--text-3); font-size: 12px; }
.toast { position: fixed; bottom: 16px; right: 16px; padding: 10px 16px; border-radius: 4px; z-index: 1000; opacity: 0; transition: opacity 0.2s; font-size: 13px; }
.toast.show { opacity: 1; }
.toast.error { background: var(--danger); color: #fff; }
.toast.success { background: var(--ok); color: #fff; }
.toast.info { background: var(--text); color: var(--surface); }
.loading { text-align: center; padding: 30px; color: var(--text-3); }
.spinner { border: 2px solid var(--border); border-top: 2px solid var(--accent); border-radius: 50%; width: 24px; height: 24px; animation: spin 0.8s linear infinite; margin: 0 auto 8px; }
@keyframes spin { to { transform: rotate(360deg); } }
.modal-bg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 999; justify-content: center; align-items: flex-start; padding: 30px 16px; overflow-y: auto; }
.modal-bg.show { display: flex; }
.modal { background: var(--surface); border-radius: 6px; max-width: 800px; width: 100%; padding: 20px; }
.modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.modal-head h2 { font-size: 14px; font-weight: 600; }
.modal-close { background: none; border: none; font-size: 20px; cursor: pointer; color: var(--text-3); }
.modal-close:hover { color: var(--text); }
/* form */
.form-sec { margin-bottom: 20px; }
.form-sec-title { font-size: 13px; font-weight: 600; color: var(--text-2); margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
.form-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.form-row label { width: 160px; font-size: 13px; color: var(--text-2); flex-shrink: 0; }
.form-row input[type=text], .form-row input[type=number], .form-row select { flex: 1; padding: 5px 8px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; background: var(--surface); color: var(--text); }
.form-row input:focus, .form-row select:focus { outline: none; border-color: var(--accent); }
.form-row input[type=checkbox] { width: 16px; height: 16px; }
.form-row .desc { font-size: 11px; color: var(--text-3); }
.form-row .ro { background: var(--surface-2); color: var(--text-3); }
</style>
</head>
<body>
<header>
  <div>
    <h1>MaiBot 知识库管理</h1>
    <div class="sub">向量 / BM25 混合检索</div>
  </div>
  <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">深色</button>
</header>
<div class="wrap">
  <div class="toolbar">
    <span class="muted">Token:</span>
    <input type="password" id="tokenInput" placeholder="无认证则留空" style="width:180px">
    <button class="btn btn-sm" onclick="saveToken()">确定</button>
    <span style="flex:1"></span>
    <button class="btn btn-sm btn-ghost" onclick="refreshAll()">刷新</button>
  </div>
  <div class="tabs">
    <button class="tab active" onclick="switchTab(this,'stats')">统计</button>
    <button class="tab" onclick="switchTab(this,'files')">文件</button>
    <button class="tab" onclick="switchTab(this,'search')">检索</button>
    <button class="tab" onclick="switchTab(this,'config')">配置</button>
  </div>

  <div id="panel-stats" class="panel">
    <div class="panel-title">知识库概览</div>
    <div id="statsContent" class="stats"><div class="loading"><div class="spinner"></div>加载中</div></div>
    <div style="margin-top:16px;display:flex;gap:8px">
      <button class="btn" onclick="ingest(false)">扫描目录导入新文件</button>
      <button class="btn btn-danger" onclick="ingest(true)">全量重建</button>
    </div>
    <p class="muted" style="margin-top:6px">扫描目录：检查 knowledge_base 文件夹中新增或修改的文件。全量重建：清空所有向量后重新导入（切换 embedding 模型后使用）。</p>
  </div>

  <div id="panel-files" class="panel" style="display:none">
    <div class="panel-title">文件管理</div>
    <div class="toolbar">
      <input type="text" id="fileFilter" placeholder="筛选文件名" style="flex:1;min-width:120px">
      <select id="fileStatusFilter"><option value="">全部状态</option><option value="ready">ready</option><option value="failed">failed</option></select>
      <select id="fileCategoryFilter"><option value="">全部分类</option></select>
      <button class="btn btn-sm" onclick="loadFiles()">筛选</button>
      <span style="flex:1"></span>
      <input type="file" id="uploadInput" accept=".md,.markdown,.txt" multiple onchange="uploadFiles(event)" style="display:none">
      <button class="btn btn-sm" onclick="document.getElementById('uploadInput').click()">上传文件</button>
    </div>
    <div style="overflow-x:auto">
      <table><thead><tr><th>文件名</th><th>分类</th><th>状态</th><th>Chunks</th><th>Tokens</th><th>大小</th><th>导入时间</th><th>操作</th></tr></thead>
      <tbody id="filesTable"><tr><td colspan="8" class="loading"><div class="spinner"></div>加载中</td></tr></tbody></table>
    </div>
  </div>

  <div id="panel-search" class="panel" style="display:none">
    <div class="panel-title">检索测试</div>
    <div class="toolbar">
      <input type="text" id="searchQuery" placeholder="输入查询内容" onkeydown="if(event.key==='Enter')doSearch()" style="flex:1;min-width:180px">
      <select id="searchMode"><option value="hybrid">混合</option><option value="vector">仅向量</option><option value="bm25">仅 BM25</option></select>
      <select id="searchTopK"><option value="3">Top 3</option><option value="5" selected>Top 5</option><option value="10">Top 10</option></select>
      <button class="btn" onclick="doSearch()">检索</button>
    </div>
    <div id="searchResults"></div>
  </div>

  <div id="panel-config" class="panel" style="display:none">
    <div class="panel-title">插件配置</div>
    <div id="configForm"><div class="loading"><div class="spinner"></div>加载中</div></div>
    <div style="margin-top:16px;display:flex;gap:8px">
      <button class="btn" onclick="saveConfig()">保存配置</button>
      <button class="btn btn-ghost" onclick="loadConfig()">重新加载</button>
    </div>
    <p class="muted" style="margin-top:8px">保存后写入 config.toml，插件自动热重载。</p>
  </div>
</div>

<div id="chunkModal" class="modal-bg" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="modal-head"><h2 id="modalTitle">切片详情</h2><button class="modal-close" onclick="closeModal()">&times;</button></div>
    <div id="modalBody"></div>
  </div>
</div>
<div id="toast" class="toast"></div>

<script>
let token = localStorage.getItem('maikb_token') || '';
let theme = localStorage.getItem('maikb_theme') || '';
if (theme === 'dark') { document.documentElement.setAttribute('data-theme','dark'); document.getElementById('themeBtn').textContent = '浅色'; }
document.getElementById('tokenInput').value = token;
refreshAll();

const cfgSchema = [
  {s:'plugin',l:'插件',fields:[
    {k:'enabled',l:'启用插件',t:'bool'},
    {k:'config_version',l:'配置版本',t:'text',ro:true},
  ]},
  {s:'database',l:'数据库',fields:[
    {k:'db_filename',l:'数据库文件名',t:'text'},
    {k:'auto_backup_on_start',l:'启动时自动备份',t:'bool'},
  ]},
  {s:'knowledge_base',l:'知识库',fields:[
    {k:'enabled',l:'启用知识库',t:'bool'},
    {k:'knowledge_dir',l:'源文件目录',t:'text',d:'相对于插件 data 目录'},
    {k:'auto_ingest_on_start',l:'启动时自动导入',t:'bool'},
    {k:'target_chars',l:'目标切片字符数',t:'int'},
    {k:'max_chars',l:'最大切片字符数',t:'int'},
    {k:'min_chars',l:'最小切片字符数',t:'int'},
    {k:'overlap_chars',l:'切片重叠字符数',t:'int',d:'相邻 chunk 共享的字符数'},
    {k:'embedding_provider',l:'Embedding 提供方',t:'select',opts:[['maibot','MaiBot（默认）'],['openai','OpenAI 兼容'],['dummy','Dummy（测试）']]},
    {k:'embedding_model',l:'Embedding 模型',t:'text',d:'maibot 模式填 default'},
    {k:'embedding_dimension',l:'Embedding 维度',t:'int',d:'0 = 自动探测'},
    {k:'embedding_api_key',l:'API Key',t:'text',d:'openai 模式必填'},
    {k:'embedding_base_url',l:'API Base URL',t:'text',d:'openai 模式'},
    {k:'embedding_batch_size',l:'批量大小',t:'int'},
    {k:'default_category',l:'默认分类',t:'text'},
  ]},
  {s:'interceptor',l:'消息拦截',fields:[
    {k:'enabled',l:'启用前缀拦截',t:'bool'},
    {k:'prefixes',l:'拦截前缀',t:'list',d:'逗号分隔'},
    {k:'log_blocked',l:'记录被拦截消息',t:'bool'},
  ]},
  {s:'injector',l:'自动注入',fields:[
    {k:'enabled',l:'启用自动注入',t:'bool'},
    {k:'min_score',l:'RRF 分数阈值',t:'float'},
    {k:'min_vector_score',l:'向量相似度阈值',t:'float'},
    {k:'top_k',l:'注入条数',t:'int'},
    {k:'max_chars',l:'注入最大字符数',t:'int'},
    {k:'dedup_lookback',l:'去重回溯消息数',t:'int'},
    {k:'skip_if_tool_called',l:'LLM 已调 Tool 时跳过',t:'bool'},
  ]},
  {s:'webui',l:'Web UI',fields:[
    {k:'enabled',l:'启用 Web UI',t:'bool'},
    {k:'host',l:'监听地址',t:'text'},
    {k:'port',l:'监听端口',t:'int'},
    {k:'token',l:'访问令牌',t:'text',d:'留空则无认证'},
  ]},
];

function toggleTheme() {
  const r = document.documentElement;
  if (r.getAttribute('data-theme') === 'dark') { r.removeAttribute('data-theme'); localStorage.setItem('maikb_theme','light'); document.getElementById('themeBtn').textContent = '深色'; }
  else { r.setAttribute('data-theme','dark'); localStorage.setItem('maikb_theme','dark'); document.getElementById('themeBtn').textContent = '浅色'; }
}
function saveToken() { token = document.getElementById('tokenInput').value.trim(); localStorage.setItem('maikb_token', token); toast('Token 已保存', 'success'); refreshAll(); }
function hdr() { const h = {'Content-Type':'application/json'}; if (token) h['Authorization'] = 'Bearer ' + token; return h; }
async function api(path, opts = {}) {
  const resp = await fetch(path, {...opts, headers: {...hdr(), ...(opts.headers||{})}});
  if (!resp.ok) { let m = resp.status + ' ' + resp.statusText; try { const j = await resp.json(); m = j.detail || m; } catch(e){} throw new Error(m); }
  return resp.json();
}
function toast(msg, type = 'info') { const el = document.getElementById('toast'); el.textContent = msg; el.className = 'toast show ' + type; setTimeout(() => el.className = 'toast', 2500); }
function esc(s) { return (s==null?'':String(s)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
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
    document.getElementById('statsContent').innerHTML = [
      ['文件总数', s.files_total, ''],
      ['成功导入', s.files_ready, ''],
      ['失败', s.files_failed, ''],
      ['总切片', s.chunks_total, ''],
      ['总 Tokens', s.tokens_total, ''],
      ['总大小', s.size_human, ''],
      ['内存索引', s.vector_index_size, ''],
      ['Embedding', s.embedding_model || '-', s.embedding_dimension + 'd'],
    ].map(r => `<div class="stat"><div class="l">${r[0]}</div><div class="v">${r[1]}${r[2]?'<span class="u">'+r[2]+'</span>':''}</div></div>`).join('');
  } catch(e) { document.getElementById('statsContent').innerHTML = '<div class="stat" style="grid-column:1/-1;color:var(--danger)">加载失败: ' + esc(e.message) + '</div>'; }
}

async function ingest(force) {
  if (force && !confirm('确认全量重建？可能需要较长时间。')) return;
  toast(force ? '全量重建中...' : '增量导入中...');
  try {
    const r = await api('/api/' + (force ? 'rebuild' : 'ingest'), {method:'POST', body:JSON.stringify({force_rebuild:force})});
    toast('完成: new=' + r.new + ' updated=' + r.updated + ' unchanged=' + r.unchanged + ' failed=' + r.failed, 'success');
    loadStats();
  } catch(e) { toast('失败: ' + e.message, 'error'); }
}

async function loadCategories() {
  try { const r = await api('/api/categories'); const sel = document.getElementById('fileCategoryFilter'); const cur = sel.value; sel.innerHTML = '<option value="">全部分类</option>' + r.categories.map(c => '<option value="'+esc(c)+'">'+esc(c)+'</option>').join(''); sel.value = cur; } catch(e) {}
}

async function loadFiles() {
  const st = document.getElementById('fileStatusFilter').value;
  const cat = document.getElementById('fileCategoryFilter').value;
  const filter = document.getElementById('fileFilter').value.toLowerCase();
  const params = new URLSearchParams();
  if (st) params.set('status', st);
  if (cat) params.set('category', cat);
  try {
    const r = await api('/api/files' + (params.toString() ? '?' + params.toString() : ''));
    let items = r.items;
    if (filter) items = items.filter(f => (f.file_name||'').toLowerCase().includes(filter));
    if (!items.length) { document.getElementById('filesTable').innerHTML = '<tr><td colspan="8" style="text-align:center;padding:16px;color:var(--text-3)">无文件</td></tr>'; return; }
    document.getElementById('filesTable').innerHTML = items.map(f => `<tr>
      <td title="${esc(f.file_path)}">${esc(f.file_name)}<div class="muted">${esc(f.title||'')}</div></td>
      <td>${f.category?'<span class="badge b-cat">'+esc(f.category)+'</span>':'<span class="muted">-</span>'}</td>
      <td><span class="badge b-${esc(f.status)}">${esc(f.status)}</span></td>
      <td>${f.chunk_count}</td><td>${f.total_tokens}</td><td>${f.size_human}</td>
      <td class="muted">${f.last_ingested_at?new Date(f.last_ingested_at).toLocaleString():'-'}</td>
      <td><button class="btn btn-sm" onclick="viewChunks('${esc(f.file_id)}')">切片</button> <button class="btn btn-sm btn-danger" onclick="deleteFile('${esc(f.file_id)}','${esc(f.file_name)}')">删除</button></td>
    </tr>`).join('');
  } catch(e) { document.getElementById('filesTable').innerHTML = '<tr><td colspan="8" style="color:var(--danger)">' + esc(e.message) + '</td></tr>'; }
}

async function viewChunks(id) {
  try {
    const r = await api('/api/files/' + id + '/chunks');
    document.getElementById('modalTitle').textContent = '切片详情（' + r.count + ' 个）';
    let html = '<p class="muted" style="margin-bottom:10px">' + esc(r.file_path) + '</p>';
    if (!r.count) html += '<p class="muted">无切片</p>';
    else r.items.forEach(c => { html += `<div class="result"><div class="r-head"><span class="r-title">#${c.chunk_index} ${esc(c.heading||'')}</span><span class="r-meta">${c.char_count} chars / ${c.token_count} tokens ${c.has_embedding?'✓':'⚠'}</span></div><div class="muted" style="margin-bottom:4px">${esc((c.title_path||[]).join(' > '))}</div><div class="r-body">${esc(c.content)}</div></div>`; });
    document.getElementById('modalBody').innerHTML = html;
    document.getElementById('chunkModal').classList.add('show');
  } catch(e) { toast('加载失败: ' + e.message, 'error'); }
}
function closeModal() { document.getElementById('chunkModal').classList.remove('show'); }

async function deleteFile(id, name) {
  if (!confirm('删除文件 "' + name + '" 及其所有切片？')) return;
  try { await api('/api/files/' + id, {method:'DELETE'}); toast('已删除: ' + name, 'success'); loadFiles(); loadStats(); } catch(e) { toast('删除失败: ' + e.message, 'error'); }
}

async function uploadFiles(event) {
  const input = event.target;
  if (!input.files || input.files.length === 0) return;
  const files = Array.from(input.files);
  let ok = 0, fail = 0;
  toast('上传 ' + files.length + ' 个文件中...');
  for (const file of files) {
    const fd = new FormData(); fd.append('file', file);
    try {
      const resp = await fetch('/api/upload', {method:'POST', headers: token ? {'Authorization':'Bearer '+token} : {}, body: fd});
      if (!resp.ok) { let m = resp.status; try { m = (await resp.json()).detail || m; } catch(e){} throw new Error(m); }
      ok++;
    } catch(e) { fail++; toast('失败: ' + file.name + ' - ' + e.message, 'error'); }
  }
  if (ok > 0) toast('完成: 成功 ' + ok + (fail > 0 ? ' 失败 ' + fail : ''), 'success');
  loadFiles(); loadStats(); loadCategories();
  input.value = '';
}

async function doSearch() {
  const q = document.getElementById('searchQuery').value.trim();
  if (!q) { toast('请输入查询', 'error'); return; }
  const mode = document.getElementById('searchMode').value;
  const topK = parseInt(document.getElementById('searchTopK').value);
  document.getElementById('searchResults').innerHTML = '<div class="loading"><div class="spinner"></div>检索中</div>';
  try {
    const r = await api('/api/search', {method:'POST', body:JSON.stringify({query:q, top_k:topK, use_vector:mode!=='bm25', use_bm25:mode!=='vector'})});
    if (!r.count) { document.getElementById('searchResults').innerHTML = '<p style="text-align:center;padding:16px;color:var(--text-3)">未找到相关结果</p>'; return; }
    document.getElementById('searchResults').innerHTML = r.items.map(h => `<div class="result">
      <div class="r-head"><span class="r-title">${esc(h.heading||(h.title_path||[]).join(' > ')||'-')}</span><span class="r-scores">score=${h.score.toFixed(4)} vec=${h.vector_score.toFixed(3)} bm25=${h.bm25_score.toFixed(3)}</span></div>
      <div class="r-meta">来源: ${esc(h.source_name||'-')} | ${esc((h.title_path||[]).join(' > '))}</div>
      <div class="r-body" style="margin-top:6px">${esc(h.content)}</div>
    </div>`).join('');
  } catch(e) { document.getElementById('searchResults').innerHTML = '<p style="color:var(--danger)">' + esc(e.message) + '</p>'; }
}

async function loadConfig() {
  try {
    const c = await api('/api/config');
    let html = '';
    cfgSchema.forEach(sec => {
      html += '<div class="form-sec"><div class="form-sec-title">' + esc(sec.l) + '</div>';
      sec.fields.forEach(f => {
        const val = (c[sec.s] || {})[f.k];
        const id = 'cfg_' + sec.s + '_' + f.k;
        if (f.t === 'bool') {
          html += `<div class="form-row"><label>${esc(f.l)}</label><input type="checkbox" id="${id}" ${val?'checked':''}><span class="desc">${esc(f.d||'')}</span></div>`;
        } else if (f.t === 'select') {
          html += `<div class="form-row"><label>${esc(f.l)}</label><select id="${id}">${f.opts.map(o=>'<option value="'+o[0]+'"'+(val===o[0]?' selected':'')+'>'+o[1]+'</option>').join('')}</select><span class="desc">${esc(f.d||'')}</span></div>`;
        } else if (f.t === 'list') {
          html += `<div class="form-row"><label>${esc(f.l)}</label><input type="text" id="${id}" value="${esc(Array.isArray(val)?val.join(', '):val||'')}"${f.ro?' class="ro" readonly':''}><span class="desc">${esc(f.d||'')}</span></div>`;
        } else {
          html += `<div class="form-row"><label>${esc(f.l)}</label><input type="${f.t==='int'||f.t==='float'?'number':'text'}" id="${id}" value="${esc(val ?? '')}"${f.t==='int'?' step="1"':''}${f.t==='float'?' step="0.01"':''}${f.ro?' class="ro" readonly':''}><span class="desc">${esc(f.d||'')}</span></div>`;
        }
      });
      html += '</div>';
    });
    document.getElementById('configForm').innerHTML = html;
  } catch(e) { document.getElementById('configForm').innerHTML = '<p style="color:var(--danger)">加载失败: ' + esc(e.message) + '</p>'; }
}

async function saveConfig() {
  const config = {};
  cfgSchema.forEach(sec => {
    config[sec.s] = {};
    sec.fields.forEach(f => {
      const el = document.getElementById('cfg_' + sec.s + '_' + f.k);
      if (!el) return;
      if (f.t === 'bool') config[sec.s][f.k] = el.checked;
      else if (f.t === 'int') config[sec.s][f.k] = parseInt(el.value) || 0;
      else if (f.t === 'float') config[sec.s][f.k] = parseFloat(el.value) || 0;
      else if (f.t === 'list') config[sec.s][f.k] = el.value.split(',').map(s=>s.trim()).filter(Boolean);
      else config[sec.s][f.k] = el.value;
    });
  });
  try {
    await api('/api/config', {method:'PUT', body:JSON.stringify({config})});
    toast('配置已保存，插件将热重载', 'success');
  } catch(e) { toast('保存失败: ' + e.message, 'error'); }
}
</script>
</body>
</html>
"""


__all__ = ["WebServer"]

"""tests.conftest — pytest 共享 fixture"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

# 让 tests 能找到 maikb 包
sys.path.insert(0, str(Path(__file__).parent.parent))

# 优先使用真实 maibot_sdk；若环境未安装，回退到随仓库分发的测试桩
# （tests/_sdk_stub/maibot_sdk）。真实 SDK 行为以 MaiBot 运行时为准。
try:
    import maibot_sdk  # noqa: F401
except ImportError:
    _STUB_DIR = Path(__file__).parent / "_sdk_stub"
    if _STUB_DIR.is_dir():
        sys.path.insert(0, str(_STUB_DIR))


@pytest_asyncio.fixture
async def db_instance():
    """提供一个初始化好的临时数据库实例（含迁移）。"""

    tmpdir = tempfile.mkdtemp(prefix="maikb_test_")
    db_path = os.path.join(tmpdir, "test.db")

    from maikb import MaiKBDatabase, run_migrations
    db = MaiKBDatabase(db_path)
    await db.initialize()
    # 跑迁移，让 preferences 表中有 initial_schema_v1 标记
    await run_migrations(db)
    yield db
    await db.close()

"""tests.test_migrations — 迁移框架测试"""

import pytest

from maikb import (
    MaiKBDatabase,
    register_migration,
    run_migrations,
    list_registered_migrations,
    SharedPreferences,
)


@pytest.mark.asyncio
async def test_initial_migration_runs_once(db_instance):
    """已注册的 initial_schema_v1 迁移应被标记为完成。"""

    sp = SharedPreferences(db_instance)
    # initialize() 已经跑过一次迁移
    assert await sp.is_migration_done("initial_schema_v1") is True


@pytest.mark.asyncio
async def test_run_migrations_is_idempotent(db_instance):
    """重复跑迁移不应有副作用。"""

    # 再跑一次
    results = await run_migrations(db_instance)
    assert all(results.values())
    # initial_schema_v1 应该是 True（已完成）
    assert results.get("initial_schema_v1") is True


@pytest.mark.asyncio
async def test_custom_migration(tmp_path):
    """注册一个自定义迁移并验证它会被执行。"""

    # 标记位文件，用 closure 捕获
    state = {"executed": 0}

    @register_migration("test_custom_migration_xyz")
    async def _migrate(db: MaiKBDatabase) -> None:
        state["executed"] += 1

    assert "test_custom_migration_xyz" in list_registered_migrations()

    db_path = tmp_path / "test_migrations.db"
    db = MaiKBDatabase(db_path)
    await db.initialize()

    # 第一次跑：应该执行
    results = await run_migrations(db)
    assert results["test_custom_migration_xyz"] is True
    assert state["executed"] == 1

    # 第二次跑：应该跳过
    results = await run_migrations(db)
    assert results["test_custom_migration_xyz"] is True
    assert state["executed"] == 1  # 没有再次执行

    await db.close()


@pytest.mark.asyncio
async def test_ensure_columns_idempotent(tmp_path):
    """_ensure_xxx_column 应该可重复执行。"""

    db_path = tmp_path / "test_ensure.db"
    db = MaiKBDatabase(db_path)
    await db.initialize()

    # 再次初始化不应抛错
    await db.initialize()

    # 验证列存在
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        result = await conn.execute(text("PRAGMA table_info(personas)"))
        cols = {row[1] for row in result.fetchall()}
        assert "skills" in cols
        assert "custom_error_message" in cols

    await db.close()

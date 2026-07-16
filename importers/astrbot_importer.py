"""importers.astrbot_importer

从 AstrBot 的 data_v4.db 导入数据到本插件的 maikb.db。

用法：

    python -m importers.astrbot_importer \\
        --src /path/to/AstrBot/data/data_v4.db \\
        --dst /path/to/maibot/data/plugins/maibot-team.astrbot-db-port/maikb.db

也支持以模块方式被插件代码调用：

    from importers.astrbot_importer import import_from_astrbot
    await import_from_astrbot(src_path, dst_path)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# 允许 `python -m importers.astrbot_importer` 直接执行
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from maikb import MaiKBDatabase, init_db, close_db, get_db
from maikb.models import (
    Attachment,
    CommandConfig,
    CommandConflict,
    ConversationV2,
    CronJob,
    Preference,
    Persona,
    PersonaFolder,
    PlatformMessageHistory,
    PlatformSession,
    PlatformStat,
    ProviderStat,
    UmoAlias,
    WebChatThread,
)


logger = logging.getLogger("maikb.importer")


# 表名 → SQLModel 类映射
TABLE_TO_MODEL = {
    "platform_stats": PlatformStat,
    "provider_stats": ProviderStat,
    "conversations": ConversationV2,
    "persona_folders": PersonaFolder,
    "personas": Persona,
    "cron_jobs": CronJob,
    "preferences": Preference,
    "platform_message_history": PlatformMessageHistory,
    "webchat_threads": WebChatThread,
    "platform_sessions": PlatformSession,
    "umo_aliases": UmoAlias,
    "attachments": Attachment,
    "command_configs": CommandConfig,
    "command_conflicts": CommandConflict,
}


async def import_from_astrbot(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    batch_size: int = 500,
    skip_tables: set[str] | None = None,
) -> dict[str, int]:
    """从 AstrBot v4 数据库导入到本插件数据库。

    Args:
        src_path: AstrBot data_v4.db 路径
        dst_path: 本插件 maikb.db 路径
        batch_size: 每批读取的行数
        skip_tables: 跳过的表名集合

    Returns:
        Dict[str, int]: 每张表导入的行数
    """

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    skip_tables = skip_tables or set()

    if not src_path.exists():
        raise FileNotFoundError(f"AstrBot 数据库不存在: {src_path}")

    logger.info(f"开始从 AstrBot 数据库导入: {src_path} → {dst_path}")

    # 初始化目标库
    await init_db(dst_path)
    db = get_db()

    # 用 aiosqlite 直接连源库读
    import aiosqlite
    counts: dict[str, int] = {}

    async with aiosqlite.connect(str(src_path)) as src:
        src.row_factory = aiosqlite.Row

        for table_name, model_cls in TABLE_TO_MODEL.items():
            if table_name in skip_tables:
                logger.info(f"跳过表: {table_name}")
                continue

            # 检查源库是否有这张表
            async with src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    logger.info(f"源库无表 {table_name}，跳过")
                    continue

            # 读源数据
            async with src.execute(f"SELECT * FROM {table_name}") as cur:
                rows = await cur.fetchall()

            if not rows:
                logger.info(f"表 {table_name} 为空")
                counts[table_name] = 0
                continue

            # 字段对齐：源库可能缺我们新增的列，按 model_cls 字段子集导入
            model_fields = set(model_cls.model_fields.keys())
            src_columns = set(rows[0].keys())
            common = model_fields & src_columns

            inserted = 0
            async with db.get_db() as session:
                async with session.begin():
                    for row in rows:
                        data = {k: row[k] for k in common if row[k] is not None}
                        if not data:
                            continue
                        # 跳过自增主键，避免冲突
                        for pk in ("id", "inner_id", "inner_conversation_id", "inner_attachment_id"):
                            data.pop(pk, None)
                        try:
                            session.add(model_cls(**data))
                            inserted += 1
                            if inserted % batch_size == 0:
                                await session.flush()
                        except Exception as exc:
                            logger.warning(
                                f"表 {table_name} 第 {inserted} 行导入失败: {exc}"
                            )

            counts[table_name] = inserted
            logger.info(f"表 {table_name}: 已导入 {inserted} 行")

    await close_db()
    logger.info(f"导入完成: {counts}")
    return counts


def main() -> None:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(
        description="从 AstrBot data_v4.db 导入数据到 MaiBot 插件 maikb.db"
    )
    parser.add_argument(
        "--src", required=True, help="AstrBot data_v4.db 路径"
    )
    parser.add_argument(
        "--dst", required=True, help="目标 maikb.db 路径"
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="跳过的表名（空格分隔）",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="日志级别"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    counts = asyncio.run(
        import_from_astrbot(
            args.src, args.dst, skip_tables=set(args.skip)
        )
    )
    print("\n=== 导入结果 ===")
    for table, n in counts.items():
        print(f"  {table:30s} {n:>8} 行")
    print(f"\n总计: {sum(counts.values())} 行")


if __name__ == "__main__":
    main()

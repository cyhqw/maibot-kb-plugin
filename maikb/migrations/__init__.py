"""maikb.migrations

幂等迁移框架（移植自 AstrBot 自研迁移机制）。
"""

from .manager import (
    list_registered_migrations,
    register_migration,
    run_migrations,
)

__all__ = [
    "register_migration",
    "run_migrations",
    "list_registered_migrations",
]

"""astrdb.memory.decay_scheduler

衰减调度器 — 每日定时执行重要性衰减 + 清理。

移植自 LivingMemory `core/schedulers/decay_scheduler.py`。

每日 00:05 执行：
1. apply_daily_decay: importance × (1-decay_rate)^days
2. cleanup_low_importance: 删除 >30天 且 importance<0.3
3. run_maintenance: 状态机推进
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from .atom_store import AtomStore
from .lifecycle import AtomLifecycleManager


logger = logging.getLogger("astrdb.memory.decay_scheduler")


class DecayScheduler:
    """每日衰减调度器。

    用法：
        scheduler = DecayScheduler(atom_store, lifecycle_manager)
        await scheduler.start()  # 在 plugin.on_load 中
        ...
        await scheduler.stop()   # 在 plugin.on_unload 中
    """

    def __init__(
        self,
        atom_store: AtomStore,
        lifecycle_manager: AtomLifecycleManager,
        *,
        decay_rate: float = 0.01,
        check_hour: int = 0,
        check_minute: int = 5,
        cleanup_days_threshold: int = 30,
        cleanup_importance_threshold: float = 0.3,
    ) -> None:
        self._store = atom_store
        self._lifecycle = lifecycle_manager
        self._decay_rate = decay_rate
        self._check_hour = check_hour
        self._check_minute = check_minute
        self._cleanup_days = cleanup_days_threshold
        self._cleanup_importance = cleanup_importance_threshold

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_decay_date: Optional[str] = None

    def start(self) -> None:
        """启动调度器。"""

        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"DecayScheduler 已启动，每日 {self._check_hour:02d}:{self._check_minute:02d} 执行"
        )

    async def stop(self) -> None:
        """停止调度器。"""

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("DecayScheduler 已停止")

    async def _loop(self) -> None:
        """主循环：等到下一个执行时间，执行后等下一个。"""

        while self._running:
            try:
                wait_seconds = self._seconds_until_next_run()
                logger.debug(f"下次衰减执行在 {wait_seconds/3600:.1f} 小时后")
                await asyncio.sleep(min(wait_seconds, 3600))  # 最多等 1 小时检查一次
                if not self._running:
                    break
                if self._is_time_to_run():
                    await self._execute_decay()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"DecayScheduler 循环异常: {exc}", exc_info=True)
                await asyncio.sleep(60)  # 出错后等 1 分钟

    def _seconds_until_next_run(self) -> float:
        """计算到下次执行时间的秒数。"""

        now = datetime.now(timezone.utc)
        target = now.replace(
            hour=self._check_hour, minute=self._check_minute, second=0, microsecond=0
        )
        if target <= now:
            target = target.replace(day=target.day + 1)
        return (target - now).total_seconds()

    def _is_time_to_run(self) -> bool:
        """检查是否到执行时间。"""

        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        if self._last_decay_date == today_str:
            return False
        return now.hour == self._check_hour and now.minute >= self._check_minute

    async def _execute_decay(self) -> None:
        """执行一次衰减。"""

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info(f"开始执行每日衰减 ({today_str})")

        try:
            # 1. 重要性衰减
            decayed = await self._store.apply_daily_decay(
                decay_rate=self._decay_rate, days=1
            )
            logger.info(f"重要性衰减: {decayed} 个 atoms")

            # 2. 清理低重要性
            cleaned = await self._store.cleanup_low_importance(
                days_threshold=self._cleanup_days,
                importance_threshold=self._cleanup_importance,
            )
            logger.info(f"清理低重要性: {cleaned} 个 atoms")

            # 3. 生命周期维护
            lifecycle_result = await self._lifecycle.run_maintenance()
            logger.info(f"生命周期维护: {lifecycle_result}")

            self._last_decay_date = today_str
        except Exception as exc:
            logger.error(f"衰减执行失败: {exc}", exc_info=True)

    async def run_now(self) -> dict[str, int]:
        """手动触发一次衰减（用于测试或管理命令）。"""

        decayed = await self._store.apply_daily_decay(
            decay_rate=self._decay_rate, days=1
        )
        cleaned = await self._store.cleanup_low_importance(
            days_threshold=self._cleanup_days,
            importance_threshold=self._cleanup_importance,
        )
        lifecycle_result = await self._lifecycle.run_maintenance()
        return {
            "decayed": decayed,
            "cleaned": cleaned,
            **lifecycle_result,
        }


__all__ = ["DecayScheduler"]

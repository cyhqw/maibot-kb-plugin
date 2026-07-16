"""astrdb.memory.models

MemoryAtom 数据模型 — 移植自 LivingMemory 的核心创新。

每个 MemoryAtom 是从对话总结中抽取的细粒度事实单元，拥有：
- 独立 TTL（按类型+重要性+强化次数动态计算）
- 衰减曲线（LINEAR / EXPONENTIAL / STEP）
- 生命周期状态机（ACTIVE → EXPIRED → FORGOTTEN → 物理删除）
- 强化机制（被相似内容命中时 confidence EMA 更新 + TTL 续期）

5 种 AtomType：
- EPISODIC    事件性记忆（7天 TTL，指数衰减）
- FACTUAL     事实性记忆（180天 TTL，指数衰减）
- RELATIONAL  关系性记忆（90天 TTL，线性衰减）
- PREFERENCE  偏好性记忆（60天 TTL，指数衰减）
- PLANNED     计划性记忆（2天 TTL，阶梯衰减，到期骤降至 0.05）

设计来源：LivingMemory `core/models/memory_atom.py`
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Text
from sqlmodel import Field, SQLModel

from ..models import TimestampMixin, _uuid_str


# ----------------------------------------------------------------------
# 枚举
# ----------------------------------------------------------------------

class AtomType(str, enum.Enum):
    """记忆原子类型。"""

    EPISODIC = "episodic"        # 事件性：具体发生的事
    FACTUAL = "factual"          # 事实性：客观事实
    RELATIONAL = "relational"    # 关系性：人与人/物之间的关系
    PREFERENCE = "preference"    # 偏好性：用户喜好
    PLANNED = "planned"          # 计划性：未来要做的事
    UNKNOWN = "unknown"          # 未分类


class AtomStatus(str, enum.Enum):
    """记忆原子生命周期状态。"""

    ACTIVE = "active"            # 活跃，可被检索
    DORMANT = "dormant"          # 休眠（重要性低于阈值但未过期）
    SUPERSEDED = "superseded"    # 被新记忆取代
    EXPIRED = "expired"          # 已过期（TTL 到期）
    FORGOTTEN = "forgotten"      # 已遗忘（EXPIRED + forget_delay 后）


class DecayType(str, enum.Enum):
    """衰减曲线类型。"""

    LINEAR = "linear"            # 线性衰减
    EXPONENTIAL = "exponential"  # 指数衰减
    STEP = "step"                # 阶梯衰减（到期骤降）


# ----------------------------------------------------------------------
# TTL 配置表（按 AtomType）
# ----------------------------------------------------------------------

ATOM_TYPE_CONFIG: dict[AtomType, dict[str, Any]] = {
    AtomType.EPISODIC:    {"base_ttl_days": 7.0,   "decay_type": DecayType.EXPONENTIAL},
    AtomType.FACTUAL:     {"base_ttl_days": 180.0, "decay_type": DecayType.EXPONENTIAL},
    AtomType.RELATIONAL:  {"base_ttl_days": 90.0,  "decay_type": DecayType.LINEAR},
    AtomType.PREFERENCE:  {"base_ttl_days": 60.0,  "decay_type": DecayType.EXPONENTIAL},
    AtomType.PLANNED:     {"base_ttl_days": 2.0,   "decay_type": DecayType.STEP},
    AtomType.UNKNOWN:     {"base_ttl_days": 30.0,  "decay_type": DecayType.EXPONENTIAL},
}


def compute_ttl(
    atom_type: AtomType,
    importance: float = 0.5,
    reinforcement_count: int = 0,
) -> tuple[float, DecayType]:
    """动态计算 TTL。

    公式（移植自 LivingMemory）：
        ttl = base_ttl × (0.5 + importance) × (1.0 + min(0.5, reinforcement_count × 0.1))

    Returns:
        (ttl_days, decay_type)
    """

    cfg = ATOM_TYPE_CONFIG.get(atom_type, ATOM_TYPE_CONFIG[AtomType.UNKNOWN])
    base_ttl = cfg["base_ttl_days"]
    decay_type = cfg["decay_type"]

    # importance 在 [0, 1]，乘以 (0.5 + importance) → [0.5, 1.5]
    importance_factor = 0.5 + max(0.0, min(1.0, importance))

    # 强化次数最多贡献 +50%
    reinforcement_factor = 1.0 + min(0.5, reinforcement_count * 0.1)

    ttl = base_ttl * importance_factor * reinforcement_factor
    return ttl, decay_type


def compute_decay_factor(
    decay_type: DecayType,
    days_since_created: float,
    ttl_days: float,
) -> float:
    """计算衰减因子（0-1，1 = 全新，0 = 完全衰减）。

    - LINEAR: 线性从 1 衰减到 0（TTL 到期时为 0）
    - EXPONENTIAL: 指数衰减，半衰期 = TTL/2
    - STEP: TTL 内保持 1，到期骤降至 0.05
    """

    if ttl_days <= 0:
        return 0.0

    ratio = days_since_created / ttl_days

    if decay_type == DecayType.LINEAR:
        return max(0.0, 1.0 - ratio)
    elif decay_type == DecayType.EXPONENTIAL:
        # 半衰期 = TTL/2，即 ratio=1 时衰减到 0.25
        return math.exp(-1.386 * ratio)  # ln(4) ≈ 1.386
    elif decay_type == DecayType.STEP:
        return 1.0 if ratio < 1.0 else 0.05
    return max(0.0, 1.0 - ratio)


# ----------------------------------------------------------------------
# 数据模型
# ----------------------------------------------------------------------

class MemoryAtom(TimestampMixin, SQLModel, table=True):
    """记忆原子表。

    每个 atom 是从对话总结中抽取的细粒度事实，独立检索/衰减/强化。
    parent_memory_id 关联到 conversations 表的 conversation_id（本插件设计）。
    """

    __tablename__ = "memory_atoms"

    id: Optional[int] = Field(default=None, primary_key=True)
    atom_id: str = Field(max_length=36, unique=True, default_factory=_uuid_str, index=True)

    # 关联（parent_memory_id 指向 conversations.conversation_id 字符串）
    parent_memory_id: str = Field(max_length=36, index=True, description="父记忆 ID（conversation_id）")

    # 内容
    atom_type: str = Field(max_length=32, index=True, description="AtomType 枚举值")
    content: str = Field(sa_type=Text, description="单条事实内容")
    entities: list[str] = Field(default_factory=list, sa_type=JSON, description="实体列表")

    # 重要性 & 置信度
    importance: float = Field(default=0.5, description="重要性 [0,1]")
    confidence: float = Field(default=0.7, description="置信度 [0,1]")

    # 时间（unix 秒）
    created_at_ts: float = Field(default=0.0, index=True, description="创建时间戳")
    last_accessed_at_ts: float = Field(default=0.0, description="最后访问时间戳")
    last_reinforced_at_ts: float = Field(default=0.0, description="最后强化时间戳")
    event_time_ts: Optional[float] = Field(default=None, description="事件时间（PLANNED 类型用）")

    # TTL & 衰减
    ttl_days: float = Field(default=30.0, description="TTL 天数")
    expires_at_ts: float = Field(default=0.0, index=True, description="过期时间戳")
    decay_type: str = Field(default="exponential", max_length=16)

    # 生命周期
    status: str = Field(default="active", index=True, description="AtomStatus 枚举值")
    reinforcement_count: int = Field(default=0, description="强化次数")

    # 上下文
    session_id: Optional[str] = Field(default=None, index=True, description="会话 ID（UMO）")
    persona_id: Optional[str] = Field(default=None, index=True, description="人格 ID")

    # 元数据
    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)

    # 来源
    source: str = Field(default="auto", max_length=32, description="auto / agent_tool / manual")


# ----------------------------------------------------------------------
# 复合索引（在 SQLModel 中通过 __table_args__ 定义）
# ----------------------------------------------------------------------

from sqlalchemy import Index, UniqueConstraint

MemoryAtom.__table_args__ = (
    # 召回主查询路径：按 session+persona+status 过滤
    Index("idx_atoms_scope_status", "session_id", "persona_id", "status"),
    # 过期清理路径
    Index("idx_atoms_status_expires", "status", "expires_at_ts"),
    # 按父记忆反查
    Index("idx_atoms_parent", "parent_memory_id"),
)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def atom_type_from_str(s: str) -> AtomType:
    """字符串转 AtomType，未知值返回 UNKNOWN。"""

    try:
        return AtomType(s)
    except ValueError:
        return AtomType.UNKNOWN


def atom_status_from_str(s: str) -> AtomStatus:
    """字符串转 AtomStatus，未知值返回 ACTIVE。"""

    try:
        return AtomStatus(s)
    except ValueError:
        return AtomStatus.ACTIVE


def decay_type_from_str(s: str) -> DecayType:
    """字符串转 DecayType，未知值返回 EXPONENTIAL。"""

    try:
        return DecayType(s)
    except ValueError:
        return DecayType.EXPONENTIAL


def now_ts() -> float:
    """当前 UTC 时间戳（秒）。"""

    return datetime.now(timezone.utc).timestamp()


__all__ = [
    "AtomType",
    "AtomStatus",
    "DecayType",
    "ATOM_TYPE_CONFIG",
    "MemoryAtom",
    "compute_ttl",
    "compute_decay_factor",
    "atom_type_from_str",
    "atom_status_from_str",
    "decay_type_from_str",
    "now_ts",
]

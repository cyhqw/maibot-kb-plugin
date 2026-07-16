"""astrdb.memory.atom_classifier

Atom 分类器 — 纯规则，零 LLM 调用。

移植自 LivingMemory `core/processors/atom_classifier.py`。

分类优先级：PLANNED > PREFERENCE > RELATIONAL > FACTUAL > EPISODIC > UNKNOWN

5 种 AtomType：
- PLANNED     有时间指示 + 动作动词（"明天开会"）
- PREFERENCE  偏好词（"喜欢"、"讨厌"）
- RELATIONAL  关系词（"同事"、"朋友"）
- FACTUAL     状态词（"是"、"有"、"属于"）
- EPISODIC    动作但无时间（"去吃饭了"）
- UNKNOWN     兜底
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import AtomType


# 时间指示词
_TIME_INDICATORS = [
    "明天", "后天", "大后天", "今天", "今晚", "今早",
    "下周", "下月", "下年", "明年",
    "周一周二周三周四周五周六周日",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日",
    "早晨", "上午", "中午", "下午", "晚上", "夜里",
    r"\d+月\d+日", r"\d+月\d+号", r"\d+号", r"\d+日",
    r"\d+点", r"\d+:\d+",
]

# 动作动词
_ACTION_VERBS = [
    "开会", "讨论", "参加", "去", "做", "要", "需要", "准备",
    "完成", "提交", "汇报", "见面", "约会", "聚餐", "出差",
    "旅游", "出差", "上课", "上班", "下班", "回家",
]

# 偏好词
_PREFERENCE_PATTERNS = [
    r"喜欢", r"讨厌", r"偏好", r"爱吃", r"不爱吃", r"想", r"想要",
    r"希望", r"讨厌", r"反感", r"钟爱", r"偏爱", r"厌恶",
    r"最喜欢", r"最讨厌",
]

# 关系词
_RELATION_PATTERNS = [
    r"同事", r"朋友", r"家人", r"搭档", r"同学", r"室友",
    r"邻居", r"老师", r"学生", r"老板", r"下属", r"上司",
    r"男朋友", r"女朋友", r"老公", r"老婆", r"爸爸", r"妈妈",
    r"哥哥", r"姐姐", r"弟弟", r"妹妹",
]

# 状态动词（事实性）
_STATIVE_PATTERNS = [
    r"是", r"有", r"属于", r"等于", r"位于", r"包含", r"拥有",
    r"成立", r"出生", r"毕业于", r"工作于",
]


def _match_any(patterns: list[str], text: str) -> bool:
    """检查文本是否匹配任一模式。"""

    for p in patterns:
        if re.search(p, text):
            return True
    return False


def parse_event_time(text: str) -> Optional[float]:
    """解析中文相对时间为 unix 时间戳。

    支持：明天/后天/今天/下周X/周X/某月某日 等。
    失败返回 None。
    """

    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 明天
    if re.search(r"明天|明早|明晚", text):
        target = today + timedelta(days=1)
    elif re.search(r"后天", text):
        target = today + timedelta(days=2)
    elif re.search(r"大后天", text):
        target = today + timedelta(days=3)
    elif re.search(r"今天|今晚|今早", text):
        target = today
    elif re.search(r"下周", text):
        # 下周X → 7-13 天后
        m = re.search(r"下周([一二三四五六日天])", text)
        if m:
            day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
            target_day = day_map.get(m.group(1), 0)
            days_ahead = 7 + (target_day - today.weekday()) % 7
            target = today + timedelta(days=days_ahead)
        else:
            target = today + timedelta(days=7)
    elif re.search(r"周([一二三四五六日天])", text):
        m = re.search(r"周([一二三四五六日天])", text)
        if m:
            day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
            target_day = day_map.get(m.group(1), 0)
            days_ahead = (target_day - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target = today + timedelta(days=days_ahead)
        else:
            return None
    else:
        # 尝试 \d+月\d+日
        m = re.search(r"(\d+)月(\d+)[日号]", text)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            try:
                target = today.replace(month=month, day=day)
                if target < today:
                    target = target.replace(year=today.year + 1)
            except ValueError:
                return None
        else:
            return None

    return target.timestamp()


def classify_atom(content: str) -> tuple[AtomType, float, Optional[float]]:
    """分类单个 atom。

    Args:
        content: atom 文本内容

    Returns:
        (atom_type, confidence, event_time_ts)
        - confidence: 0.6-0.85，规则匹配越精确置信度越高
        - event_time_ts: PLANNED 类型才有，其他为 None
    """

    if not content or not content.strip():
        return AtomType.UNKNOWN, 0.5, None

    text = content.strip()

    # 1. PLANNED：时间指示 + 动作动词
    has_time = _match_any(_TIME_INDICATORS, text)
    has_action = _match_any(_ACTION_VERBS, text)
    if has_time and has_action:
        event_time = parse_event_time(text)
        return AtomType.PLANNED, 0.85, event_time
    if has_time:
        # 只有时间没动作，也算 PLANNED 但置信度低
        event_time = parse_event_time(text)
        return AtomType.PLANNED, 0.7, event_time

    # 2. PREFERENCE
    if _match_any(_PREFERENCE_PATTERNS, text):
        return AtomType.PREFERENCE, 0.8, None

    # 3. RELATIONAL
    if _match_any(_RELATION_PATTERNS, text):
        return AtomType.RELATIONAL, 0.78, None

    # 4. FACTUAL
    if _match_any(_STATIVE_PATTERNS, text):
        return AtomType.FACTUAL, 0.75, None

    # 5. EPISODIC：有动作动词但没时间
    if has_action:
        return AtomType.EPISODIC, 0.7, None

    # 6. UNKNOWN
    return AtomType.UNKNOWN, 0.6, None


def classify_atoms(
    contents: list[str],
    parent_importance: float = 0.5,
) -> list[tuple[AtomType, float, Optional[float], float]]:
    """批量分类。

    Args:
        contents: atom 文本列表
        parent_importance: 父记忆的重要性（atom 继承）

    Returns:
        List of (atom_type, confidence, event_time_ts, importance)
    """

    results = []
    for content in contents:
        atom_type, confidence, event_time = classify_atom(content)
        # importance 继承父记忆，但 PLANNED 类型提升（重要事件）
        importance = parent_importance
        if atom_type == AtomType.PLANNED:
            importance = min(1.0, parent_importance + 0.2)
        elif atom_type == AtomType.PREFERENCE:
            importance = min(1.0, parent_importance + 0.1)
        results.append((atom_type, confidence, event_time, importance))
    return results


__all__ = [
    "classify_atom",
    "classify_atoms",
    "parse_event_time",
]

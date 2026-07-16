"""tests.test_models — 模型基础测试"""

import pytest

from maikb.models import (
    ConversationV2,
    Persona,
    build_umo,
    parse_umo,
)


def test_build_umo():
    umo = build_umo("aiocqhttp", "GroupMessage", "123456")
    assert umo == "aiocqhttp:GroupMessage:123456"


def test_parse_umo():
    p, t, s = parse_umo("webchat:FriendMessage:user-abc")
    assert p == "webchat"
    assert t == "FriendMessage"
    assert s == "user-abc"


def test_parse_umo_invalid():
    with pytest.raises(ValueError):
        parse_umo("invalid")
    with pytest.raises(ValueError):
        parse_umo("a:b")  # 只有两段


def test_umo_roundtrip():
    """构造 → 解析 → 应能还原。"""

    cases = [
        ("aiocqhttp", "GroupMessage", "123456789"),
        ("webchat", "FriendMessage", "webchat!astrbot!user123"),
        ("discord", "GroupMessage", "987654321"),
    ]
    for platform, msg_type, session_id in cases:
        umo = build_umo(platform, msg_type, session_id)
        p, t, s = parse_umo(umo)
        assert (p, t, s) == (platform, msg_type, session_id)


def test_conversation_model_uuid():
    """ConversationV2 实例化时应自动生成 conversation_id。"""

    conv = ConversationV2(
        user_id="aiocqhttp:GroupMessage:1",
        platform_id="aiocqhttp",
    )
    assert conv.conversation_id  # 不为空
    assert len(conv.conversation_id) == 36  # UUID 长度


def test_persona_default_values():
    """Persona 字段默认值。"""

    p = Persona(name="test")
    assert p.system_prompt == ""
    assert p.begin_dialogs == []
    assert p.tools == []
    assert p.skills == []
    assert p.is_default is False
    assert p.sort_order == 0

"""tests.test_conversations — 对话 CRUD 测试"""

import pytest

from maikb.models import build_umo


@pytest.mark.asyncio
async def test_create_and_get(db_instance):
    umo = build_umo("aiocqhttp", "GroupMessage", "123456")
    conv = await db_instance.create_conversation(
        user_id=umo,
        platform_id="aiocqhttp",
        title="测试对话",
        content=[{"role": "system", "content": "你好"}],
    )

    assert conv.conversation_id  # UUID 已生成
    assert conv.title == "测试对话"
    assert conv.content == [{"role": "system", "content": "你好"}]
    assert conv.token_usage == 0

    # 重新读出来
    loaded = await db_instance.get_conversation_by_id(conv.conversation_id)
    assert loaded is not None
    assert loaded.conversation_id == conv.conversation_id
    assert loaded.title == "测试对话"
    assert loaded.content[0]["content"] == "你好"


@pytest.mark.asyncio
async def test_list_by_user(db_instance):
    umo = build_umo("webchat", "FriendMessage", "user-abc")

    # 创建 3 个对话
    for i in range(3):
        await db_instance.create_conversation(
            user_id=umo,
            platform_id="webchat",
            title=f"conv-{i}",
        )

    convs = await db_instance.get_conversations_by_user(umo)
    assert len(convs) == 3
    # 应按 updated_at 倒序
    titles = [c.title for c in convs]
    assert "conv-0" in titles
    assert "conv-2" in titles


@pytest.mark.asyncio
async def test_update_content(db_instance):
    umo = build_umo("aiocqhttp", "GroupMessage", "111")
    conv = await db_instance.create_conversation(
        user_id=umo, platform_id="aiocqhttp"
    )

    new_content = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    ok = await db_instance.update_conversation_content(
        conv.conversation_id, new_content, token_usage_delta=50
    )
    assert ok is True

    loaded = await db_instance.get_conversation_by_id(conv.conversation_id)
    assert loaded.content == new_content
    assert loaded.token_usage == 50


@pytest.mark.asyncio
async def test_delete(db_instance):
    umo = build_umo("aiocqhttp", "GroupMessage", "222")
    conv = await db_instance.create_conversation(user_id=umo, platform_id="aiocqhttp")

    ok = await db_instance.delete_conversation(conv.conversation_id)
    assert ok is True

    # 再删一次
    ok = await db_instance.delete_conversation(conv.conversation_id)
    assert ok is False

    loaded = await db_instance.get_conversation_by_id(conv.conversation_id)
    assert loaded is None


@pytest.mark.asyncio
async def test_isolation_between_users(db_instance):
    """不同 UMO 的对话应互相隔离。"""

    umo1 = build_umo("aiocqhttp", "GroupMessage", "111")
    umo2 = build_umo("aiocqhttp", "GroupMessage", "222")

    await db_instance.create_conversation(user_id=umo1, platform_id="aiocqhttp", title="u1-1")
    await db_instance.create_conversation(user_id=umo1, platform_id="aiocqhttp", title="u1-2")
    await db_instance.create_conversation(user_id=umo2, platform_id="aiocqhttp", title="u2-1")

    assert len(await db_instance.get_conversations_by_user(umo1)) == 2
    assert len(await db_instance.get_conversations_by_user(umo2)) == 1

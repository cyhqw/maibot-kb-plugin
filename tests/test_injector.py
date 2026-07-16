"""tests.test_injector — 自动注入器测试"""

import pytest

from maikb.injector import (
    _extract_last_user_text,
    _format_chunks,
    _has_recent_tool_call,
)


def test_extract_last_user_text_dict():
    """从 dict 形式 messages 提取最后一条 user 文本。"""

    messages = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "法涅斯是谁？"},
    ]
    assert _extract_last_user_text(messages) == "法涅斯是谁？"


def test_extract_last_user_text_with_whitespace():
    """文本会被 strip。"""

    messages = [{"role": "user", "content": "  hello world  "}]
    assert _extract_last_user_text(messages) == "hello world"


def test_extract_last_user_text_multimodal():
    """OpenAI 多模态 content（list of {type, text}）。"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图是什么"},
                {"type": "image_url", "image_url": {"url": "..."}},
            ],
        }
    ]
    assert _extract_last_user_text(messages) == "这张图是什么"


def test_extract_last_user_text_no_user():
    """没有 user 消息时返回空。"""

    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "hi"},
    ]
    assert _extract_last_user_text(messages) == ""


def test_extract_last_user_text_empty():
    assert _extract_last_user_text([]) == ""
    assert _extract_last_user_text(None) == ""  # type: ignore


def test_has_recent_tool_call_dict():
    """检测 dict 形式的 tool_calls。"""

    messages = [
        {"role": "user", "content": "查一下法涅斯"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "knowledge_search",
                        "arguments": '{"query": "法涅斯"}',
                    },
                }
            ],
        },
    ]
    assert _has_recent_tool_call(messages, "knowledge_search") is True


def test_has_recent_tool_call_other_tool():
    """其他 tool 名不算。"""

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search_memory", "arguments": "{}"}},
            ],
        }
    ]
    assert _has_recent_tool_call(messages, "knowledge_search") is False


def test_has_recent_tool_call_empty():
    assert _has_recent_tool_call([], "knowledge_search") is False
    assert _has_recent_tool_call(None, "knowledge_search") is False  # type: ignore


def test_has_recent_tool_call_lookback_limit():
    """lookback 限制：超出范围的不算。"""

    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "knowledge_search"}}],
        },
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
    ]
    # lookback=2 时，tool_call 在第 2 条，超出最后 2 条范围
    assert _has_recent_tool_call(messages, "knowledge_search", look_back=2) is False
    # lookback=10 时能查到
    assert _has_recent_tool_call(messages, "knowledge_search", look_back=10) is True


def test_format_chunks_empty():
    assert _format_chunks([]) == ""


def test_format_chunks_basic():
    """格式化 chunks 为 LLM 可读文本。"""

    class FakeHit:
        def __init__(self, content, heading, title_path, source_name, vector_score, bm25_score):
            self.content = content
            self.heading = heading
            self.title_path = title_path
            self.source_name = source_name
            self.vector_score = vector_score
            self.bm25_score = bm25_score

    hits = [
        FakeHit(
            content="法涅斯是原初之人",
            heading="法涅斯的诞生",
            title_path=["蒙德", "第二幕", "法涅斯的诞生"],
            source_name="test.md",
            vector_score=0.85,
            bm25_score=1.23,
        )
    ]
    text = _format_chunks(hits)
    assert "### 1." in text
    assert "法涅斯的诞生" in text
    assert "test.md" in text
    assert "法涅斯是原初之人" in text
    assert "---" in text


def test_injector_mixin_class():
    """InjectorMixin 类可继承。"""

    from maikb.injector import InjectorMixin

    class FakePlugin(InjectorMixin):
        pass

    assert hasattr(FakePlugin, "hook_auto_inject")
